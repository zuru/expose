import cv2
import os
import argparse
import torch
from typing import Mapping, Tuple, Dict
import json
import numpy as np
import toolz
import tqdm
import glob
import smplx

from collections import defaultdict

from torchvision.transforms import Compose, ToTensor
from expose.data.targets.bbox import BoundingBox
from expose.data.utils.bbox import bbox_to_center_scale

from expose.utils.plot_utils import HDRenderer
from expose.models.expose import ExPose
from expose.utils.img_utils import read_img
from expose.data import transforms as T
from expose.models.camera.camera_projection import WeakPerspectiveCamera

from logging import getLogger

logger = getLogger(__name__)

def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))

class JointMapper(torch.nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)

def load_checkpoint(
    model: torch.nn.Module, 
    filename: str
) -> Mapping[str, torch.Tensor]:
    ckpt = torch.load(filename, map_location='cpu')
    # if 'smplx.head_idxs' in ckpt['model']:
    #     del ckpt['model']['smplx.head_idxs']
    missing, unexpected = model.load_state_dict(
        toolz.keymap(lambda k: k.replace('smplx.', ''), ckpt['model']), 
        strict=False
    )
    if len(missing) > 0:
        logger.warning(f'The following keys were not found: {missing}')
    if len(unexpected):
        logger.warning(
            f'The following keys were not expected: {unexpected}')

def get_means_std() -> Tuple[np.array, np.array]:
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return means, std

def weak_persp_to_blender(
        targets,
        camera_scale,
        camera_transl,
        H, W,
        sensor_width=36,
        focal_length=5000):
    ''' Converts weak-perspective camera to a perspective camera
    '''
    if torch.is_tensor(camera_scale):
        camera_scale = camera_scale.detach().cpu().numpy()
    if torch.is_tensor(camera_transl):
        camera_transl = camera_transl.detach().cpu().numpy()

    output = defaultdict(lambda: [])
    for ii, target in enumerate(targets):
        orig_bbox_size = target.get_field('orig_bbox_size')
        bbox_center = target.get_field('orig_center')
        z = 2 * focal_length / (camera_scale[ii] * orig_bbox_size)

        transl = [
            camera_transl[ii, 0].item(), camera_transl[ii, 1].item(),
            z.item()]
        shift_x = - (bbox_center[0] / W - 0.5)
        shift_y = (bbox_center[1] - 0.5 * H) / W
        focal_length_in_mm = focal_length / W * sensor_width
        output['shift_x'].append(shift_x)
        output['shift_y'].append(shift_y)
        output['transl'].append(transl)
        output['focal_length_in_mm'].append(focal_length_in_mm)
        output['focal_length_in_px'].append(focal_length)
        output['center'].append(bbox_center)
        output['sensor_width'].append(sensor_width)
    for key in output:
        output[key] = np.stack(output[key], axis=0)
    return output

def undo_img_normalization(image, mean, std, add_alpha=True):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy().squeeze()

    out_img = (image * std[np.newaxis, :, np.newaxis, np.newaxis] +
               mean[np.newaxis, :, np.newaxis, np.newaxis])
    if add_alpha:
        out_img = np.pad(
            out_img, [[0, 0], [0, 1], [0, 0], [0, 0]],
            mode='constant', constant_values=1.0)
    return out_img

def read_keypoints(
        filename: str,
        load_hands=True,
        load_face=True,
        load_face_contour=False
    ) -> Dict[str, torch.Tensor]:
        with open(filename) as keypoint_file:
            data = json.load(keypoint_file)
        keypoints, gender_pd, gender_gt = [], [], []
        for person in data['people']:
            body = np.array(person['pose_keypoints_2d'], dtype=np.float32)
            body = body.reshape([-1, 3])
            if load_hands:
                left_hand = np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
                right_hand = np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
                body = np.concatenate([body, left_hand, right_hand], axis=0)
            if load_face:
                face = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
                contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
                if load_face_contour:
                    contour_keyps = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[:17, :]
                body = np.concatenate([body, face, contour_keyps], axis=0)

            gender_pd.append(person.get('gender_pd', None))
            gender_gt.append(person.get('gender_gt', None))
            keypoints.append(torch.from_numpy(body))
        return {
            'keypoints': keypoints,
            'gender_pd': gender_pd,
            'gender_gt': gender_gt, 
        }

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
        
    parser = argparse.ArgumentParser(
        description='ExPose & DensePose Web Demo'
    )
    parser.add_argument('--input_glob', required=True)
    args = parser.parse_args()

    ## ExPose init
    CKPT_FILENAME = './data/checkpoints/model.ckpt'
    smplx_device = torch.device('cuda:0')
    model = ExPose().to(smplx_device)
    load_checkpoint(model, CKPT_FILENAME)
    model = model.eval()

    transform = Compose(
        [ToTensor(), ]
    )
    mean, std = get_means_std()
    transforms = T.Compose([
        T.Crop(crop_size=256, is_train=False,
            scale_factor_max=1.0, scale_factor_min=1.0,
            scale_factor=0.0, scale_dist='normal'
        ),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])     
    
    folder = os.path.dirname(args.input_glob)
    keypoints_glob = os.path.join(folder, 'keypoints', "*.json")
    for img_filename, kpts_filename in tqdm.tqdm(
        zip(glob.glob(args.input_glob), glob.glob(keypoints_glob)),
        desc="ExPose"
    ):
        img = read_img(img_filename)
        data = read_keypoints(kpts_filename)
        with torch.no_grad():            
            scale_factor = 1.2
            H, W = img.shape[:2]

            keypoints = data['keypoints']            
            def _get_area(keypoints: torch.Tensor) -> float:
                min_x = keypoints[..., 0].min()
                min_y = keypoints[..., 1].min()
                max_x = keypoints[..., 0].max()
                max_y = keypoints[..., 1].max()
                return (max_x - min_x) * (max_y - min_y) * keypoints[..., 2].sum()
            keypoints = [max(keypoints, key=_get_area)]
            keypoints = torch.stack(keypoints, dim=0).squeeze()
            x = keypoints[:25, 0]
            y = keypoints[:25, 1]
            x, y = keypoints[:25, :2].split(1, dim=1)
            x = x[torch.nonzero(x)]
            y = y[torch.nonzero(y)]
            x_min, y_min = x.min()[np.newaxis], y.min()[np.newaxis]
            x_max, y_max = x.max()[np.newaxis], y.max()[np.newaxis]
            # extra_w = (scale_factor - 1.0) * 0.5 * (x_max - x_min)
            # extra_h = (scale_factor - 1.0) * 0.5 * (y_max - y_min)
            # x_min = np.clip(x_min - extra_w, 0.0, W)
            # x_max = np.clip(x_max + extra_w, 0.0, W)
            # y_min = np.clip(y_min - extra_h, 0.0, H)
            # y_max = np.clip(y_max + extra_h, 0.0, H)
            bbox = torch.cat([x_min, y_min, x_max, y_max], dim=0)
            center, scale, bbox_size = bbox_to_center_scale(
                bbox, dset_scale_factor=scale_factor
            )
            bbox = BoundingBox(bbox.numpy(), img.shape)
            bbox.add_field('bbox_size', bbox_size.numpy())
            bbox.add_field('orig_bbox_size', bbox_size.numpy())
            bbox.add_field('orig_center', center)
            bbox.add_field('center', center)
            bbox.add_field('scale', scale)
            # ExPose
            img, cropped, target = transforms(img, bbox)
            body = model(
                cropped.unsqueeze(0).cuda(),
                [target], 
                full_imgs=img.unsqueeze(0).cuda()
            )
            stage_n_out = body.get(f'stage_02', {})
            model_vertices = stage_n_out.get('vertices', None)
            faces = stage_n_out['faces']
            model_vertices = model_vertices.detach().cpu().numpy()
            camera_parameters = body.get('camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach()
            camera_transl = camera_parameters['translation'].detach()
            stage_n_out = body.get('final', {})
            final_model_vertices = stage_n_out.get('vertices', None)
            final_model_vertices = final_model_vertices.detach().cpu().numpy()
            camera_parameters = body.get('camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach()
            camera_transl = camera_parameters['translation'].detach()
            
            sensor_width: float = 36
            render_params = weak_persp_to_blender(
                [target],
                camera_scale=camera_scale,
                camera_transl=camera_transl,
                H=H, W=W,
                sensor_width=sensor_width,
                focal_length=5000,
            )
            bg_img = np.clip(undo_img_normalization(
                torch.flip(img, dims=[0]).unsqueeze(0).numpy(), mean, std
            ), 0.0, 1.0)
            renderer = HDRenderer(img_size=256)
            body_fit = renderer(
                model_vertices,
                faces,
                focal_length=render_params['focal_length_in_px'],
                camera_translation=render_params['transl'],
                camera_center=render_params['center'],
                bg_imgs=bg_img,
                return_with_alpha=True,
            )
            expressive_fit = renderer(
                    final_model_vertices,
                    faces,
                    focal_length=render_params['focal_length_in_px'],
                    camera_translation=render_params['transl'],
                    camera_center=render_params['center'],
                    bg_imgs=bg_img,
                    return_with_alpha=True,
                    body_color=[0.4, 0.4, 0.7]
                )
            body_img = body_fit[0].transpose(1, 2, 0)[..., :3] * 255
            expressive_img = expressive_fit[0].transpose(1, 2, 0)[..., :3] * 255
            name, _ = os.path.splitext(os.path.basename(img_filename))
            cv2.imwrite(f"{name}_body.png", body_img.astype(np.uint8))
            cv2.imwrite(f"{name}_expressive.png", expressive_img.astype(np.uint8))
            body_model_expressive = smplx.SMPLX(
                model_path='./data/models/smplx/',
                joint_mapper=JointMapper(smpl_to_openpose()),
                create_global_orient=False,
                create_body_pose=False,
                create_betas=False,
                create_left_hand_pose=False, # create_left_hand_pose,
                create_right_hand_pose=False, # create_right_hand_pose,
                create_expression=False, # create_expression,
                create_jaw_pose=False, # create_jaw_pose,
                create_leye_pose=True, # create_left_eye_pose,
                create_reye_pose=True, # create_right_eye_pose,
                create_transl=False,
                dtype=torch.float32,
                batch_size=1,
                gender='neutral',
                age='adult',
                num_pca_comps=12,
                use_pca=False,
                num_betas=10,
            ).to(smplx_device)

            from expose.utils.rotation_utils import batch_rot2aa

            orig_bbox_size = target.get_field('orig_bbox_size')
            bbox_center = target.get_field('orig_center')
            z = 2 * 5000 / (camera_scale * float(orig_bbox_size))
            # z = 2 * 5000 / (camera_scale * H)
            transl = torch.cat([camera_transl, z], dim=1)
            body_params = body_model_expressive.forward(
                betas=stage_n_out['betas'],
                global_orient=batch_rot2aa(stage_n_out['global_orient'][0]),
                body_pose=batch_rot2aa(stage_n_out['body_pose'][0]).flatten()[np.newaxis, ...],
                left_hand_pose=batch_rot2aa(stage_n_out['left_hand_pose'][0]).flatten()[np.newaxis, ...],
                right_hand_pose=batch_rot2aa(stage_n_out['right_hand_pose'][0]).flatten()[np.newaxis, ...],
                transl=torch.zeros_like(transl),
                expression=stage_n_out['expression'],
                jaw_pose=batch_rot2aa(stage_n_out['jaw_pose'][0]),
                leye_pose=None,
                reye_pose=None,
                return_verts=True,
                return_full_pose=True,
                pose2rot=True,
                return_shaped=True
            )
            
            import pyrender

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0)
            )
            def _create_raymond_lights():
                thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
                phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

                nodes = []

                for phi, theta in zip(phis, thetas):
                    xp = np.sin(theta) * np.cos(phi)
                    yp = np.sin(theta) * np.sin(phi)
                    zp = np.cos(theta)

                    z = np.array([xp, yp, zp])
                    z = z / np.linalg.norm(z)
                    x = np.array([-z[1], z[0], 0.0])
                    if np.linalg.norm(x) == 0:
                        x = np.array([1.0, 0.0, 0.0])
                    x = x / np.linalg.norm(x)
                    y = np.cross(z, x)

                    matrix = np.eye(4)
                    matrix[:3,:3] = np.c_[x,y,z]
                    nodes.append(pyrender.Node(
                        light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                        matrix=matrix
                    ))

                return nodes
            scene = pyrender.Scene(
                bg_color=[0.0, 0.0, 0.0, 0.0],
                ambient_light=(0.3, 0.3, 0.3)
            )
            for light in _create_raymond_lights():
                scene.add_node(light)
            renderer = pyrender.OffscreenRenderer(
                viewport_width=W, viewport_height=H, point_size=1.0
            )
            rotation = np.eye(3)
            translation = transl.detach().cpu().numpy().squeeze()

            import trimesh

            tmesh = trimesh.Trimesh(
                body_params['vertices'].detach().cpu().numpy().squeeze(),
                body_model_expressive.faces,
                process=False
            )
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            tmesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(tmesh, material=material)
            node = scene.add(mesh, 'mesh')

            # Equivalent to 180 degrees around the y-axis. Transforms the fit to
            # OpenGL compatible coordinate system.
            translation[0] *= -1.0
            # translation[1] = 0.8
            wcam = WeakPerspectiveCamera()
            cam_scale = W / float(orig_bbox_size) * camera_scale
            # proj = wcam.forward(body_params['joints'], cam_scale, camera_transl)
            joints = body_params['joints']
            joints = joints + torch.from_numpy(translation).to(joints)
            proj = wcam.forward(joints[..., :2] / joints[..., 2:], cam_scale, camera_transl)
            offset_x = ((keypoints[:, 0] - (W//2)) // (W//2) - proj[0, :, 0].cpu()).mean()
            offset_y = ((keypoints[:, 1] - (H//2)) // (H//2) - proj[0, :, 1].cpu()).mean()            
            # translation[0] += offset_x
            # translation[1] += offset_y

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = rotation
            camera_pose[:3, 3] = translation

            camera = pyrender.camera.IntrinsicsCamera(
                fx=5000, cx=W // 2,
                fy=5000, cy=H // 2,
            )
            cam = scene.add(camera, pose=camera_pose)

            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0

            background = np.clip(undo_img_normalization(img, mean, std, True), 0.0, 1.0)
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = background.squeeze().transpose(1, 2, 0)[..., :3]
            output_img = 255.0 * (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
            output_img = output_img.astype(np.uint8)

            for i in range(25):
                pt1 = tuple(keypoints[i, :2].cpu().numpy().astype(np.int32))
                pt2 = tuple((
                    # proj[0, i, :].cpu().numpy() * np.array([W / 2, H / 2]) + np.array([W / 2, H / 2])
                    proj[0, i, :].cpu().numpy() * np.array([W / 2, H / 2]) + render_params['center'].squeeze()
                ).astype(np.int32))
                cv2.drawMarker(output_img, pt1, (200, 100, 0), cv2.MARKER_DIAMOND, H // 100, H // 1000)
                cv2.drawMarker(output_img, pt2, (0, 100, 200), cv2.MARKER_CROSS, H // 100, H // 1000)
                
            cv2.imwrite(f'{name}_rendered.png', output_img)
            scene.remove_node(node)
            scene.remove_node(cam)
            