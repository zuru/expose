from typing import List, Dict, Tuple, Callable, Optional, Union
from collections import defaultdict

import math
import os
import numpy as np
import torch

from expose.models.attention.hand_predictor_premade import HandPredictor
from expose.models.attention.head_predictor_premade import HeadPredictor
from expose.models.common.networks import FrozenBatchNorm2d

from smplx import build_layer as build_body_model
from smplx.utils import find_joint_kin_chain

from expose.models.backbone.hrnet_premade import HighResolutionNet
from expose.models.common.networks import MLP, IterativeRegression
from expose.models.common.bbox_sampler import CropSampler, ToCrops
from expose.models.common.pose_utils import build_all_pose_params
from expose.models.camera import build_cam_proj, CameraParams
from expose.data.targets import ImageList, ImageListPacked
from expose.data.targets.keypoints import KEYPOINT_NAMES, get_part_idxs
from expose.data.utils import flip_pose
from expose.utils.typing_utils import Tensor

from yacs.config import CfgNode as Cfg

class ExPose(torch.nn.Module):
    def __init__(self,
        body_model_folder: str='./data/models/',
        body_pose_type: str='cont_rot_repr',
        gender: str='neutral', # construction
        global_orient_type: str='cont_rot_repr', # construction
        j14_regressor_path: str='./data/SMPLX_to_J14.pkl',
        jaw_pose_type: str='cont_rot_repr',
        left_hand_pose_type: str='cont_rot_repr',
        flat_hand_mean: bool=False,
        num_pca_comps: int=12,
        mean_pose_path: str='./data/all_means.pkl',
        num_betas: int=10,
        num_expression_coeffs: int=10,
        right_hand_pose_type: str='cont_rot_repr',
        shape_mean_path: str='./data/shape_mean.npy',
        use_compressed: bool=True,
        use_face_contour: bool=True,
        use_face_keypoints: bool=True,
        use_feet_keypoints: bool=True,
        hand_crop_size: int=224,
        hand_scale_factor: float=3.0,
        head_crop_size: int=256,
        head_scale_factor: float=2.0,
        focal_length: float=5000.0,
        num_head_betas: int=100,
        num_head_expression_coeffs: int=50,
    ) -> None:
        super(ExPose, self).__init__()
        self.focal_length = focal_length
        self.num_stages = 3  
        self.pose_last_stage = True
        self.body_model_cfg = Cfg({
            'body_pose': {'param_type': body_pose_type},
            'gender': gender,
            'global_orient': {'param_type': global_orient_type},
            'j14_regressor_path': j14_regressor_path,
            'jaw_pose': {'param_type': jaw_pose_type},
            'left_hand_pose': {
                'param_type': left_hand_pose_type,
                'flat_hand_mean': flat_hand_mean,
                'num_pca_comps': num_pca_comps,
            },
            'mean_pose_path': mean_pose_path,
            'model_folder': body_model_folder,
            'num_betas': num_betas,
            'num_expression_coeffs': num_expression_coeffs,
            'right_hand_pose': {
                'param_type': right_hand_pose_type,
                'flat_hand_mean': flat_hand_mean,
                'num_pca_comps': num_pca_comps,
            },
            'shape_mean_path': shape_mean_path,
            'use_compressed': use_compressed,
            'use_face_contour': use_face_contour,
            'use_face_keypoints': use_face_keypoints,
            'use_feet_keypoints': use_feet_keypoints,
        })
        self.body_model = build_body_model(
            body_model_folder, model_type='smplx',
            dtype=torch.float32, **self.body_model_cfg
        )
        self.num_betas = num_betas
        if os.path.exists(shape_mean_path):
            shape_mean = torch.from_numpy(
                np.load(shape_mean_path, allow_pickle=True)
            ).to(dtype=torch.float32).reshape(1, -1)[:, :num_betas].reshape(-1)
        else:
            shape_mean = torch.zeros([num_betas], dtype=torch.float32)        
        self.num_expression_coeffs = num_expression_coeffs
        expression_mean = torch.zeros(
            [num_expression_coeffs], dtype=torch.float32)
        # Build the pose parameterization for all the parameters
        pose_desc_dict = build_all_pose_params(
            self.body_model_cfg, 0, self.body_model
        )
        self.global_orient_decoder = pose_desc_dict['global_orient'].decoder
        global_orient_mean = pose_desc_dict['global_orient'].mean
        # Rotate the model 180 degrees around the x-axis
        if global_orient_type == 'aa':
            global_orient_mean[0] = math.pi
        elif global_orient_type == 'cont_rot_repr':
            global_orient_mean[3] = -1
        global_orient_dim = pose_desc_dict['global_orient'].dim
        self.body_pose_decoder = pose_desc_dict['body_pose'].decoder
        body_pose_mean = pose_desc_dict['body_pose'].mean
        body_pose_dim = pose_desc_dict['body_pose'].dim
        self.left_hand_pose_decoder = pose_desc_dict['left_hand_pose'].decoder
        left_hand_pose_mean = pose_desc_dict['left_hand_pose'].mean
        left_hand_pose_dim = pose_desc_dict['left_hand_pose'].dim
        self.right_hand_pose_decoder = pose_desc_dict[
            'right_hand_pose'].decoder
        right_hand_pose_mean = pose_desc_dict['right_hand_pose'].mean
        right_hand_pose_dim = pose_desc_dict['right_hand_pose'].dim
        self.jaw_pose_decoder = pose_desc_dict['jaw_pose'].decoder
        jaw_pose_mean = pose_desc_dict['jaw_pose'].mean
        jaw_pose_dim = pose_desc_dict['jaw_pose'].dim
        ''' CREATE MEANS '''
        mean_lst = []
        start = 0
        # global orient
        global_orient_idxs = list(range(start, start + global_orient_dim))
        global_orient_idxs = torch.tensor(global_orient_idxs, dtype=torch.long)
        self.register_buffer('global_orient_idxs', global_orient_idxs)
        start += global_orient_dim
        mean_lst.append(global_orient_mean.view(-1))
        # body pose
        body_pose_idxs = list(range(start, start + body_pose_dim))
        self.register_buffer('body_pose_idxs', torch.tensor(body_pose_idxs, dtype=torch.long))
        start += body_pose_dim
        mean_lst.append(body_pose_mean.view(-1))
        # left hand
        left_hand_pose_idxs = list(range(start, start + left_hand_pose_dim))
        self.register_buffer('left_hand_pose_idxs', torch.tensor(left_hand_pose_idxs, dtype=torch.long))
        start += left_hand_pose_dim
        mean_lst.append(left_hand_pose_mean.view(-1))
        # right hand
        right_hand_pose_idxs = list(range(
            start, start + right_hand_pose_dim))
        self.register_buffer('right_hand_pose_idxs', torch.tensor(right_hand_pose_idxs, dtype=torch.long))
        start += right_hand_pose_dim
        mean_lst.append(right_hand_pose_mean.view(-1))
        # jaw
        jaw_pose_idxs = list(range(
            start, start + jaw_pose_dim))
        self.register_buffer('jaw_pose_idxs', torch.tensor(jaw_pose_idxs, dtype=torch.long))
        start += jaw_pose_dim
        mean_lst.append(jaw_pose_mean.view(-1))
        # shape
        shape_idxs = list(range(start, start + num_betas))
        self.register_buffer('shape_idxs', torch.tensor(shape_idxs, dtype=torch.long))
        start += num_betas
        mean_lst.append(shape_mean.view(-1))
        # expression
        expression_idxs = list(range(
            start, start + num_expression_coeffs))
        self.register_buffer('expression_idxs', torch.tensor(expression_idxs, dtype=torch.long))
        start += num_expression_coeffs
        mean_lst.append(expression_mean.view(-1))
        # camera
        camera_cfg = {
            'perspective': {
                'focal_length': focal_length,
                'regress_focal_length': False,
                'regress_rotation': False,
                'regress_translation': False,
            },
            'pos_func': 'softplus',
            'type': 'weak-persp',
            'weak_persp': {
                'mean_scale': 0.9,
                'regress_scale': True,
                'regress_translation': True,
            },
        }
        camera_data = build_cam_proj(camera_cfg, dtype=torch.float32)
        self.projection = camera_data['camera']
        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        #  self.camera_mean = camera_mean
        self.register_buffer('camera_mean', camera_mean)
        self.camera_scale_func = camera_data['scale_func']
        camera_idxs = list(range(
            start, start + camera_param_dim))
        self.register_buffer('camera_idxs', torch.tensor(camera_idxs, dtype=torch.long))
        mean_lst.append(camera_mean)
        # merge
        param_mean = torch.cat(mean_lst).view(1, -1)
        param_dim = param_mean.numel()

        # Construct the feature extraction backbone
        self.backbone = HighResolutionNet()
        feat_dims = self.backbone.get_output_dim()

        self.body_feature_key = 'concat'
        feat_dim = feat_dims[self.body_feature_key]
        regressor_cfg = {
            'activ_type': 'none', 'bias_init': 0.0,
            'dropout': 0.5, 'gain': 0.01,
            'init_type': 'xavier', 'layers': [1024, 1024],
            'lrelu_slope': 0.2, 'norm_type': 'none',
            'num_groups': 3,
        }
        regressor = MLP(feat_dim + param_dim, param_dim, **regressor_cfg)
        self.regressor = IterativeRegression(regressor, param_mean, num_stages=3)

        # Find the kinematic chain for the right wrist        
        self.right_wrist_idx = KEYPOINT_NAMES.index('right_wrist')        
        self.left_wrist_idx = KEYPOINT_NAMES.index('left_wrist')

        self.hand_predictor = HandPredictor(
            camera_cfg,
            pose_desc_dict['global_orient'],
            pose_desc_dict['right_hand_pose'],
            camera_data,
            detach_mean=False,
            append_params=True,
            num_stages=3,
            dtype=torch.float32
        )

        self.hand_scale_factor = hand_scale_factor
        self.hand_crop_size = hand_crop_size
        self.hand_cropper = CropSampler(hand_crop_size)

        self.head_crop_size = head_crop_size
        self.head_scale_factor = head_scale_factor
        self.head_cropper = CropSampler(head_crop_size)

        camera_cfg['weak_persp']['mean_scale'] = 8.0
        self.head_predictor = HeadPredictor(
            camera_cfg,            
            pose_desc_dict['global_orient'],
            pose_desc_dict['jaw_pose'], camera_data,
            detach_mean=False,
            dtype=torch.float32,
            num_betas=num_head_betas,
            num_expression_coeffs=num_head_expression_coeffs,
        )
        self.points_to_crops = ToCrops()
        right_wrist_kin_chain = find_joint_kin_chain(
            self.right_wrist_idx,
            self.body_model.parents)
        right_wrist_kin_chain = torch.tensor(
            right_wrist_kin_chain, dtype=torch.long)
        self.register_buffer('right_wrist_kin_chain', right_wrist_kin_chain)
        self.register_buffer('abs_pose_mean', self.global_orient_decoder.get_mean().unsqueeze(dim=0))
        # Find the kinematic chain for the left wrist
        left_wrist_kin_chain = find_joint_kin_chain(self.left_wrist_idx,
            self.body_model.parents
        )
        left_wrist_kin_chain = torch.tensor(left_wrist_kin_chain, dtype=torch.long)
        self.register_buffer('left_wrist_kin_chain', left_wrist_kin_chain)
        # Find the kinematic chain for the neck
        neck_idx = KEYPOINT_NAMES.index('neck')
        neck_kin_chain = find_joint_kin_chain(
            neck_idx, self.body_model.parents
        )
        self.register_buffer('neck_kin_chain', torch.tensor(neck_kin_chain, dtype=torch.long))

        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        head_idxs = idxs_dict['head']

        self.register_buffer('body_idxs', torch.tensor(body_idxs))
        self.register_buffer('left_hand_idxs', torch.tensor(left_hand_idxs))
        self.register_buffer('right_hand_idxs', torch.tensor(right_hand_idxs))
        self.register_buffer('head_idxs', torch.tensor(head_idxs))

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.regressor.parameters():
            param.requires_grad = False
        # Stop updating batch norm statistics
        self.backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(self.backbone)
        self.regressor = FrozenBatchNorm2d.convert_frozen_batchnorm(self.regressor)
        # Build part merging functions
        self.right_hand_pose_merging_func = self._merge_func
        self.left_hand_pose_merging_func = self._merge_func
        self.left_wrist_pose_merging_func = self._merge_func
        self.right_wrist_pose_merging_func = self._merge_func
        self.jaw_pose_merging_func = self._merge_func
        self.expression_merging_func = self._merge_func

    def _merge_func(self,
        part: Tensor,
    ) -> Dict[str, Tensor]:
        return {
            'merged' : part,
            'weights': None,
        }

    def flat_body_params_to_dict(self, param_tensor):
        global_orient = torch.index_select(
            param_tensor, 1, self.global_orient_idxs)
        body_pose = torch.index_select(
            param_tensor, 1, self.body_pose_idxs)
        left_hand_pose = torch.index_select(
            param_tensor, 1, self.left_hand_pose_idxs)
        right_hand_pose = torch.index_select(
            param_tensor, 1, self.right_hand_pose_idxs)
        jaw_pose = torch.index_select(
            param_tensor, 1, self.jaw_pose_idxs)
        betas = torch.index_select(param_tensor, 1, self.shape_idxs)
        expression = torch.index_select(param_tensor, 1, self.expression_idxs)

        return {
            'betas': betas,
            'expression': expression,
            'global_orient': global_orient,
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
        }

    def find_joint_global_rotation(
            self,
            kin_chain: Tensor,
            root_pose: Tensor,
            body_pose: Tensor
    ) -> Tensor:
        ''' Computes the absolute rotation of a joint from the kinematic chain
        '''
        # Create a single vector with all the poses
        parents_pose = torch.cat(
            [root_pose, body_pose], dim=1)[:, kin_chain]
        output_pose = parents_pose[:, 0]
        for idx in range(1, parents_pose.shape[1]):
            output_pose = torch.bmm(
                parents_pose[:, idx], output_pose)
        return output_pose

    def build_hand_mean(self, global_orient: Tensor,
                        body_pose: Tensor,
                        betas: Tensor,
                        flipped_left_hand_pose: Tensor,
                        right_hand_pose: Tensor,
                        hand_targets: List,
                        num_body_imgs: int = 0,
                        num_hand_imgs: int = 0
                        ) -> Tuple[Tensor, Tensor]:
        ''' Builds the initial point for the iterative regressor of the hand
        '''
        device, dtype = global_orient.device, global_orient.dtype
        hand_only_mean, parent_rots = [], []
        if num_body_imgs > 0:
            batch_size = num_body_imgs
            # Compute the absolute pose of the right wrist
            right_wrist_pose_abs = self.find_joint_global_rotation(
                self.right_wrist_kin_chain, global_orient,
                body_pose)

            right_wrist_parent_rot = self.find_joint_global_rotation(
                self.right_wrist_kin_chain[1:], global_orient,
                body_pose)

            left_wrist_parent_rot = self.find_joint_global_rotation(
                self.left_wrist_kin_chain[1:], global_orient, body_pose)
            left_to_right_wrist_parent_rot = flip_pose(
                left_wrist_parent_rot, pose_format='rot-mat')

            parent_rots += [
                right_wrist_parent_rot, left_to_right_wrist_parent_rot]

            #  if self.condition_hand_on_body:
            # Convert the absolute pose to the latent representation
            right_wrist_pose = self.global_orient_decoder.encode(
                right_wrist_pose_abs.unsqueeze(dim=1)).reshape(
                    batch_size, -1)

            # Compute the absolute rotation for the left wrist
            left_wrist_pose_abs = self.find_joint_global_rotation(
                self.left_wrist_kin_chain, global_orient, body_pose)
            # Flip the left wrist to the right
            left_to_right_wrist_pose = flip_pose(
                left_wrist_pose_abs, pose_format='rot-mat')
            # Convert to the latent representation
            left_to_right_wrist_pose = self.global_orient_decoder.encode(
                left_to_right_wrist_pose.unsqueeze(dim=1)).reshape(
                    batch_size, -1)

            # Convert the pose of the left hand to the right hand and project
            # it to the encoder space
            left_to_right_hand_pose = self.right_hand_pose_decoder.encode(
                flipped_left_hand_pose).reshape(batch_size, -1)

            camera_mean = self.hand_predictor.get_camera_mean().expand(
                batch_size, -1)

            shape_condition = self.hand_predictor.get_shape_mean(batch_size)
            right_finger_pose_condition = right_hand_pose 
            right_hand_mean = torch.cat(
                [
                    right_wrist_pose, right_finger_pose_condition,
                    shape_condition, camera_mean,
                ], dim=1)
            left_finger_pose_condition = left_to_right_hand_pose
            # Should be Bx31
            left_hand_mean = torch.cat(
                [
                    left_to_right_wrist_pose,
                    left_finger_pose_condition,
                    shape_condition,
                    camera_mean,
                ], dim=1
            )
            hand_only_mean += [right_hand_mean, left_hand_mean]

        if num_hand_imgs > 0:
            mean_param = self.hand_predictor.get_param_mean(batch_size=num_hand_imgs)

            hand_only_mean.append(mean_param)
            hand_only_parent_rots = torch.eye(
                3, device=device, dtype=dtype).reshape(
                    1, 3, 3).expand(num_hand_imgs, -1, -1).clone()
            hand_only_parent_rots[:, 1, 1] = -1
            hand_only_parent_rots[:, 2, 2] = -1
            parent_rots.append(hand_only_parent_rots)

        hand_only_mean = torch.cat(hand_only_mean, dim=0)
        parent_rots = torch.cat(parent_rots, dim=0)
        return hand_only_mean, parent_rots

    def build_head_mean(
        self,
        global_orient: Tensor,
        body_pose: Tensor,
        betas: Tensor,
        expression: Tensor,
        jaw_pose: Tensor,
        head_targets: List,
        num_body_imgs: int = 0,
        num_head_imgs: int = 0
    ) -> Tensor:
        ''' Builds the initial point of the head regressor
        '''
        head_only_mean = []
        if num_body_imgs > 0:
            batch_size = num_body_imgs

            # Compute the absolute pose of the right wrist
            neck_pose_abs = self.find_joint_global_rotation(
                self.neck_kin_chain, global_orient, body_pose)
            # Convert the absolute neck pose to offsets
            neck_latent = self.global_orient_decoder.encode(
                neck_pose_abs.unsqueeze(dim=1))
            neck_pose = neck_latent.reshape(batch_size, -1)

            camera_mean = self.head_predictor.get_camera_mean(
                batch_size=batch_size)

            neck_pose_condition = self.head_predictor.get_neck_pose_mean(batch_size)
            jaw_pose_condition = jaw_pose.reshape(batch_size, -1)
                
            head_num_betas = self.head_predictor.get_num_betas()
            shape_padding_size = head_num_betas - self.num_betas
            betas_condition = self.head_predictor.get_shape_mean(batch_size=batch_size)

            head_num_expression_coeffs = (
                self.head_predictor.get_num_expression_coeffs())
            expr_padding_size = (head_num_expression_coeffs -
                                 self.num_expression_coeffs)
            expression_condition = torch.nn.functional.pad(
                expression.reshape(batch_size, -1), (0, expr_padding_size)
            )

            # Should be Bx(Head pose params)
            head_only_mean.append(torch.cat(
                [neck_pose_condition, jaw_pose_condition,
                 betas_condition, expression_condition,
                 camera_mean.reshape(batch_size, -1),
                 ], dim=1
            ))

        if num_head_imgs > 0:
            mean_param = self.head_predictor.get_param_mean(batch_size=num_head_imgs)
            head_only_mean.append(mean_param)

        head_only_mean = torch.cat(head_only_mean, dim=0)
        return head_only_mean


    #########################################
    #                                       #
    #            F O R W A R D              #
    #                                       #
    #########################################



    def forward(self,
        images: Tensor,
        targets: List = None,
        hand_imgs: Optional[Tensor] = None,
        hand_targets: Optional[List] = None,
        head_imgs: Optional[Tensor] = None,
        head_targets: Optional[List] = None,
        full_imgs: Optional[Union[ImageList, ImageListPacked]] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        ''' Forward pass of the attention predictor
        '''
        batch_size, _, crop_size, _ = images.shape
        device = images.device
        dtype = images.dtype

        feat_dict = self.backbone(images)
        body_features = feat_dict[self.body_feature_key]

        body_parameters, body_deltas = self.regressor(body_features)

        # A list of dicts for the parameters predicted at each stage. The key
        # is the name of the parameters and the value is the prediction of the
        # model at the i-th stage of the iteration
        param_dicts = []
        # A dict of lists. Each key is the name of the parameter and the
        # corresponding item is a list of offsets that are predicted by the
        # model
        deltas_dict = defaultdict(lambda: [])
        param_delta_iter = zip(body_parameters, body_deltas)
        for idx, (params, deltas) in enumerate(param_delta_iter):
            curr_params_dict = self.flat_body_params_to_dict(params)

            out_dict = {}
            for key, val in curr_params_dict.items():
                if hasattr(self, f'{key}_decoder'):
                    decoder = getattr(self, f'{key}_decoder')
                    out_dict[key] = decoder(val)
                    out_dict[f'raw_{key}'] = val.clone()
                else:
                    out_dict[key] = val

            param_dicts.append(out_dict)
            curr_params_dict.clear()
            for key, val in self.flat_body_params_to_dict(deltas).items():
                deltas_dict[key].append(val)

        for key in deltas_dict:
            deltas_dict[key] = torch.stack(deltas_dict[key], dim=1).sum(dim=1)

        if self.pose_last_stage:
            merged_params = param_dicts[-1]
        else:
            merged_params = {}
            for key in param_dicts[0].keys():
                param = []
                for idx in range(self.num_stages):
                    if param_dicts[idx][key] is None:
                        continue
                    param.append(param_dicts[idx][key])
                merged_params[key] = torch.cat(param, dim=0)

        # Compute the body surface using the current estimation of the pose and
        # the shape
        body_model_output = self.body_model(
            get_skin=True, return_shaped=True, **merged_params)

        # Split the vertices, joints, etc. to stages
        out_params = defaultdict(lambda: dict())
        for key in body_model_output:
            if torch.is_tensor(body_model_output[key]):
                curr_val = body_model_output[key]
                out_list = torch.split(
                    curr_val, batch_size, dim=0)
                # If the number of outputs is equal to the number of stages
                # then store each stage
                if len(out_list) == self.num_stages:
                    for idx in range(len(out_list)):
                        out_params[f'stage_{idx:02d}'][key] = out_list[idx]
                # Else add only the last
                else:
                    out_key = f'stage_{self.num_stages - 1:02d}'
                    out_params[out_key][key] = out_list[-1]

        # Add the predicted parameters to the output dictionary
        for stage in range(self.num_stages):
            stage_key = f'stage_{stage:02d}'
            if len(out_params[stage_key]) < 1:
                continue
            out_params[stage_key].update(param_dicts[stage])
            out_params[stage_key]['faces'] = self.body_model.faces

        global_orient_from_body_net = param_dicts[-1]['global_orient'].clone()
        body_pose_from_body_net = param_dicts[-1]['body_pose'].clone()

        raw_body_pose_from_body_net = param_dicts[-1]['raw_body_pose'].clone(
        ).reshape(batch_size, 21, -1)
        left_hand_pose = param_dicts[-1]['left_hand_pose'].clone()
        right_hand_pose = param_dicts[-1]['right_hand_pose'].clone()
        jaw_pose = param_dicts[-1]['jaw_pose'].clone()

        # Extract the camera parameters estimated by the body only image
        camera_params = torch.index_select(
            body_parameters[-1], 1, self.camera_idxs)
        scale = camera_params[:, 0].view(-1, 1)
        translation = camera_params[:, 1:3]
        # Pass the predicted scale through exp() to make sure that the
        # scale values are always positive
        scale = self.camera_scale_func(scale)

        # Extract the final shape and expression parameters predicted by the
        # body only model
        betas = param_dicts[-1].get('betas').clone()
        expression = param_dicts[-1].get('expression')

        # Project the joints on the image plane
        proj_joints = self.projection(
            out_params[f'stage_{self.num_stages - 1:02d}']['joints'],
            scale=scale, translation=translation)

        # Add the projected joints
        out_params['proj_joints'] = proj_joints
        # the number of stages
        out_params['num_stages'] = self.num_stages
        # and the camera parameters to the output
        out_params['camera_parameters'] = CameraParams(
            translation=translation, scale=scale)

        # Clone the body pose so that we can update it with the predicted
        # sub-parts        
        final_body_pose = raw_body_pose_from_body_net.clone()

        hand_predictions, head_predictions = {}, {}
        num_hand_imgs = 0
        # Get the left, right and head crops from the full body
        left_hand_joints = (
            (torch.index_select(proj_joints, 1, self.left_hand_idxs) *
                0.5 + 0.5) * crop_size)
        #  left_hand_joints = torch.index_select(
        #  proj_joints, 1, self.left_hand_idxs)
        left_hand_points_to_crop = self.points_to_crops(
            full_imgs, left_hand_joints, targets,
            scale_factor=self.hand_scale_factor, crop_size=crop_size, # 1.2 / 256
        )
        left_hand_center = left_hand_points_to_crop['center']
        left_hand_orig_bbox_size = left_hand_points_to_crop[
            'orig_bbox_size']
        left_hand_inv_crop_transforms = left_hand_points_to_crop[
            'inv_crop_transforms']

        left_hand_cropper_out = self.hand_cropper(
            full_imgs, left_hand_center, left_hand_orig_bbox_size)
        left_hand_crops = left_hand_cropper_out['images']
        left_hand_points = left_hand_cropper_out['sampling_grid']
        left_hand_crop_transform = left_hand_cropper_out['transform']

        right_hand_joints = (torch.index_select(
            proj_joints, 1, self.right_hand_idxs) * 0.5 + 0.5) * crop_size
        right_hand_points_to_crop = self.points_to_crops(
            full_imgs, right_hand_joints, targets,
            scale_factor=self.hand_scale_factor, crop_size=crop_size,
        )
        right_hand_center = right_hand_points_to_crop['center']
        right_hand_orig_bbox_size = right_hand_points_to_crop[
            'orig_bbox_size']
        right_hand_bbox_size = right_hand_points_to_crop['bbox_size']

        right_hand_cropper_out = self.hand_cropper(# [481.1823, 622.9359]] / 132.1798]
            full_imgs, right_hand_center, right_hand_orig_bbox_size)
        right_hand_crops = right_hand_cropper_out['images']
        right_hand_points = right_hand_cropper_out['sampling_grid']
        right_hand_crop_transform = right_hand_cropper_out['transform']

        # Store the transformation parameters
        out_params['left_hand_crops'] = left_hand_crops.detach()
        out_params['left_hand_points'] = left_hand_points.detach()
        out_params['right_hand_crops'] = right_hand_crops.detach()
        out_params['right_hand_points'] = right_hand_points.detach()

        out_params['right_hand_crop_transform'] = (
            right_hand_crop_transform.detach())
        out_params['left_hand_crop_transform'] = (
            left_hand_crop_transform.detach())

        out_params['left_hand_hd_to_crop'] = (
            left_hand_cropper_out['hd_to_crop'])
        out_params['left_hand_inv_crop_transforms'] = (
            left_hand_points_to_crop['inv_crop_transforms'])

        out_params['right_hand_hd_to_crop'] = (
            right_hand_cropper_out['hd_to_crop'])
        out_params['right_hand_inv_crop_transforms'] = (
            right_hand_points_to_crop['inv_crop_transforms'])

        # Flip the left hand to a right hand
        all_hand_imgs = []
        hand_global_orient = []
        hand_body_pose = []
        all_hand_imgs.append(right_hand_crops)
        all_hand_imgs.append(torch.flip(left_hand_crops, dims=(-1,)))
        hand_global_orient += [
            global_orient_from_body_net,
            flip_pose(
                global_orient_from_body_net, pose_format='rot-mat')]
        hand_body_pose += [
            body_pose_from_body_net, body_pose_from_body_net]

        if hand_imgs is not None:
            # Add the hand only images
            num_hand_imgs = len(hand_imgs)
            all_hand_imgs.append(hand_imgs)

            body_identity = torch.eye(
                3, device=device, dtype=dtype).reshape(1, 1, 3, 3).expand(
                    num_hand_imgs, body_pose_from_body_net.shape[1], -1,
                    -1)
            hand_body_pose.append(body_identity)
            global_identity = torch.eye(
                3, device=device, dtype=dtype).reshape(
                    1, 1, 3, 3).expand(
                        num_hand_imgs,
                        global_orient_from_body_net.shape[1], -1, -1).clone()
            global_identity[:, :, 1, 1] = -1
            global_identity[:, :, 2, 2] = -1
            hand_global_orient.append(global_identity)

        num_body_imgs = batch_size
        num_hand_net_ins = len(hand_body_pose) + num_body_imgs
        if num_hand_net_ins > 0:
            hand_body_pose = torch.cat(hand_body_pose, dim=0)
            hand_global_orient = torch.cat(hand_global_orient, dim=0)

            # Flip the pose of the left hand
            flipped_left_hand_pose = flip_pose(
                param_dicts[-1]['left_hand_pose'], pose_format='rot-mat')

            # Build the mean used to condition the hand network using the
            # parameters estimated by the body network
            hand_mean, parent_rots = self.build_hand_mean(
                param_dicts[-1]['global_orient'],
                param_dicts[-1]['body_pose'],
                betas=param_dicts[-1]['betas'],
                flipped_left_hand_pose=flipped_left_hand_pose,
                right_hand_pose=param_dicts[-1]['raw_right_hand_pose'],
                hand_targets=hand_targets,
                num_body_imgs=num_body_imgs,
                num_hand_imgs=num_hand_imgs,
            )

            # Feed the hand images and the offsets to the hand-only
            # predictor
            all_hand_imgs = torch.cat(all_hand_imgs, dim=0)

            hand_predictions = self.hand_predictor(
                all_hand_imgs,
                hand_mean=hand_mean,
                global_orient_from_body_net=hand_global_orient,
                body_pose_from_body_net=hand_body_pose,
                parent_rots=parent_rots,
                num_hand_imgs=num_hand_imgs,
            )
            num_hand_stages = hand_predictions.get('num_stages', 1)
            hand_network_output = hand_predictions.get(
                f'stage_{num_hand_stages - 1:02d}')

        # Find which images belong to the left hand and which ones to
        # the right hand
        hands_from_body_idxs = torch.arange(
            0, 2 * batch_size, dtype=torch.long, device=device)
        right_hand_from_body_idxs = hands_from_body_idxs[
            :batch_size]
        left_hand_from_body_idxs = hands_from_body_idxs[batch_size:]

        raw_right_hand_pose_dict = self.right_hand_pose_merging_func(
            part=hand_network_output.get(
                'raw_right_hand_pose')[right_hand_from_body_idxs],
        )
        raw_right_hand_pose = raw_right_hand_pose_dict['merged']

        right_wrist_pose_from_part = hand_network_output.get(
            'raw_right_wrist_pose')                    
        raw_right_wrist_pose_dict = (
            self.right_wrist_pose_merging_func(
                part=right_wrist_pose_from_part,
            )
        )
        raw_right_wrist_pose = raw_right_wrist_pose_dict['merged']
        final_body_pose[:, self.right_wrist_idx - 1] = (
            raw_right_wrist_pose)

        # Project the flipped left hand pose to the rotation latent
        # space using the decoder for the right hand
        raw_left_to_right_hand_pose = (
            self.right_hand_pose_decoder.encode(
                flipped_left_hand_pose).reshape(batch_size, -1))
        # Convert the pose of the left hand to the right hand and
        # project it to the encoder space
        # Merge the predictions of the body network and the part
        # network for the articulation of the left hand
        left_hand_pose_from_part = hand_network_output.get(
            'raw_right_hand_pose')[left_hand_from_body_idxs]
        raw_left_to_right_hand_pose_dict = (
            self.left_hand_pose_merging_func(
                part=left_hand_pose_from_part,
            )
        )
        raw_left_to_right_hand_pose = raw_left_to_right_hand_pose_dict[
            'merged']

        left_wrist_pose_from_part = hand_network_output.get(
            'raw_left_wrist_pose')
        raw_left_wrist_pose_dict = (
            self.left_wrist_pose_merging_func(
                part=left_wrist_pose_from_part,
            )
        )
        raw_left_wrist_pose = raw_left_wrist_pose_dict['merged']
        final_body_pose[:, self.left_wrist_idx - 1] = (
            raw_left_wrist_pose)

        right_hand_pose = self.right_hand_pose_decoder(
            raw_right_hand_pose)
        # Decode the predicted pose and flip it back to the left hand
        # space
        left_hand_pose = flip_pose(self.right_hand_pose_decoder(
            raw_left_to_right_hand_pose), pose_format='rot-mat')

        num_head_imgs = 0
        head_joints = (torch.index_select(
            proj_joints, 1, self.head_idxs) * 0.5 + 0.5) * crop_size
        #  head_joints = torch.index_select(
        #  proj_joints, 1, self.head_idxs)
        head_point_to_crop_output = self.points_to_crops(
            full_imgs, head_joints, targets,
            scale_factor=self.head_scale_factor, crop_size=crop_size,
        )
        head_center = head_point_to_crop_output['center']
        head_orig_bbox_size = head_point_to_crop_output[
            'orig_bbox_size']
        head_inv_crop_transforms = head_point_to_crop_output[
            'inv_crop_transforms']

        head_cropper_out = self.head_cropper(
            full_imgs, head_center, head_orig_bbox_size)
        head_crops = head_cropper_out['images']
        head_points = head_cropper_out['sampling_grid']
        # Contains the transformation that is used to transform the
        # sampling grid from head image coordinates to HD image
        # coordinates.
        head_crop_transform = head_cropper_out['transform']

        out_params['head_crops'] = head_crops.detach()
        out_params['head_points'] = head_points.detach()
        out_params['head_crop_transform'] = (
            head_crop_transform.detach())

        out_params['head_hd_to_crop'] = head_cropper_out['hd_to_crop']
        out_params['head_inv_crop_transforms'] = (
            head_point_to_crop_output['inv_crop_transforms'])

        all_head_imgs = []
        all_head_imgs.append(head_crops)

        # The global and body pose data used to pose the model inside the
        # head-only sub-network.
        head_global_orient, head_body_pose = [], []
        head_global_orient += [global_orient_from_body_net]
        head_body_pose += [body_pose_from_body_net]

        if head_imgs is not None:
            all_head_imgs.append(head_imgs)
            num_head_imgs = len(head_imgs)
            body_identity = torch.eye(
                3, device=device, dtype=dtype).reshape(
                    1, 1, 3, 3).expand(
                        num_head_imgs, body_pose_from_body_net.shape[1],
                        -1, -1)
            head_body_pose.append(body_identity)
            global_identity = torch.eye(
                3, device=device, dtype=dtype).reshape(
                    1, 1, 3, 3).expand(num_head_imgs, -1, -1, -1).clone()
            global_identity[:, :, 1, 1] = -1
            global_identity[:, :, 2, 2] = -1
            head_global_orient.append(global_identity)

        num_body_imgs = batch_size
        num_head_net_ins = len(head_global_orient) + num_body_imgs
        if num_head_net_ins > 0:
            head_global_orient = torch.cat(head_global_orient, dim=0)
            head_body_pose = torch.cat(head_body_pose, dim=0)

            head_mean = self.build_head_mean(
                param_dicts[-1]['global_orient'],
                param_dicts[-1]['body_pose'],
                betas=param_dicts[-1]['betas'],
                expression=param_dicts[-1]['expression'],
                jaw_pose=param_dicts[-1]['raw_jaw_pose'],
                num_head_imgs=num_head_imgs,
                num_body_imgs=num_body_imgs,
                head_targets=head_targets,
            )
            all_head_imgs = torch.cat(all_head_imgs, dim=0)

            head_predictions = self.head_predictor(
                all_head_imgs,
                head_mean=head_mean,
                global_orient_from_body_net=head_global_orient,
                body_pose_from_body_net=head_body_pose,
                num_head_imgs=num_head_imgs,
            )

            num_head_stages = head_predictions.get('num_stages', 1)
            head_network_output = head_predictions.get(
                f'stage_{num_head_stages - 1:02d}')
            head_from_body_idxs = torch.arange(
                0, batch_size, dtype=torch.long, device=device)                    
            # Replace the jaw pose only from the predictions taken from
            # valid head crops
            raw_jaw_pose_from_part = head_network_output.get(
                'raw_jaw_pose')[head_from_body_idxs]
            raw_jaw_pose_dict = self.jaw_pose_merging_func(
                part=raw_jaw_pose_from_part,
            )
            raw_jaw_pose = raw_jaw_pose_dict['merged']

            expression_from_head = head_network_output.get(
                'expression')[head_from_body_idxs,
                                :self.num_expression_coeffs]
            expression_dict = self.expression_merging_func(
                part=expression_from_head,
            )
            expression = expression_dict['merged']
            jaw_pose = self.jaw_pose_decoder(raw_jaw_pose)

        
        body_pose = self.body_pose_decoder(
            final_body_pose.reshape(batch_size, -1))        

        final_body_parameters = {
            'global_orient': param_dicts[-1].get('global_orient'),
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
            'betas': betas,
            'expression': expression
        }

        # Compute the mesh using the new hand and face parameters
        final_body_model_output = self.body_model(
            get_skin=True, return_shaped=True, **final_body_parameters)
        param_dicts.append({
            **final_body_parameters, **final_body_model_output})

        out_params['final'] = {
            **final_body_parameters, **final_body_model_output}
        joints3d = final_body_model_output.get('joints')
        proj_joints = self.projection(
            joints3d, scale=scale, translation=translation)
        out_params['final_proj_joints'] = proj_joints
        # Update the camera parameters with the new projected joints
        out_params['proj_joints'] = proj_joints
        out_params['final']['proj_joints'] = proj_joints

        body_crop_size = images.shape[2]
        # Convert the projected joints from [-1, 1] to body image
        # coordinates
        proj_joints_in_body_crop = (
            proj_joints * 0.5 + 0.5) * body_crop_size

        # Transform the projected points back to the HD image
        hd_proj_joints = torch.einsum(
            'bij,bkj->bki',
            [head_inv_crop_transforms[:, :2, :2],
                proj_joints_in_body_crop]) + head_inv_crop_transforms[
                    :, :2, 2].unsqueeze(dim=1)
        out_params['hd_proj_joints'] = hd_proj_joints.detach()
        # hd_proj_joints = torch.einsum(
        #     'bij,bkj->bki',
        #     [left_hand_inv_crop_transforms[:, :2, :2],
        #         proj_joints_in_body_crop]) + left_hand_inv_crop_transforms[
        #             :, :2, 2].unsqueeze(dim=1)
        # out_params['hd_proj_joints'] = hd_proj_joints.detach()

        inv_head_crop_transf = torch.inverse(head_crop_transform)
        head_img_keypoints = torch.einsum(
            'bij,bkj->bki',
            [inv_head_crop_transf[:, :2, :2],
                hd_proj_joints]) + inv_head_crop_transf[:, :2, 2].unsqueeze(
                    dim=1)
        out_params['head_proj_joints'] = (
            head_img_keypoints.detach() * self.head_crop_size)

        inv_left_hand_crop_transf = torch.inverse(left_hand_crop_transform)
        left_hand_img_keypoints = torch.einsum(
            'bij,bkj->bki',
            [inv_left_hand_crop_transf[:, :2, :2],
                hd_proj_joints]) + inv_left_hand_crop_transf[
                    :, :2, 2].unsqueeze(dim=1)
        out_params['left_hand_proj_joints'] = left_hand_img_keypoints.detach() * self.hand_crop_size

        inv_right_hand_crop_transf = torch.inverse(
            right_hand_crop_transform)
        right_hand_img_keypoints = torch.einsum(
            'bij,bkj->bki',
            [inv_right_hand_crop_transf[:, :2, :2],
                hd_proj_joints]) + inv_right_hand_crop_transf[
                    :, :2, 2].unsqueeze(dim=1)
        out_params['right_hand_proj_joints'] = (
            right_hand_img_keypoints.detach() * self.hand_crop_size)

        return out_params