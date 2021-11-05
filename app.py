import gradio
import torch
import os
from typing import Mapping, Tuple
import numpy as np
import toolz
import cv2
import functools
import datetime

from collections import defaultdict

from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor
from expose.data.targets.bbox import BoundingBox
from expose.data.utils.bbox import bbox_to_center_scale

from expose.utils.plot_utils import HDRenderer
from expose.models.smplx import SMPLX
from expose.utils.img_utils import read_img
from expose.data import transforms as T
from expose.models.common.bbox_sampler import CropSampler

from detectron2.config import CfgNode, get_cfg
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import (
    extract_boxes_xywh_from_instances,
)
from densepose.converters import ToChartResultConverterWithConfidences
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)

from logging import getLogger

logger = getLogger(__name__)

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

def get_body_crop_size() -> int:
    return 256

def get_overlayer() -> HDRenderer:
    return None

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

# create model
# load ckpt
# get renderer
# estimate bbox
# crop input & preserve hd
# predict body
# render overlays

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # IMG_FILENAME = './samples/man-in-red-crew-neck-sweatshirt-photography-941693.png'
    
    
    ## DensePose
    DENSEPOSE_CFG = './data/detectron2/densepose_rcnn_R_50_FPN_s1x.yaml'
    DENSEPOSE_WEIGHTS = './data/detectron2/model_final_162be9.pkl'
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(DENSEPOSE_CFG)
    cfg.MODEL.WEIGHTS = DENSEPOSE_WEIGHTS
    densepose = DefaultPredictor(cfg)   

    ## ExPose
    CKPT_FILENAME = './data/checkpoints/model.ckpt'
    smplx_device = torch.device('cuda:0')
    model = SMPLX().to(smplx_device)
    load_checkpoint(model, CKPT_FILENAME)
    model = model.eval()
    rcnn_model = keypointrcnn_resnet50_fpn(pretrained=True).cpu()
    rcnn_model.eval()
    transform = Compose(
        [ToTensor(), ]
    )
    mean, std = get_means_std()
    transforms = T.Compose([
        T.Crop(crop_size=get_body_crop_size(), 
            is_train=False,
            scale_factor_max=1.0,
            scale_factor_min=1.0,
            scale_factor=0.0,
            scale_dist='uniform'
        ),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])     

    def predict(
        img: np.array,
        gender: str,
        focal_length: float,
        # detach_mean: bool,
        # append_params: bool,
        # predict_hands: bool,
        # update_wrists: bool,
        hand_crop_size: int,
        head_crop_size: int,
        head_scale_factor: float,
        hand_scale_factor: float,
        num_iterations: int
    ):
        global model
        # model.predict_hands = predict_hands
        # if not model.predict_hands:
        #     model.apply_hand_network_on_body = False
        # model.update_wrists = update_wrists
        model.head_scale_factor = head_scale_factor
        model.hand_scale_factor = hand_scale_factor        
        model.hand_crop_size = hand_crop_size
        model.hand_cropper = CropSampler(int(hand_crop_size)).to(smplx_device)
        model.head_crop_size = head_crop_size
        model.head_cropper = CropSampler(int(head_crop_size)).to(smplx_device)
        if gender != model.body_model.gender or\
            focal_length != model.focal_length or\
            model.num_stages != int(num_iterations):
            # detach_mean != model.detach_mean or\
            # append_params != model.append_params or\
            model = SMPLX(
                gender=gender,
                # append_params=append_params,
                # detach_mean=detach_mean,
                focal_length=focal_length,
                num_stages=int(num_iterations),
            ).to(smplx_device)
            model = model.eval()
            load_checkpoint(model, CKPT_FILENAME)
        
        with torch.no_grad():            
            img = img.astype(np.float32) / 255.0
            tensor_img = transform(img)
            output = rcnn_model([tensor_img.cpu()])
            scale_factor = 1.2
            bbox = output[0]['boxes'][0].cpu().detach()
            center, scale, bbox_size = bbox_to_center_scale(
                bbox, dset_scale_factor=scale_factor
            )
            bbox = BoundingBox(bbox.numpy(), tensor_img.shape)
            bbox.add_field('bbox_size', bbox_size.numpy())
            bbox.add_field('orig_bbox_size', bbox_size.numpy())
            bbox.add_field('orig_center', center)
            bbox.add_field('center', center)
            bbox.add_field('scale', scale)
            # bbox.add_field('fname', os.path.basename(IMG_FILENAME))

            # DensePose
            outputs = densepose(img * 255)["instances"]
            dpout = outputs.pred_densepose
            boxes_xyxy = outputs.pred_boxes
            boxes_xywh = extract_boxes_xywh_from_instances(outputs)
            converter = ToChartResultConverterWithConfidences()
            results = [converter.convert(dpout[i], boxes_xyxy[[i]]) for i in range(len(dpout))]
            # visualizer = DensePoseResultsContourVisualizer()
            visualizer = DensePoseResultsUVisualizer()
            densepose_vis = visualizer.visualize(img.astype(np.uint8), (results, boxes_xywh))
            visualizer = DensePoseResultsFineSegmentationVisualizer()
            densepose_mask = visualizer.visualize(img.astype(np.uint8), (results, boxes_xywh))            
            densepose_mask = np.repeat((densepose_vis.sum(axis=-1) != 0).astype(np.float32)[:, :, None], 3, axis=-1)

            # ExPose
            img, cropped, target = transforms(img, bbox)
            output = model(
                cropped.unsqueeze(0).cuda(),
                [target], 
                full_imgs=img.unsqueeze(0).cuda()
            )
            body = output['body']
            num_stages = body.get('num_stages', 3)
            stage_n_out = body.get(f'stage_{num_stages - 1:02d}', {})
            model_vertices = stage_n_out.get('vertices', None)
            faces = stage_n_out['faces']
            if model_vertices is not None:
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
            H, W = img.shape[1:]        
            sensor_width: float = 36
            render_params = weak_persp_to_blender(
                [target],
                camera_scale=camera_scale,
                camera_transl=camera_transl,
                H=H, W=W,
                sensor_width=sensor_width,
                focal_length=focal_length,
            )
            bg_img = np.clip(undo_img_normalization(
                img.unsqueeze(0).numpy(), mean, std
            ), 0.0, 1.0)
            renderer = HDRenderer(img_size=get_body_crop_size())
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
            # cv2.imwrite('orig.png', (orig_overlay[0].transpose(1, 2, 0) * 255).astype(np.uint8))
            return [
                body_fit[0].transpose(1, 2, 0),
                expressive_fit[0].transpose(1, 2, 0),
                densepose_vis,
                densepose_mask,
            ]
            # return body_fit[0].transpose(1, 2, 0), expressive_fit[0].transpose(1, 2, 0)
        
    # cv2.imwrite('orig.png', (orig_overlay[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    # cv2.waitKey(-1)
    
    # img = (read_img(IMG_FILENAME) * 255).astype(np.uint8)
    # expose(img)

    iface = gradio.Interface(predict,
        [
            gradio.inputs.Image(), 
            gradio.inputs.Radio(
                ['male', 'neutral', 'female'],
                type="value",
                default='neutral',
                label='Gender'
            ),
            gradio.inputs.Slider(
                minimum=1000.0, maximum=10000.0,
                step=100.0, default=5000.0,
                label='focal length'
            ),
            # gradio.inputs.Checkbox(default=False, label='Detach mean'),
            # gradio.inputs.Checkbox(default=True, label='Append params'),
            # gradio.inputs.Checkbox(default=True, label='Predict hands'),
            # gradio.inputs.Checkbox(default=True, label='Update wrists'),
            gradio.inputs.Number(default=224, label='Hand crop'),
            gradio.inputs.Number(default=256, label='Head crop'),
            gradio.inputs.Slider(
                minimum=0.25, maximum=2.5,
                step=0.05, default=1.2,
                label='Head scale'
            ),
            gradio.inputs.Slider(
                minimum=0.25, maximum=2.5,
                step=0.05, default=1.2,
                label='Hand scale'
            ),
            gradio.inputs.Slider(
                minimum=1, maximum=3,
                step=1, default=3,
                label='Iterations'
            ),
        ],
        gradio.outputs.Carousel([
            gradio.outputs.Image(label="1. Body"), 
            gradio.outputs.Image(label="2. Expressive"),
            gradio.outputs.Image(label="3. DensePose Chart"),
            gradio.outputs.Image(label="4. DensePose Mask"),
        ], label="Results"),
        title="Body Fit",
        description="Upload an image and select the body fitting parameters.",
        theme="darkgrass",
        flagging_options=[
            "challenging_good", "ok_good", "bad_pose", "bad_shape"
        ],
        allow_flagging=True,
        flagging_dir='flags_from_'+str(datetime.datetime.now().date()),
        # show_tips=True,
        layout="unaligned",
        # interpretation="default",
        css=".output_image, .input_image {height: 40rem !important; width: 100% !important;}",
        examples='./test/web',
    )
    iface.launch(inbrowser=True, share=True)
