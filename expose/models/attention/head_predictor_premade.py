# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys

from typing import Dict, NewType, Optional

from copy import deepcopy
import pickle
import time

from collections import defaultdict
import math
import os.path as osp

from loguru import logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nninit

from smplx import build_layer

from ..backbone.resnet import resnet18
from ..common.networks import MLP, IterativeRegression
from ..common.pose_utils import build_pose_decoder
from ..camera import build_cam_proj, CameraParams

from expose.data.targets.keypoints import KEYPOINT_NAMES
from expose.utils.rotation_utils import batch_rodrigues, batch_rot2aa

from expose.utils.typing_utils import Tensor


class HeadPredictor(nn.Module):
    def __init__(self, 
            camera_cfg,
            global_orient_desc,
            jaw_pose_desc,
            camera_data,
            num_betas: int=100,
            num_expression_coeffs: int=50,
            append_params: bool=True,
            num_stages: int=3,
            detach_mean=False,
            feature_key: str='avg_pooling',
            dtype=torch.float32
        ):
        super(HeadPredictor, self).__init__()
        self.neck_index = KEYPOINT_NAMES.index('neck')
        self.head_model_type = 'flame'
        self.num_stages = num_stages
        self.append_params = append_params

        camera_data = build_cam_proj(camera_cfg, dtype=dtype)
        self.projection = camera_data['camera']
        camera_param_dim = camera_data['dim']
        camera_mean = camera_data['mean']
        self.register_buffer('camera_mean', camera_mean)
        self.camera_scale_func = camera_data['scale_func']

        self.num_betas = num_betas
        #  self.num_betas = self.head_model.num_betas
        shape_mean = torch.zeros([self.num_betas], dtype=dtype)
        self.register_buffer('shape_mean', shape_mean)

        #  self.num_expression_coeffs = self.head_model.num_expression_coeffs
        self.num_expression_coeffs = num_expression_coeffs
        expression_mean = torch.zeros(
            [self.num_expression_coeffs], dtype=dtype)
        self.register_buffer('expression_mean', expression_mean)

        self.global_orient_decoder = global_orient_desc.decoder

        cfg = {'param_type': global_orient_desc.decoder.get_type()}
        self.neck_pose_decoder = build_pose_decoder(cfg, 1)
        neck_pose_mean = self.neck_pose_decoder.get_mean().clone()
        neck_pose_type = cfg['param_type']
        if neck_pose_type == 'aa':
            neck_pose_mean[0] = math.pi
        elif neck_pose_type == 'cont_rot_repr':
            neck_pose_mean[3] = -1
        neck_pose_dim = self.neck_pose_decoder.get_dim_size()
        self.register_buffer('neck_pose_mean', neck_pose_mean)

        self.jaw_pose_decoder = jaw_pose_desc.decoder
        jaw_pose_mean = jaw_pose_desc.mean
        jaw_pose_dim = jaw_pose_desc.dim

        mean_lst = []
        start = 0
        neck_pose_idxs = list(range(start, start + neck_pose_dim))
        self.register_buffer('neck_pose_idxs',
            torch.tensor(neck_pose_idxs, dtype=torch.long)
        )
        start += neck_pose_dim
        mean_lst.append(neck_pose_mean.view(-1))

        jaw_pose_idxs = list(range(
            start, start + jaw_pose_dim))
        self.register_buffer('jaw_pose_idxs',
            torch.tensor(jaw_pose_idxs, dtype=torch.long)
        )
        start += jaw_pose_dim
        mean_lst.append(jaw_pose_mean.view(-1))

        shape_idxs = list(range(start, start + self.num_betas))
        self.register_buffer('shape_idxs', 
            torch.tensor(shape_idxs, dtype=torch.long)
        )
        start += self.num_betas
        mean_lst.append(shape_mean.view(-1))

        expression_idxs = list(range(
            start, start + self.num_expression_coeffs))
        self.register_buffer('expression_idxs',
            torch.tensor(expression_idxs, dtype=torch.long)
        )
        start += self.num_expression_coeffs
        mean_lst.append(expression_mean.view(-1))

        camera_idxs = list(range(
            start, start + camera_param_dim))
        self.register_buffer('camera_idxs',
            torch.tensor(camera_idxs, dtype=torch.long)
        )
        start += camera_param_dim
        mean_lst.append(camera_mean)

        param_mean = torch.cat(mean_lst).view(1, -1)
        param_dim = param_mean.numel()
        self.param_dim = param_dim

        # Construct the feature extraction backbone
        self.backbone = resnet18(replace_stride_with_dilation=[False, False, False])
        feat_dims = self.backbone.get_output_dim()

        self.append_params = append_params
        self.num_stages = num_stages

        self.feature_key = feature_key
        feat_dim = feat_dims[self.feature_key]
        self.feat_dim = feat_dim

        regressor_cfg = {
            'activ_type': 'relu', 'bias_init': 0.0,
            'dropout': 0.5, 'gain': 0.01,
            'init_type': 'xavier', 'layers': [1024, 1024],
            'lrelu_slope': 0.2, 'norm_type': 'none',
            'num_groups': 32,
        }
        regressor = MLP(feat_dim + self.append_params * param_dim,
                        param_dim, **regressor_cfg)
        self.regressor = IterativeRegression(
            regressor, param_mean, detach_mean=detach_mean,
            num_stages=self.num_stages)

    def get_feat_dim(self) -> int:
        ''' Returns the dimension of the expected feature vector '''
        return self.feat_dim

    def get_param_dim(self) -> int:
        ''' Returns the dimension of the predicted parameter vector '''
        return self.param_dim

    def get_num_stages(self) -> int:
        ''' Returns the number of stages for the iterative predictor'''
        return self.num_stages

    def get_num_betas(self) -> int:
        return self.num_betas

    def get_num_expression_coeffs(self) -> int:
        return self.num_expression_coeffs

    def param_tensor_to_dict(
            self, param_tensor: Tensor) -> Dict[str, Tensor]:
        ''' Converts a flattened tensor to a dictionary of tensors '''
        neck_pose = torch.index_select(param_tensor, 1,
                                       self.neck_pose_idxs)
        jaw_pose = torch.index_select(param_tensor, 1, self.jaw_pose_idxs)

        betas = torch.index_select(param_tensor, 1, self.shape_idxs)
        expression = torch.index_select(param_tensor, 1, self.expression_idxs)

        return dict(neck_pose=neck_pose,
                    jaw_pose=jaw_pose,
                    expression=expression,
                    betas=betas)

    def get_camera_mean(self, batch_size: int = 1) -> Tensor:
        ''' Returns the camera mean '''
        return self.camera_mean.reshape(1, -1).expand(batch_size, -1)

    def get_neck_pose_mean(self, batch_size=1) -> Tensor:
        ''' Returns neck pose mean '''
        return self.neck_pose_mean.reshape(1, -1).expand(batch_size, -1)

    def get_jaw_pose_mean(self, batch_size=1) -> Tensor:
        ''' Returns jaw pose mean '''
        return self.jaw_pose_mean.reshape(1, -1).expand(batch_size, -1)

    def get_shape_mean(self, batch_size=1) -> Tensor:
        ''' Returns shape mean '''
        return self.shape_mean.reshape(1, -1).expand(batch_size, -1)

    def get_expression_mean(self, batch_size=1) -> Tensor:
        ''' Returns expression mean '''
        return self.expression_mean.reshape(1, -1).expand(batch_size, -1)

    def get_param_mean(self, batch_size: int = 1):
        ''' Return the mean that will be given to the iterative regressor
        '''
        return self.regressor.get_mean().clone().reshape(1, -1).expand(
            batch_size, -1).clone()

    def forward(self,
                head_imgs: Tensor,
                global_orient_from_body_net: Optional[Tensor] = None,
                body_pose_from_body_net: Optional[Tensor] = None,
                left_hand_pose_from_body_net: Optional[Tensor] = None,
                right_hand_pose_from_body_net: Optional[Tensor] = None,
                jaw_pose_from_body_net: Optional[Tensor] = None,
                num_head_imgs: int = 0,
                head_mean: Optional[Tensor] = None,
                device: torch.device = None,
                ) -> Dict[str, Dict[str, Tensor]]:
        '''
        '''
        batch_size = head_imgs.shape[0]

        if batch_size == 0:
            return {}

        head_features = self.backbone(head_imgs)
        head_parameters, _ = self.regressor(
            head_features[self.feature_key],
            cond=head_mean)

        head_model_params = []
        model_parameters = []
        for _, parameters in enumerate(head_parameters):
            parameters_dict = self.param_tensor_to_dict(parameters)

            dec_neck_pose_abs = self.neck_pose_decoder(
                parameters_dict['neck_pose'])
            dec_jaw_pose = self.jaw_pose_decoder(parameters_dict['jaw_pose'])

            model_betas = parameters_dict['betas']
            # Parameters that will be returned
            model_parameters.append(
                dict(head_pose=dec_neck_pose_abs,
                     raw_jaw_pose=parameters_dict['jaw_pose'],
                     jaw_pose=dec_jaw_pose,
                     betas=model_betas,
                     expression=parameters_dict['expression'],
                     )
            )

            # Parameters used to pose the model
            if self.head_model_type == 'flame':
                head_model_params.append(
                    dict(global_orient=dec_neck_pose_abs,
                         jaw_pose=dec_jaw_pose,
                         betas=model_betas,
                         expression=parameters_dict['expression'],
                         )
                )
            else:
                raise RuntimeError(
                    f'Invalid head model type: {self.head_model_type}')

        output = {
            'num_stages': self.num_stages,
            'features': head_features[self.feature_key],
        }

        for stage in range(self.num_stages):
            # Only update the current stage if there are enough params
            key = f'stage_{stage:02d}'
            output[key] = model_parameters[stage]

        return output
