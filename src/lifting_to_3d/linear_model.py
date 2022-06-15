#!/usr/bin/env python
# -*- coding: utf-8 -*-

# some code from https://raw.githubusercontent.com/weigq/3d_pose_baseline_pytorch/master/src/model.py


from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# from priors.vae_pose_model.vae_model import VAEmodel
from priors.normalizing_flow_prior.normalizing_flow_prior import NormalizingFlowPrior


def weight_init_dangerous(m):
    # this is dangerous as it may overwrite the normalizing flow weights
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        out = x + y
        return out


class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5, 
                 input_size=16*2,
                 output_size=16*3):
        super(LinearModel, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        # input
        self.input_size = input_size        # 2d joints: 16 * 2
        # output
        self.output_size = output_size      # 3d joints: 16 * 3
        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)
        # post-processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        # helpers (relu and dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        # post-processing
        y = self.w2(y)
        return y

 
class LinearModelComplete(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage_comb=2,
                 num_stage_heads=1,
                 num_stage_heads_pose=1, 
                 trans_sep=False,
                 p_dropout=0.5, 
                 input_size=16*2,
                 intermediate_size=1024,
                 output_info=None,
                 n_joints=25,
                 n_z=512, 
                 add_z_to_3d_input=False,
                 n_segbps=64*2,
                 add_segbps_to_3d_input=False,
                 structure_pose_net='default', 
                 fix_vae_weights=True, 
                 nf_version=None):       # 0): n_silh_enc       
        super(LinearModelComplete, self).__init__()
        if add_z_to_3d_input:
            self.n_z_to_add = n_z    # 512
        else:
            self.n_z_to_add = 0
        if add_segbps_to_3d_input:
            self.n_segbps_to_add = n_segbps    # 64
        else:
            self.n_segbps_to_add = 0
        self.input_size = input_size 
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage_comb = num_stage_comb
        self.num_stage_heads = num_stage_heads
        self.num_stage_heads_pose = num_stage_heads_pose
        self.trans_sep = trans_sep
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.structure_pose_net = structure_pose_net
        self.fix_vae_weights = fix_vae_weights  # only relevant if structure_pose_net='vae'
        self.nf_version = nf_version
        if output_info is None:
            pose = {'name': 'pose', 'n': n_joints*6, 'out_shape':[n_joints, 6]}
            cam = {'name': 'flength', 'n': 1}
            if self.trans_sep:
                translation_xy = {'name': 'trans_xy', 'n': 2}
                translation_z = {'name': 'trans_z', 'n': 1}
                self.output_info = [pose, translation_xy, translation_z, cam]
            else:
                translation = {'name': 'trans', 'n': 3}
                self.output_info = [pose, translation, cam]
            if self.structure_pose_net == 'vae' or self.structure_pose_net == 'normflow':
                global_pose = {'name': 'global_pose', 'n': 1*6, 'out_shape':[1, 6]}
                self.output_info.append(global_pose)
        else:
            self.output_info = output_info
        self.linear_combined = LinearModel(linear_size=self.linear_size,
                                            num_stage=self.num_stage_comb,
                                            p_dropout=p_dropout, 
                                            input_size=self.input_size + self.n_segbps_to_add + self.n_z_to_add,       ######
                                            output_size=self.intermediate_size)
        self.output_info_linear_models = []
        for ind_el, element in enumerate(self.output_info):
            if element['name'] == 'pose':
                num_stage = self.num_stage_heads_pose
                if self.structure_pose_net == 'default':
                    output_size_pose_lin = element['n']
                elif self.structure_pose_net == 'vae':
                    # load vae decoder
                    self.pose_vae_model = VAEmodel()
                    self.pose_vae_model.initialize_with_pretrained_weights()
                    # define the input size of the vae decoder
                    output_size_pose_lin = self.pose_vae_model.latent_size
                elif self.structure_pose_net == 'normflow':
                    # the following will automatically be initialized
                    self.pose_normflow_model = NormalizingFlowPrior(nf_version=self.nf_version)
                    output_size_pose_lin = element['n'] - 6 # no global rotation
                else:
                    raise NotImplementedError
                self.output_info_linear_models.append(LinearModel(linear_size=self.linear_size,
                                        num_stage=num_stage,
                                        p_dropout=p_dropout, 
                                        input_size=self.intermediate_size,
                                        output_size=output_size_pose_lin))
            else: 
                if element['name'] == 'global_pose':
                    num_stage = self.num_stage_heads_pose
                else:
                    num_stage = self.num_stage_heads
                self.output_info_linear_models.append(LinearModel(linear_size=self.linear_size,
                                        num_stage=num_stage,
                                        p_dropout=p_dropout, 
                                        input_size=self.intermediate_size,
                                        output_size=element['n']))
            element['linear_model_index'] = ind_el
        self.output_info_linear_models = nn.ModuleList(self.output_info_linear_models)

    def forward(self, x):
        device = x.device
        # combined stage
        if x.shape[1] == self.input_size + self.n_segbps_to_add + self.n_z_to_add:
            y = self.linear_combined(x)
        elif x.shape[1] == self.input_size + self.n_segbps_to_add:
            x_mod = torch.cat((x, torch.normal(0, 1, size=(x.shape[0], self.n_z_to_add)).to(device)), dim=1)
            y = self.linear_combined(x_mod)
        else:
            print(x.shape)
            print(self.input_size)
            print(self.n_segbps_to_add)
            print(self.n_z_to_add)
            raise ValueError
        # heads
        results = {}
        results_trans = {}
        for element in self.output_info:
            linear_model = self.output_info_linear_models[element['linear_model_index']]
            if  element['name'] == 'pose':  
                if self.structure_pose_net == 'default':
                    results['pose'] = (linear_model(y)).reshape((-1, element['out_shape'][0], element['out_shape'][1]))
                    normflow_z = None
                elif self.structure_pose_net == 'vae':
                    res_lin = linear_model(y)
                    if self.fix_vae_weights:
                        self.pose_vae_model.requires_grad_(False)     # let gradients flow through but don't update the parameters
                        res_vae = self.pose_vae_model.inference(feat=res_lin)
                        self.pose_vae_model.requires_grad_(True)    
                    else:
                        res_vae = self.pose_vae_model.inference(feat=res_lin)
                    res_pose_not_glob = res_vae.reshape((-1, element['out_shape'][0], element['out_shape'][1])) 
                    normflow_z = None
                elif self.structure_pose_net == 'normflow':
                    normflow_z = linear_model(y)*0.1
                    self.pose_normflow_model.requires_grad_(False)     # let gradients flow though but don't update the parameters
                    res_pose_not_glob = self.pose_normflow_model.run_backwards(z=normflow_z).reshape((-1, element['out_shape'][0]-1, element['out_shape'][1]))
                else:
                    raise NotImplementedError
            elif element['name'] == 'global_pose':
                res_pose_glob = (linear_model(y)).reshape((-1, element['out_shape'][0], element['out_shape'][1]))
            elif element['name'] == 'trans_xy' or element['name'] == 'trans_z':
                results_trans[element['name']] = linear_model(y)
            else:
                results[element['name']] = linear_model(y)
        if self.trans_sep:
            results['trans'] = torch.cat((results_trans['trans_xy'], results_trans['trans_z']), dim=1)
        # prepare pose including global rotation
        if self.structure_pose_net == 'vae':
            # results['pose'] = torch.cat((res_pose_glob, res_pose_not_glob), dim=1)
            results['pose'] = torch.cat((res_pose_glob, res_pose_not_glob[:, 1:, :]), dim=1)   
        elif self.structure_pose_net == 'normflow':
            results['pose'] = torch.cat((res_pose_glob, res_pose_not_glob[:, :, :]), dim=1)    
        # return a dictionary which contains all results
        results['normflow_z'] = normflow_z
        return results      # this is a dictionary





# ------------------------------------------
# for pretraining of the 3d model only:
#   (see combined_model/model_shape_v2.py)

class Wrapper_LinearModelComplete(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage_comb=2,
                 num_stage_heads=1,
                 num_stage_heads_pose=1, 
                 trans_sep=False,
                 p_dropout=0.5, 
                 input_size=16*2,
                 intermediate_size=1024,
                 output_info=None,
                 n_joints=25,
                 n_z=512, 
                 add_z_to_3d_input=False,
                 n_segbps=64*2,
                 add_segbps_to_3d_input=False,
                 structure_pose_net='default', 
                 fix_vae_weights=True,
                 nf_version=None):
        self.add_segbps_to_3d_input = add_segbps_to_3d_input
        super(Wrapper_LinearModelComplete, self).__init__()
        self.model_3d = LinearModelComplete(linear_size=linear_size,
                    num_stage_comb=num_stage_comb,
                    num_stage_heads=num_stage_heads,
                    num_stage_heads_pose=num_stage_heads_pose,
                    trans_sep=trans_sep,
                    p_dropout=p_dropout,        # 0.5, 
                    input_size=input_size,
                    intermediate_size=intermediate_size,
                    output_info=output_info,
                    n_joints=n_joints,
                    n_z=n_z,
                    add_z_to_3d_input=add_z_to_3d_input,
                    n_segbps=n_segbps,
                    add_segbps_to_3d_input=add_segbps_to_3d_input,
                    structure_pose_net=structure_pose_net, 
                    fix_vae_weights=fix_vae_weights, 
                    nf_version=nf_version)
    def forward(self, input_vec):
        # input_vec = torch.cat((keypoints_prepared.reshape((batch_size, -1)), bone_lengths_prepared), axis=1)  
        # predict 3d parameters (those are normalized, we need to correct mean and std in a next step)
        output = self.model_3d(input_vec)  
        return output    