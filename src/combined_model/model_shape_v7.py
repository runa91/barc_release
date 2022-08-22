
import pickle as pkl
import numpy as np
import torchvision.models as models
from torchvision import transforms
import torch
from torch import nn
from torch.nn.parameter import Parameter
from kornia.geometry.subpix import dsnt     # kornia 0.4.0


import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from stacked_hourglass.utils.evaluation import get_preds_soft
from stacked_hourglass import hg1, hg2, hg8
from lifting_to_3d.linear_model import LinearModelComplete, LinearModel      
from lifting_to_3d.inn_model_for_shape import INNForShape
from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d
from smal_pytorch.smal_model.smal_torch_new import SMAL
from smal_pytorch.renderer.differentiable_renderer import SilhRenderer
from bps_2d.bps_for_segmentation import SegBPS
from configs.SMAL_configs import UNITY_SMAL_SHAPE_PRIOR_DOGS as SHAPE_PRIOR
from configs.SMAL_configs import MEAN_DOG_BONE_LENGTHS_NO_RED, VERTEX_IDS_TAIL
import time
def ckpt_time_v1(ckpt=None, display=0, desc=''):
    if not ckpt:
        return time.time()
    else:
        if display:
            print(desc + ' consume time {:0.4f}'.format(time.time() - float(ckpt)))
        return time.time() - float(ckpt), time.time()
## usage sample:    
# time1=ckpt_time_v1()
# ckpt,time2=ckpt_time_v1(time1,display=True,desc="====")


class SmallLinear(nn.Module):
    def __init__(self, input_size=64, output_size=30, linear_size=128):
        super(SmallLinear, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(input_size, linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)
        self.w3 = nn.Linear(linear_size, output_size)
    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.relu(y)
        y = self.w2(y)
        y = self.relu(y)
        y = self.w3(y)
        return y


class MyConv1d(nn.Module):
    def __init__(self, input_size=37, output_size=30, start=True):
        super(MyConv1d, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.start = start
        self.weight = Parameter(torch.ones((self.output_size)))
        self.bias = Parameter(torch.zeros((self.output_size)))
    def forward(self, x):
        # pre-processing
        if self.start:
            y = x[:, :self.output_size]
        else:
            y = x[:, -self.output_size:]
        y = y * self.weight[None, :] + self.bias[None, :]
        return y


class ModelShapeAndBreed(nn.Module):
    def __init__(self, n_betas=10, n_betas_limbs=13, n_breeds=121, n_z=512, structure_z_to_betas='default'):
        super(ModelShapeAndBreed, self).__init__()
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs   # n_betas_logscale
        self.n_breeds = n_breeds
        self.structure_z_to_betas = structure_z_to_betas
        if self.structure_z_to_betas == '1dconv':
            if not (n_z == self.n_betas+self.n_betas_limbs):
                raise ValueError
        # shape branch
        self.resnet = models.resnet34(pretrained=False)  
        # replace the first layer
        n_in = 3 + 1
        self.resnet.conv1 = nn.Conv2d(n_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # replace the last layer
        self.resnet.fc = nn.Linear(512, n_z) 
        # softmax
        self.soft_max = torch.nn.Softmax(dim=1)
        # fc network (and other versions) to connect z with betas
        p_dropout = 0.2
        if self.structure_z_to_betas == 'default':
            self.linear_betas = LinearModel(linear_size=1024,     
                                                num_stage=1,
                                                p_dropout=p_dropout, 
                                                input_size=n_z,
                                                output_size=self.n_betas)
            self.linear_betas_limbs = LinearModel(linear_size=1024,    
                                                num_stage=1,
                                                p_dropout=p_dropout, 
                                                input_size=n_z,
                                                output_size=self.n_betas_limbs)
        elif self.structure_z_to_betas == 'lin':
            self.linear_betas = nn.Linear(n_z, self.n_betas)
            self.linear_betas_limbs = nn.Linear(n_z, self.n_betas_limbs)
        elif self.structure_z_to_betas == 'fc_0':
            self.linear_betas = SmallLinear(linear_size=128,     # 1024,
                                                input_size=n_z,
                                                output_size=self.n_betas)
            self.linear_betas_limbs = SmallLinear(linear_size=128,     # 1024,
                                                input_size=n_z,
                                                output_size=self.n_betas_limbs)
        elif structure_z_to_betas == 'fc_1':
            self.linear_betas = LinearModel(linear_size=64,     # 1024,
                                                num_stage=1,
                                                p_dropout=0, 
                                                input_size=n_z,
                                                output_size=self.n_betas)
            self.linear_betas_limbs = LinearModel(linear_size=64,     # 1024,
                                                num_stage=1,
                                                p_dropout=0, 
                                                input_size=n_z,
                                                output_size=self.n_betas_limbs)
        elif self.structure_z_to_betas == '1dconv':
            self.linear_betas = MyConv1d(n_z, self.n_betas, start=True)
            self.linear_betas_limbs = MyConv1d(n_z, self.n_betas_limbs, start=False)
        elif self.structure_z_to_betas == 'inn':
            self.linear_betas_and_betas_limbs = INNForShape(self.n_betas, self.n_betas_limbs, betas_scale=1.0, betas_limbs_scale=1.0)
        else:
            raise ValueError
        # network to connect latent shape vector z with dog breed classification
        self.linear_breeds = LinearModel(linear_size=1024,    # 1024,
                                            num_stage=1,
                                            p_dropout=p_dropout, 
                                            input_size=n_z,
                                            output_size=self.n_breeds)
        # shape multiplicator
        self.shape_multiplicator_np = np.ones(self.n_betas)
        with open(SHAPE_PRIOR, 'rb') as file:
            u = pkl._Unpickler(file)
            u.encoding = 'latin1'
            res = u.load()
        # shape predictions are centered around the mean dog of our dog model
        self.betas_mean_np = res['dog_cluster_mean']  
                                        
    def forward(self, img, seg_raw=None, seg_prep=None):
        # img is the network input image 
        # seg_raw is before softmax and subtracting 0.5
        # seg_prep would be the prepared_segmentation
        if seg_prep is None:
            seg_prep = self.soft_max(seg_raw)[:, 1:2, :, :] - 0.5       
        input_img_and_seg = torch.cat((img, seg_prep), axis=1)
        time_res1=ckpt_time_v1()
        res_output = self.resnet(input_img_and_seg)
        ckpt,time_res2=ckpt_time_v1(time_res1,display=True,desc="====resnet infer")
        dog_breed_output = self.linear_breeds(res_output) 
        ckpt,time_breed=ckpt_time_v1(time_res2,display=True,desc="====linear_breeds infer")
        if self.structure_z_to_betas == 'inn':
            shape_output_orig, shape_limbs_output_orig = self.linear_betas_and_betas_limbs(res_output)
            ckpt,time_inn=ckpt_time_v1(time_breed,display=True,desc="====time_inn infer")
        else:
            shape_output_orig = self.linear_betas(res_output) * 0.1
            betas_mean = torch.tensor(self.betas_mean_np).float().to(img.device)
            shape_output = shape_output_orig + betas_mean[None, 0:self.n_betas]
            shape_limbs_output_orig = self.linear_betas_limbs(res_output)
            shape_limbs_output = shape_limbs_output_orig * 0.1
            ckpt,time_linear_betas=ckpt_time_v1(time_breed,display=True,desc="====time_linear_betas infer")
        output_dict = {'z': res_output,
                        'breeds': dog_breed_output,
                        'betas': shape_output_orig,
                        'betas_limbs': shape_limbs_output_orig}
        return output_dict



class LearnableShapedirs(nn.Module):
    def __init__(self, sym_ids_dict, shapedirs_init, n_betas, n_betas_fixed=10):
        super(LearnableShapedirs, self).__init__()
        # shapedirs_init = self.smal.shapedirs.detach()
        self.n_betas = n_betas
        self.n_betas_fixed = n_betas_fixed
        self.sym_ids_dict = sym_ids_dict
        sym_left_ids = self.sym_ids_dict['left']
        sym_right_ids = self.sym_ids_dict['right']
        sym_center_ids = self.sym_ids_dict['center']
        self.n_center = sym_center_ids.shape[0]
        self.n_left = sym_left_ids.shape[0]
        self.n_sd = self.n_betas - self.n_betas_fixed     # number of learnable shapedirs
        # get indices to go from half_shapedirs to shapedirs
        inds_back = np.zeros((3889))
        for ind in range(0, sym_center_ids.shape[0]):
            ind_in_forward = sym_center_ids[ind]
            inds_back[ind_in_forward] = ind
        for ind in range(0, sym_left_ids.shape[0]):
            ind_in_forward = sym_left_ids[ind]
            inds_back[ind_in_forward] = sym_center_ids.shape[0] + ind
        for ind in range(0, sym_right_ids.shape[0]):
            ind_in_forward = sym_right_ids[ind]
            inds_back[ind_in_forward] = sym_center_ids.shape[0] + sym_left_ids.shape[0] + ind
        self.register_buffer('inds_back_torch', torch.Tensor(inds_back).long())
        # self.smal.shapedirs: (51, 11667)
        # shapedirs: (3889, 3, n_sd)
        # shapedirs_half: (2012, 3, n_sd)
        sd = shapedirs_init[:self.n_betas, :].permute((1, 0)).reshape((-1, 3, self.n_betas))
        self.register_buffer('sd', sd)
        sd_center = sd[sym_center_ids, :, self.n_betas_fixed:]
        sd_left = sd[sym_left_ids, :, self.n_betas_fixed:]
        self.register_parameter('learnable_half_shapedirs_c0', torch.nn.Parameter(sd_center[:, 0, :].detach()))
        self.register_parameter('learnable_half_shapedirs_c2', torch.nn.Parameter(sd_center[:, 2, :].detach()))
        self.register_parameter('learnable_half_shapedirs_l0', torch.nn.Parameter(sd_left[:, 0, :].detach()))
        self.register_parameter('learnable_half_shapedirs_l1', torch.nn.Parameter(sd_left[:, 1, :].detach()))
        self.register_parameter('learnable_half_shapedirs_l2', torch.nn.Parameter(sd_left[:, 2, :].detach()))
    def forward(self):
        device = self.learnable_half_shapedirs_c0.device
        half_shapedirs_center = torch.stack((self.learnable_half_shapedirs_c0, \
                                            torch.zeros((self.n_center, self.n_sd)).to(device), \
                                            self.learnable_half_shapedirs_c2), axis=1)
        half_shapedirs_left = torch.stack((self.learnable_half_shapedirs_l0, \
                                            self.learnable_half_shapedirs_l1, \
                                            self.learnable_half_shapedirs_l2), axis=1)
        half_shapedirs_right = torch.stack((self.learnable_half_shapedirs_l0, \
                                            - self.learnable_half_shapedirs_l1, \
                                            self.learnable_half_shapedirs_l2), axis=1)
        half_shapedirs_tot = torch.cat((half_shapedirs_center, half_shapedirs_left, half_shapedirs_right))
        shapedirs = torch.index_select(half_shapedirs_tot, dim=0, index=self.inds_back_torch)
        shapedirs_complete = torch.cat((self.sd[:, :, :self.n_betas_fixed], shapedirs), axis=2)      # (3889, 3, n_sd)
        shapedirs_complete_prepared = torch.cat((self.sd[:, :, :10], shapedirs), axis=2).reshape((-1, 30)).permute((1, 0))   # (n_sd, 11667)
        return shapedirs_complete, shapedirs_complete_prepared





class ModelImageToBreed(nn.Module):
    def __init__(self, arch='hg8', n_joints=35, n_classes=20, n_partseg=15, n_keyp=20, n_bones=24, n_betas=10, n_betas_limbs=7, n_breeds=121, image_size=256, n_z=512, thr_keyp_sc=None, add_partseg=True):
        super(ModelImageToBreed, self).__init__()
        self.n_classes = n_classes
        self.n_partseg = n_partseg
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs
        self.n_keyp = n_keyp
        self.n_bones = n_bones
        self.n_breeds = n_breeds
        self.image_size = image_size
        self.upsample_seg = True
        self.threshold_scores = thr_keyp_sc 
        self.n_z = n_z
        self.add_partseg = add_partseg
        # ------------------------------ STACKED HOUR GLASS ------------------------------        
        if arch == 'hg8':
            self.stacked_hourglass = hg8(pretrained=False, num_classes=self.n_classes, num_partseg=self.n_partseg, upsample_seg=self.upsample_seg, add_partseg=self.add_partseg)
        else:
            raise Exception('unrecognised model architecture: ' + arch)
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        self.breed_model = ModelShapeAndBreed(n_betas=self.n_betas, n_betas_limbs=self.n_betas_limbs, n_breeds=self.n_breeds, n_z=self.n_z)
    def forward(self, input_img, norm_dict=None, bone_lengths_prepared=None, betas=None):
        batch_size = input_img.shape[0]
        device = input_img.device
        # ------------------------------ STACKED HOUR GLASS ------------------------------
        time_hg1=ckpt_time_v1()
        hourglass_out_dict = self.stacked_hourglass(input_img)
        ckpt,time_hg2=ckpt_time_v1(time_hg1,display=True,desc="====stacked_hourglass infer")

        last_seg = hourglass_out_dict['seg_final']
        last_heatmap = hourglass_out_dict['out_list_kp'][-1] 
        # - prepare keypoints (from heatmap)
        # normalize predictions -> from logits to probability distribution
        # last_heatmap_norm = dsnt.spatial_softmax2d(last_heatmap, temperature=torch.tensor(1))
        # keypoints = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=False) + 1   # (bs, 20, 2)
        # keypoints_norm = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=True)    # (bs, 20, 2)
        keypoints_norm, scores = get_preds_soft(last_heatmap, return_maxval=True, norm_coords=True)
        if self.threshold_scores is not None:
            scores[scores>self.threshold_scores] = 1.0
            scores[scores<=self.threshold_scores] = 0.0
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        # breed_model takes as input the image as well as the predicted segmentation map 
        #     -> we need to split up ModelImageTo3d, such that we can use the silhouette
        time_breed1=ckpt_time_v1()
        resnet_output = self.breed_model(img=input_img, seg_raw=last_seg)
        ckpt,time_breed2=ckpt_time_v1(time_breed1,display=True,desc="====breed_model infer. For breeds betas small_output")
        pred_breed = resnet_output['breeds']       # (bs, n_breeds)
        pred_betas = resnet_output['betas']
        pred_betas_limbs = resnet_output['betas_limbs']
        small_output = {'keypoints_norm': keypoints_norm,
                        'keypoints_scores': scores}
        small_output_reproj = {'betas': pred_betas,
                                'betas_limbs': pred_betas_limbs,
                                'dog_breed': pred_breed}
        return small_output, None, small_output_reproj

class ModelImageTo3d_withshape_withproj(nn.Module):
    def __init__(self, arch='hg8', num_stage_comb=2, num_stage_heads=1, num_stage_heads_pose=1, trans_sep=False, n_joints=35, n_classes=20, n_partseg=15, n_keyp=20, n_bones=24, n_betas=10, n_betas_limbs=6, n_breeds=121, image_size=256, n_z=512, n_segbps=64*2, thr_keyp_sc=None, add_z_to_3d_input=True, add_segbps_to_3d_input=False, add_partseg=True, silh_no_tail=True, fix_flength=False, render_partseg=False, structure_z_to_betas='default', structure_pose_net='default', nf_version=None):
        super(ModelImageTo3d_withshape_withproj, self).__init__()
        self.n_classes = n_classes
        self.n_partseg = n_partseg
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs
        self.n_keyp = n_keyp
        self.n_bones = n_bones
        self.n_breeds = n_breeds
        self.image_size = image_size
        self.threshold_scores = thr_keyp_sc 
        self.upsample_seg = True
        self.silh_no_tail = silh_no_tail
        self.add_z_to_3d_input = add_z_to_3d_input       
        self.add_segbps_to_3d_input = add_segbps_to_3d_input
        self.add_partseg = add_partseg
        assert (not self.add_segbps_to_3d_input) or (not self.add_z_to_3d_input)
        self.n_z = n_z   
        if add_segbps_to_3d_input:
            self.n_segbps = n_segbps    # 64
            self.segbps_model = SegBPS()
        else:
            self.n_segbps = 0
        self.fix_flength = fix_flength
        self.render_partseg = render_partseg
        self.structure_z_to_betas = structure_z_to_betas
        self.structure_pose_net = structure_pose_net
        assert self.structure_pose_net in ['default', 'vae', 'normflow']
        self.nf_version = nf_version
        self.register_buffer('betas_zeros', torch.zeros((1, self.n_betas)))
        self.register_buffer('mean_dog_bone_lengths', torch.tensor(MEAN_DOG_BONE_LENGTHS_NO_RED, dtype=torch.float32))
        p_dropout = 0.2      # 0.5     
        # ------------------------------ SMAL MODEL ------------------------------
        self.smal = SMAL(template_name='neutral') 
        print(f"------------------------------self.smal: {self.smal}, {type(self.smal)}")      
        # New for rendering without tail
        f_np = self.smal.faces.detach().cpu().numpy()
        self.f_no_tail_np = f_np[np.isin(f_np[:,:], VERTEX_IDS_TAIL).sum(axis=1)==0, :]
        # in theory we could optimize for improved shapedirs, but we do not do that
        #   -> would need to implement regularizations 
        #   -> there are better ways than changing the shapedirs
        self.model_learnable_shapedirs = LearnableShapedirs(self.smal.sym_ids_dict, self.smal.shapedirs.detach(), self.n_betas, 10)
        # ------------------------------ STACKED HOUR GLASS ------------------------------        
        if arch == 'hg8':
            self.stacked_hourglass = hg8(pretrained=False, num_classes=self.n_classes, num_partseg=self.n_partseg, upsample_seg=self.upsample_seg, add_partseg=self.add_partseg)
        else:
            raise Exception('unrecognised model architecture: ' + arch)
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        self.breed_model = ModelShapeAndBreed(n_betas=self.n_betas, n_betas_limbs=self.n_betas_limbs, n_breeds=self.n_breeds, n_z=self.n_z, structure_z_to_betas=self.structure_z_to_betas)
        # ------------------------------ LINEAR 3D MODEL ------------------------------
        # 3d model -> from image to 3d parameters {2d keypoints from heatmap, pose, trans, flength}
        self.soft_max = torch.nn.Softmax(dim=1)
        input_size = self.n_keyp*3 + self.n_bones
        self.model_3d = LinearModelComplete(linear_size=1024,
                    num_stage_comb=num_stage_comb,
                    num_stage_heads=num_stage_heads,
                    num_stage_heads_pose=num_stage_heads_pose,
                    trans_sep=trans_sep, 
                    p_dropout=p_dropout,        # 0.5, 
                    input_size=input_size,
                    intermediate_size=1024,
                    output_info=None,
                    n_joints=n_joints,
                    n_z=self.n_z,
                    add_z_to_3d_input=self.add_z_to_3d_input,
                    n_segbps=self.n_segbps,
                    add_segbps_to_3d_input=self.add_segbps_to_3d_input, 
                    structure_pose_net=self.structure_pose_net,
                    nf_version = self.nf_version)
        # ------------------------------ RENDERING ------------------------------
        self.silh_renderer = SilhRenderer(image_size) 

    def forward(self, input_img, norm_dict=None, bone_lengths_prepared=None, betas=None):
        time1=ckpt_time_v1()
        batch_size = input_img.shape[0]
        device = input_img.device
        # ------------------------------ STACKED HOUR GLASS ------------------------------
        time_stacked_hourglass1=ckpt_time_v1()
        hourglass_out_dict = self.stacked_hourglass(input_img)
        ckpt,time_stacked_hourglass2=ckpt_time_v1(time_stacked_hourglass1,display=True,desc="====ModelImageTo3d_withshape_withproj time_stacked_hourglass2 infer:")
        last_seg = hourglass_out_dict['seg_final']
        last_heatmap = hourglass_out_dict['out_list_kp'][-1] 
        # - prepare keypoints (from heatmap)
        # normalize predictions -> from logits to probability distribution
        # last_heatmap_norm = dsnt.spatial_softmax2d(last_heatmap, temperature=torch.tensor(1))
        # keypoints = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=False) + 1   # (bs, 20, 2)
        # keypoints_norm = dsnt.spatial_expectation2d(last_heatmap_norm, normalized_coordinates=True)    # (bs, 20, 2)
        keypoints_norm, scores = get_preds_soft(last_heatmap, return_maxval=True, norm_coords=True)
        if self.threshold_scores is not None:
            scores[scores>self.threshold_scores] = 1.0
            scores[scores<=self.threshold_scores] = 0.0
        ckpt,time2=ckpt_time_v1(time1,display=True,desc="====forward: eypoints_norm, scores = get_preds_soft(last_heatmap...):")

        # ------------------------------ LEARNABLE SHAPE MODEL ------------------------------
        # in our cvpr 2022 paper we do not change the shapedirs
        # learnable_sd_complete has shape (3889, 3, n_sd)
        # learnable_sd_complete_prepared has shape (n_sd, 11667)
        learnable_sd_complete, learnable_sd_complete_prepared = self.model_learnable_shapedirs()
        shapedirs_sel = learnable_sd_complete_prepared        # None
        # ------------------------------ SHAPE AND BREED MODEL ------------------------------
        # breed_model takes as input the image as well as the predicted segmentation map 
        #     -> we need to split up ModelImageTo3d, such that we can use the silhouette
        time_breed1=ckpt_time_v1()
        resnet_output = self.breed_model(img=input_img, seg_raw=last_seg)
        ckpt,time_breed2=ckpt_time_v1(time_breed1,display=True,desc="====ModelImageTo3d_withshape_withproj breed_model infer: breed,z,beta,beta_limbs")
        pred_breed = resnet_output['breeds']       # (bs, n_breeds)
        pred_z = resnet_output['z']
        # - prepare shape
        pred_betas = resnet_output['betas']     
        pred_betas_limbs = resnet_output['betas_limbs'] 
        # - calculate bone lengths
        with torch.no_grad():
            use_mean_bone_lengths = False
            if use_mean_bone_lengths:
                bone_lengths_prepared = torch.cat(batch_size*[self.mean_dog_bone_lengths.reshape((1, -1))])
            else:
                assert (bone_lengths_prepared is None)
                bone_lengths_prepared = self.smal.caclulate_bone_lengths(pred_betas, pred_betas_limbs, shapedirs_sel=shapedirs_sel, short=True)
        
        ckpt,time3=ckpt_time_v1(time2,display=True,desc="====forward:         # - calculate bone lengths:")

        # ------------------------------ LINEAR 3D MODEL ------------------------------
        # 3d model -> from image to 3d parameters {2d keypoints from heatmap, pose, trans, flength}
        # prepare input for 2d-to-3d network
        keypoints_prepared = torch.cat((keypoints_norm, scores), axis=2)
        if bone_lengths_prepared is None:
            bone_lengths_prepared = torch.cat(batch_size*[self.mean_dog_bone_lengths.reshape((1, -1))])
        # should we add silhouette to 3d input? should we add z?
        if self.add_segbps_to_3d_input:
            seg_raw = last_seg
            seg_prep_bps = self.soft_max(seg_raw)[:, 1, :, :] # class 1 is the dog
            with torch.no_grad():
                seg_prep_np = seg_prep_bps.detach().cpu().numpy()
                bps_output_np = self.segbps_model.calculate_bps_points_batch(seg_prep_np)  # (bs, 64, 2)
                bps_output = torch.tensor(bps_output_np, dtype=torch.float32).to(device).reshape((batch_size, -1))
                bps_output_prep = bps_output * 2. - 1
            input_vec_keyp_bones = torch.cat((keypoints_prepared.reshape((batch_size, -1)), bone_lengths_prepared), axis=1)  
            input_vec = torch.cat((input_vec_keyp_bones, bps_output_prep), dim=1)
        elif self.add_z_to_3d_input:
            # we do not use this in our cvpr 2022 version
            input_vec_keyp_bones = torch.cat((keypoints_prepared.reshape((batch_size, -1)), bone_lengths_prepared), axis=1)  
            input_vec_additional = pred_z       
            input_vec = torch.cat((input_vec_keyp_bones, input_vec_additional), dim=1)
        else:
            input_vec = torch.cat((keypoints_prepared.reshape((batch_size, -1)), bone_lengths_prepared), axis=1)  
        # predict 3d parameters (those are normalized, we need to correct mean and std in a next step)
        time_3d1=ckpt_time_v1()
        output = self.model_3d(input_vec)  
        ckpt,time_3d2=ckpt_time_v1(time_3d1,display=True,desc="====infer: predict 3d parameters infer:")    
        ckpt,time4=ckpt_time_v1(time3,display=True,desc="====forward:predict 3d parameters:")

        # add predicted keypoints to the output dict
        output['keypoints_norm'] = keypoints_norm
        output['keypoints_scores'] = scores
        # - denormalize 3d parameters -> so far predictions were normalized, now we denormalize them again
        pred_trans = output['trans'] * norm_dict['trans_std'][None, :] + norm_dict['trans_mean'][None, :]    # (bs, 3)
        if  self.structure_pose_net == 'default':
            pred_pose_rot6d = output['pose'] + norm_dict['pose_rot6d_mean'][None, :]
        elif self.structure_pose_net == 'normflow':
            pose_rot6d_mean_zeros = torch.zeros_like(norm_dict['pose_rot6d_mean'][None, :])
            pose_rot6d_mean_zeros[:, 0, :] = norm_dict['pose_rot6d_mean'][None, 0, :]
            pred_pose_rot6d = output['pose'] + pose_rot6d_mean_zeros
        else:
            pose_rot6d_mean_zeros = torch.zeros_like(norm_dict['pose_rot6d_mean'][None, :])
            pose_rot6d_mean_zeros[:, 0, :] = norm_dict['pose_rot6d_mean'][None, 0, :]
            pred_pose_rot6d = output['pose'] + pose_rot6d_mean_zeros
        pred_pose_reshx33 = rot6d_to_rotmat(pred_pose_rot6d.reshape((-1, 6)))
        pred_pose = pred_pose_reshx33.reshape((batch_size, -1, 3, 3))
        pred_pose_rot6d = rotmat_to_rot6d(pred_pose_reshx33).reshape((batch_size, -1, 6))

        if self.fix_flength:
            output['flength'] = torch.zeros_like(output['flength'])
            pred_flength = torch.ones_like(output['flength'])*2100  # norm_dict['flength_mean'][None, :]
        else:
            pred_flength_orig = output['flength'] * norm_dict['flength_std'][None, :] + norm_dict['flength_mean'][None, :]   # (bs, 1)
            pred_flength = pred_flength_orig.clone()  # torch.abs(pred_flength_orig)
            pred_flength[pred_flength_orig<=0] = norm_dict['flength_mean'][None, :]
        ckpt,time5=ckpt_time_v1(time4,display=True,desc="====forward:denormalize 3d parameters:")

        # ------------------------------ RENDERING ------------------------------
                ## save smal input to json
        # print(f"pred_betas:{type(pred_betas)}   {pred_betas}")
        smal_pms=[]
        barc_json_path="smal_pms_barc_5555.json"
        if os.path.exists(barc_json_path):
            with open(barc_json_path, 'r', encoding ='utf8') as f:
                smal_pms=json.load(f)
        preds4={}
        preds4['pose'] = pred_pose[0].reshape(-1).cpu().tolist()        
        preds4['betas'] = pred_betas[0].reshape(-1).cpu().tolist()
        # preds4['camera'] = "null"
        preds4['trans'] = pred_trans[0].reshape(-1).cpu().tolist()
        
        # print(f"{len(pose_rot6d_mean_zeros[0])} pose_rot6d_mean_zeros[0].shape  {pose_rot6d_mean_zeros[0].shape}")
        # preds4['pose_rot6d_mean_zeros[0].shape']=f"{pose_rot6d_mean_zeros[0].shape}"
        # preds4['pose_rot6d_mean_zeros']=pose_rot6d_mean_zeros[0].reshape(1,210).cpu().tolist()
        # preds4['pose_output']=output['pose'][0].reshape(1,210).cpu().tolist()
        # preds4['pred_pose_rot6d']=pred_pose_rot6d[0].reshape(1,210).cpu().tolist()
        # preds4['pred_betas_limbs'] = pred_betas_limbs[0].reshape(1,7).cpu().tolist()
        # preds4['flength_unnorm'] =pred_flength.tolist()   
        preds4['flength'] =  output['flength'].reshape(-1).tolist() 
        preds4['breeds']=pred_breed.cpu().reshape(-1).tolist()                
        # print(f"preds4.shape:{len(preds4)} \n ")
        smal_pms.append(preds4)
        # print(f"len(smal_pms):{len(smal_pms)}  \n {smal_pms}")

        # print(f"pred_betas_json: {pred_betas_json}")
        with open(barc_json_path, 'w', encoding ='utf8') as f:
            json.dump(smal_pms,f)

        # get 3d model (SMAL)
        time_3d_smal1=ckpt_time_v1()
        V, keyp_green_3d, _ = self.smal(beta=pred_betas, betas_limbs=pred_betas_limbs, pose=pred_pose, trans=pred_trans, get_skin=True, keyp_conf='green', shapedirs_sel=shapedirs_sel)
        keyp_3d = keyp_green_3d[:, :self.n_keyp, :]     # (bs, 20, 3)
        ckpt,time_3d_smal2=ckpt_time_v1(time_3d_smal1,display=True,desc="====rendering :get 3d model (SMAL)")
        # render silhouette
        faces_prep = self.smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
        if not self.silh_no_tail:
            pred_silh_images, pred_keyp = self.silh_renderer(vertices=V, 
                points=keyp_3d, faces=faces_prep, focal_lengths=pred_flength)
        else:
            faces_no_tail_prep = torch.tensor(self.f_no_tail_np).to(device).expand((batch_size, -1, -1))
            pred_silh_images, pred_keyp = self.silh_renderer(vertices=V, 
                points=keyp_3d, faces=faces_no_tail_prep, focal_lengths=pred_flength)
        # get torch 'Meshes'
        torch_meshes = self.silh_renderer.get_torch_meshes(vertices=V, faces=faces_prep) 

        #  render body parts (not part of cvpr 2022 version)
        if self.render_partseg:
            raise NotImplementedError
        else:
            partseg_images = None
            partseg_images_hg = None
        ckpt,time6=ckpt_time_v1(time5,display=True,desc="====forward:RENDERING:")

        # ------------------------------ PREPARE OUTPUT ------------------------------
        # create output dictionarys
        # output: contains all output from model_image_to_3d
        # output_unnorm: same as output, but normalizations are undone
        # output_reproj: smal output and reprojected keypoints as well as silhouette 
        keypoints_heatmap_256 = (output['keypoints_norm'] / 2. + 0.5) * (self.image_size - 1)
        output_unnorm = {'pose_rotmat': pred_pose,
                        'flength': pred_flength,
                        'trans': pred_trans,
                        'keypoints':keypoints_heatmap_256}
        output_reproj = {'vertices_smal': V,
                        'torch_meshes': torch_meshes,
                        'keyp_3d': keyp_3d,
                        'keyp_2d': pred_keyp,
                        'silh': pred_silh_images,
                        'betas': pred_betas,
                        'betas_limbs': pred_betas_limbs,
                        'pose_rot6d': pred_pose_rot6d,       # used for pose prior...
                        'dog_breed': pred_breed,
                        'shapedirs': shapedirs_sel,
                        'z': pred_z,
                        'flength_unnorm': pred_flength,
                        'flength': output['flength'],
                        'partseg_images_rend': partseg_images,
                        'partseg_images_hg_nograd': partseg_images_hg,
                        'normflow_z': output['normflow_z']}
        ckpt,time7=ckpt_time_v1(time6,display=True,desc="====forward:PREPARE OUTPUT :")

        return output, output_unnorm, output_reproj

    def render_vis_nograd(self, vertices, focal_lengths, color=0):
        # this function is for visualization only
        # vertices: (bs, n_verts, 3)
        # focal_lengths: (bs, 1)
        # color: integer, either 0 or 1
        # returns a torch tensor of shape (bs, image_size, image_size, 3)
        with torch.no_grad():
            batch_size = vertices.shape[0]
            faces_prep = self.smal.faces.unsqueeze(0).expand((batch_size, -1, -1))
            visualizations = self.silh_renderer.get_visualization_nograd(vertices, 
                faces_prep, focal_lengths, color=color)
        return visualizations

