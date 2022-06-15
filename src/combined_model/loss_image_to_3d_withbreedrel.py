

import torch
import numpy as np
import pickle as pkl

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
# from priors.pose_prior_35 import Prior
# from priors.tiger_pose_prior.tiger_pose_prior import GaussianMixturePrior
from priors.normalizing_flow_prior.normalizing_flow_prior import NormalizingFlowPrior
from priors.shape_prior import ShapePrior
from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, batch_rot2aa
from configs.SMAL_configs import UNITY_SMAL_SHAPE_PRIOR_DOGS  

class Loss(torch.nn.Module):
    def __init__(self, data_info, nf_version=None):
        super(Loss, self).__init__()
        self.criterion_regr = torch.nn.MSELoss()        # takes the mean   
        self.criterion_class = torch.nn.CrossEntropyLoss()
        self.data_info = data_info   
        self.register_buffer('keypoint_weights', torch.tensor(data_info.keypoint_weights)[None, :])
        self.l_anchor = None
        self.l_pos = None
        self.l_neg = None

        if nf_version is not None:
            self.normalizing_flow_pose_prior = NormalizingFlowPrior(nf_version=nf_version)
        self.shape_prior = ShapePrior(UNITY_SMAL_SHAPE_PRIOR_DOGS)
        self.criterion_triplet = torch.nn.TripletMarginLoss(margin=1)

        # load 3d data for the unity dogs (an optional shape prior for 11 breeds)
        with open(UNITY_SMAL_SHAPE_PRIOR_DOGS, 'rb') as f:
            data = pkl.load(f)
        dog_betas_unity = data['dogs_betas']
        self.dog_betas_unity = {29: torch.tensor(dog_betas_unity[0, :]).float(),
                            91: torch.tensor(dog_betas_unity[1, :]).float(),
                            84: torch.tensor(0.5*dog_betas_unity[3, :] + 0.5*dog_betas_unity[14, :]).float(),
                            85: torch.tensor(dog_betas_unity[5, :]).float(),
                            28: torch.tensor(dog_betas_unity[6, :]).float(),
                            94: torch.tensor(dog_betas_unity[7, :]).float(),
                            92: torch.tensor(dog_betas_unity[8, :]).float(),
                            95: torch.tensor(dog_betas_unity[10, :]).float(),
                            20: torch.tensor(dog_betas_unity[11, :]).float(),
                            83: torch.tensor(dog_betas_unity[12, :]).float(),
                            99: torch.tensor(dog_betas_unity[16, :]).float()}

    def prepare_anchor_pos_neg(self, batch_size, device):
        l0 = np.arange(0, batch_size, 2)
        l_anchor = []
        l_pos = []
        l_neg = []
        for ind in l0:
            xx = set(np.arange(0, batch_size))
            xx.discard(ind)
            xx.discard(ind+1)
            for ind2 in xx:
                if ind2 % 2 == 0:
                    l_anchor.append(ind)
                    l_pos.append(ind + 1)
                else:
                    l_anchor.append(ind + 1)
                    l_pos.append(ind)
                l_neg.append(ind2)
        self.l_anchor = torch.Tensor(l_anchor).to(torch.int64).to(device)
        self.l_pos = torch.Tensor(l_pos).to(torch.int64).to(device)
        self.l_neg = torch.Tensor(l_neg).to(torch.int64).to(device)
        return


    def forward(self, output_reproj, target_dict, weight_dict=None):
        # output_reproj: ['vertices_smal', 'keyp_3d', 'keyp_2d', 'silh_image']
        # target_dict: ['index', 'center', 'scale', 'pts', 'tpts', 'target_weight']
        batch_size = output_reproj['keyp_2d'].shape[0]

        # loss on reprojected keypoints 
        output_kp_resh = (output_reproj['keyp_2d']).reshape((-1, 2))    
        target_kp_resh = (target_dict['tpts'][:, :, :2] / 64. * (256. - 1)).reshape((-1, 2))
        weights_resh = target_dict['tpts'][:, :, 2].reshape((-1)) 
        keyp_w_resh = self.keypoint_weights.repeat((batch_size, 1)).reshape((-1))
        loss_keyp = ((((output_kp_resh - target_kp_resh)[weights_resh>0]**2).sum(axis=1).sqrt()*weights_resh[weights_resh>0])*keyp_w_resh[weights_resh>0]).sum() / \
            max((weights_resh[weights_resh>0]*keyp_w_resh[weights_resh>0]).sum(), 1e-5)

        # loss on reprojected silhouette
        assert output_reproj['silh'].shape == (target_dict['silh'][:, None, :, :]).shape
        silh_loss_type = 'default'
        if silh_loss_type == 'default':
            with torch.no_grad():
                thr_silh = 20
                diff = torch.norm(output_kp_resh - target_kp_resh, dim=1)
                diff_x = diff.reshape((batch_size, -1))
                weights_resh_x = weights_resh.reshape((batch_size, -1))
                unweighted_kp_mean_dist = (diff_x * weights_resh_x).sum(dim=1) / ((weights_resh_x).sum(dim=1)+1e-6)
            loss_silh_bs = ((output_reproj['silh'] - target_dict['silh'][:, None, :, :]) ** 2).sum(axis=3).sum(axis=2).sum(axis=1) / (output_reproj['silh'].shape[2]*output_reproj['silh'].shape[3])
            loss_silh = loss_silh_bs[unweighted_kp_mean_dist<thr_silh].sum() / batch_size
        else:
            print('silh_loss_type: ' + silh_loss_type)
            raise ValueError

        # shape regularization
        #   'smal': loss on betas (pca coefficients), betas should be close to 0
        #   'limbs...' loss on selected betas_limbs
        loss_shape_weighted_list = [torch.zeros((1)).mean().to(output_reproj['keyp_2d'].device)]  
        for ind_sp, sp in enumerate(weight_dict['shape_options']):
            weight_sp = weight_dict['shape'][ind_sp]
            # self.logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
            if sp == 'smal':
                loss_shape_tmp = self.shape_prior(output_reproj['betas'])
            elif sp == 'limbs':
                loss_shape_tmp = torch.mean((output_reproj['betas_limbs'])**2)  
            elif sp == 'limbs7':
                limb_coeffs_list = [0.01, 1, 0.1, 1, 1, 0.1, 2]
                limb_coeffs = torch.tensor(limb_coeffs_list).to(torch.float32).to(target_dict['tpts'].device)   
                loss_shape_tmp = torch.mean((output_reproj['betas_limbs'] * limb_coeffs[None, :])**2)            
            else:
                raise NotImplementedError
            loss_shape_weighted_list.append(weight_sp * loss_shape_tmp)
        loss_shape_weighted = torch.stack((loss_shape_weighted_list)).sum()

        # 3D loss for dogs for which we have a unity model or toy figure
        loss_models3d = torch.zeros((1)).mean().to(output_reproj['betas'].device)
        if 'models3d' in weight_dict.keys():
            if weight_dict['models3d'] > 0:
                for ind_dog in range(target_dict['breed_index'].shape[0]):
                    breed_index = np.asscalar(target_dict['breed_index'][ind_dog].detach().cpu().numpy())
                    if breed_index in self.dog_betas_unity.keys():
                        betas_target = self.dog_betas_unity[breed_index][:output_reproj['betas'].shape[1]].to(output_reproj['betas'].device)
                        betas_output = output_reproj['betas'][ind_dog, :]
                        betas_limbs_output = output_reproj['betas_limbs'][ind_dog, :]
                        loss_models3d += ((betas_limbs_output**2).sum() + ((betas_output-betas_target)**2).sum()) / (output_reproj['betas'].shape[1] + output_reproj['betas_limbs'].shape[1])
        else:
            weight_dict['models3d'] = 0

        # shape resularization loss on shapedirs
        #   -> in the current version shapedirs are kept fixed, so we don't need those losses
        if weight_dict['shapedirs'] > 0:
            raise NotImplementedError  
        else:
            loss_shapedirs = torch.zeros((1)).mean().to(output_reproj['betas'].device)

        # prior on back joints (not used in cvpr 2022 paper)
        #   -> elementwise MSE loss on all 6 coefficients of 6d rotation representation
        if 'pose_0' in weight_dict.keys(): 
            if weight_dict['pose_0'] > 0:
                pred_pose_rot6d = output_reproj['pose_rot6d']
                w_rj_np = np.zeros((pred_pose_rot6d.shape[1]))
                w_rj_np[[2, 3, 4, 5]] = 1.0         # back
                w_rj = torch.tensor(w_rj_np).to(torch.float32).to(pred_pose_rot6d.device)     
                zero_rot = torch.tensor([1, 0, 0, 1, 0, 0]).to(pred_pose_rot6d.device).to(torch.float32)[None, None, :].repeat((batch_size, pred_pose_rot6d.shape[1], 1))
                loss_pose = self.criterion_regr(pred_pose_rot6d*w_rj[None, :, None], zero_rot*w_rj[None, :, None])
            else:
                loss_pose = torch.zeros((1)).mean()

        # pose prior 
        #   -> we did experiment with different pose priors, for example:
        #       * similart to SMALify (https://github.com/benjiebob/SMALify/blob/master/smal_fitter/smal_fitter.py, 
        #         https://github.com/benjiebob/SMALify/blob/master/smal_fitter/priors/pose_prior_35.py)
        #       * vae 
        #       * normalizing flow pose prior
        #   -> our cvpr 2022 paper uses the normalizing flow pose prior as implemented below
        if 'poseprior' in weight_dict.keys():
            if weight_dict['poseprior'] > 0:
                pred_pose_rot6d = output_reproj['pose_rot6d']
                pred_pose = rot6d_to_rotmat(pred_pose_rot6d.reshape((-1, 6))).reshape((batch_size, -1, 3, 3))
                if 'normalizing_flow_tiger' in weight_dict['poseprior_options']:
                    if output_reproj['normflow_z'] is not None:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss_from_z(output_reproj['normflow_z'], type='square')
                    else:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss(pred_pose_rot6d, type='square')
                elif 'normalizing_flow_tiger_logprob' in weight_dict['poseprior_options']:
                    if output_reproj['normflow_z'] is not None:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss_from_z(output_reproj['normflow_z'], type='neg_log_prob')
                    else:
                        loss_poseprior = self.normalizing_flow_pose_prior.calculate_loss(pred_pose_rot6d, type='neg_log_prob')
                else:
                    raise NotImplementedError
            else:
                loss_poseprior = torch.zeros((1)).mean()
        else:
            weight_dict['poseprior'] = 0
            loss_poseprior = torch.zeros((1)).mean()

        # add a prior which penalizes side-movement angles for legs 
        if 'poselegssidemovement' in weight_dict.keys():
            use_pose_legs_side_loss = True
        else:
            use_pose_legs_side_loss = False
        if use_pose_legs_side_loss:
            leg_indices_right = np.asarray([7, 8, 9, 10, 17, 18, 19, 20])      # front, back
            leg_indices_left = np.asarray([11, 12, 13, 14, 21, 22, 23, 24])     # front, back
            vec = torch.zeros((3, 1)).to(device=pred_pose.device, dtype=pred_pose.dtype)
            vec[2] = -1
            x0_rotmat = pred_pose   
            x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
            x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]
            x0_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3))@vec
            x0_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3))@vec
            eps=0       # 1e-7
            # use the component of the vector which points to the side
            loss_poselegssidemovement = (x0_legs_left[:, 1]**2).mean() + (x0_legs_right[:, 1]**2).mean()
        else:
            loss_poselegssidemovement = torch.zeros((1)).mean()
            weight_dict['poselegssidemovement'] = 0

        # dog breed classification loss
        dog_breed_gt = target_dict['breed_index']
        dog_breed_pred = output_reproj['dog_breed']
        loss_class = self.criterion_class(dog_breed_pred, dog_breed_gt)

        # dog breed relationship loss
        #   -> we did experiment with many other options, but none was significantly better 
        if '4' in weight_dict['breed_options']:      # we have pairs of dogs of the same breed 
            assert weight_dict['breed'] > 0
            z = output_reproj['z']   
            # go through all pairs and compare them to each other sample
            if self.l_anchor is None:
                self.prepare_anchor_pos_neg(batch_size, z.device)
            anchor = torch.index_select(z, 0, self.l_anchor)
            positive = torch.index_select(z, 0, self.l_pos)
            negative = torch.index_select(z, 0, self.l_neg)
            loss_breed = self.criterion_triplet(anchor, positive, negative)
        else:
            loss_breed = torch.zeros((1)).mean()

        # regularizarion for focal length
        loss_flength_near_mean = torch.mean(output_reproj['flength']**2)
        loss_flength = loss_flength_near_mean

        # bodypart segmentation loss
        if 'partseg' in weight_dict.keys():
            if weight_dict['partseg'] > 0:
                raise NotImplementedError
            else:
                loss_partseg = torch.zeros((1)).mean()
        else:
            weight_dict['partseg'] = 0
            loss_partseg = torch.zeros((1)).mean()

        # weight and combine losses
        loss_keyp_weighted = loss_keyp * weight_dict['keyp']
        loss_silh_weighted = loss_silh * weight_dict['silh']
        loss_shapedirs_weighted = loss_shapedirs * weight_dict['shapedirs']
        loss_pose_weighted = loss_pose * weight_dict['pose_0']
        loss_class_weighted = loss_class * weight_dict['class']
        loss_breed_weighted = loss_breed * weight_dict['breed']
        loss_flength_weighted = loss_flength * weight_dict['flength']
        loss_poseprior_weighted = loss_poseprior * weight_dict['poseprior']
        loss_partseg_weighted = loss_partseg * weight_dict['partseg']
        loss_models3d_weighted = loss_models3d * weight_dict['models3d']
        loss_poselegssidemovement_weighted = loss_poselegssidemovement * weight_dict['poselegssidemovement']

        ####################################################################################################
        loss = loss_keyp_weighted + loss_silh_weighted + loss_shape_weighted + loss_pose_weighted + loss_class_weighted + \
                loss_shapedirs_weighted + loss_breed_weighted + loss_flength_weighted + loss_poseprior_weighted + \
                loss_partseg_weighted + loss_models3d_weighted + loss_poselegssidemovement_weighted
        ####################################################################################################
        
        loss_dict = {'loss': loss.item(), 
                    'loss_keyp_weighted': loss_keyp_weighted.item(), \
                    'loss_silh_weighted': loss_silh_weighted.item(), \
                    'loss_shape_weighted': loss_shape_weighted.item(), \
                    'loss_shapedirs_weighted': loss_shapedirs_weighted.item(), \
                    'loss_pose0_weighted': loss_pose_weighted.item(), \
                    'loss_class_weighted': loss_class_weighted.item(), \
                    'loss_breed_weighted': loss_breed_weighted.item(), \
                    'loss_flength_weighted': loss_flength_weighted.item(), \
                    'loss_poseprior_weighted': loss_poseprior_weighted.item(), \
                    'loss_partseg_weighted': loss_partseg_weighted.item(), \
                    'loss_models3d_weighted': loss_models3d_weighted.item(), \
                    'loss_poselegssidemovement_weighted': loss_poselegssidemovement_weighted.item()}
                    
        return loss, loss_dict




