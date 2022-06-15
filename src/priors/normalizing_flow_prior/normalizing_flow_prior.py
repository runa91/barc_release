
from torch import distributions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal 
import numpy as np
import cv2
import trimesh
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from configs.barc_cfg_defaults import get_cfg_global_updated


class NormalizingFlowPrior(nn.Module):
    def __init__(self, nf_version=None):
        super(NormalizingFlowPrior, self).__init__()
        # the normalizing flow network takes as input a vector of size (35-1)*6 which is 
        # [all joints except root joint]*6. At the moment the rotation is represented as 6D 
        # representation, which is actually not ideal. Nevertheless, in practice the 
        # results seem to be ok.
        n_dim = (35 - 1) * 6        
        self.param_dict = self.get_version_param_dict(nf_version)
        self.model_inn = self.build_inn_network(n_dim, k_tot=self.param_dict['k_tot']) 
        self.initialize_with_pretrained_weights()

    def get_version_param_dict(self, nf_version):
        # we had trained several version of the normalizing flow pose prior, here we just provide 
        # the option that was user for the cvpr 2022 paper (nf_version=3)
        if nf_version == 3:
            param_dict = {
                'k_tot': 2,
                'path_pretrained': get_cfg_global_updated().paths.MODELPATH_NORMFLOW,
                'subnet_fc_type': '3_64'}  
        else:
            print(nf_version)
            raise ValueError
        return param_dict

    def initialize_with_pretrained_weights(self, weight_path=None):
        # The normalizing flow pose prior is pretrained separately. Afterwards all weights 
        # are kept fixed. Here we load those pretrained weights.
        if weight_path is None:
            weight_path = self.param_dict['path_pretrained']
        print(' normalizing flow pose prior: loading {}..'.format(weight_path))
        pretrained_dict = torch.load(weight_path)['model_state_dict']
        self.model_inn.load_state_dict(pretrained_dict, strict=True)

    def subnet_fc(self, c_in, c_out):
        if self.param_dict['subnet_fc_type']=='3_512':
            subnet = nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                                nn.Linear(512, 512), nn.ReLU(),
                                nn.Linear(512,  c_out))
        elif self.param_dict['subnet_fc_type']=='3_64':
            subnet = nn.Sequential(nn.Linear(c_in, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64,  c_out))
        return subnet

    def build_inn_network(self, n_input, k_tot=12, verbose=False):
        coupling_block = Fm.RNVPCouplingBlock
        nodes = [Ff.InputNode(n_input, name='input')]
        for k in range(k_tot):
            nodes.append(Ff.Node(nodes[-1],
                                coupling_block,
                                {'subnet_constructor':self.subnet_fc, 'clamp':2.0},
                                name=F'coupling_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'permute_{k}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        model = Ff.ReversibleGraphNet(nodes, verbose=verbose)
        return model

    def calculate_loss_from_z(self, z, type='square'):
        assert type in ['square', 'neg_log_prob']
        if type == 'square':
            loss = (z**2).mean()    # * 0.00001
        elif type == 'neg_log_prob':
            means = torch.zeros((z.shape[0], z.shape[1]), dtype=z.dtype, device=z.device)
            stds = torch.ones((z.shape[0], z.shape[1]), dtype=z.dtype, device=z.device)
            normal_distribution = Normal(means, stds)
            log_prob = normal_distribution.log_prob(z)
            loss = - log_prob.mean()
        return loss

    def calculate_loss(self, poses_rot6d, type='square'):
        assert type in ['square', 'neg_log_prob']
        poses_rot6d_noglob = poses_rot6d[:, 1:, :].reshape((-1, 34*6))     
        z, _ = self.model_inn(poses_rot6d_noglob, rev=False, jac=False)
        loss = self.calculate_loss_from_z(z, type=type)
        return loss

    def forward(self, poses_rot6d):
        # from pose to latent pose representation z 
        # poses_rot6d has shape (bs, 34, 6)
        poses_rot6d_noglob = poses_rot6d[:, 1:, :].reshape((-1, 34*6))   
        z, _ = self.model_inn(poses_rot6d_noglob, rev=False, jac=False)
        return z

    def run_backwards(self, z):
        # from latent pose representation z to pose 
        poses_rot6d_noglob, _ = self.model_inn(z, rev=True, jac=False)
        return poses_rot6d_noglob




 