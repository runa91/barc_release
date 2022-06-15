

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


class INNForShape(nn.Module):
    def __init__(self, n_betas, n_betas_limbs, k_tot=2, betas_scale=1.0, betas_limbs_scale=0.1):
        super(INNForShape, self).__init__()
        self.n_betas = n_betas
        self.n_betas_limbs = n_betas_limbs
        self.n_dim = n_betas + n_betas_limbs
        self.betas_scale = betas_scale
        self.betas_limbs_scale = betas_limbs_scale
        self.k_tot = 2
        self.model_inn = self.build_inn_network(self.n_dim, k_tot=self.k_tot) 

    def subnet_fc(self, c_in, c_out):
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

    def forward(self, latent_rep): 
        shape, _ = self.model_inn(latent_rep, rev=False, jac=False)
        betas = shape[:, :self.n_betas]*self.betas_scale
        betas_limbs = shape[:, self.n_betas:]*self.betas_limbs_scale
        return betas, betas_limbs

    def reverse(self, betas, betas_limbs):
        shape = torch.cat((betas/self.betas_scale, betas_limbs/self.betas_limbs_scale), dim=1)
        latent_rep, _ = self.model_inn(shape, rev=True, jac=False)
        return latent_rep