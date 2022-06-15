
# some parts of the code adapted from https://github.com/benjiebob/WLDO and https://github.com/benjiebob/SMALify

import numpy as np
import torch
import pickle as pkl



class ShapePrior(torch.nn.Module):   
    def __init__(self, prior_path):   
        super(ShapePrior, self).__init__()
        try:
            with open(prior_path, 'r') as f:
                res = pkl.load(f)
        except (UnicodeDecodeError, TypeError) as e:
            with open(prior_path, 'rb') as file:
                u = pkl._Unpickler(file)
                u.encoding = 'latin1'
                res = u.load()
        betas_mean = res['dog_cluster_mean']  
        betas_cov = res['dog_cluster_cov']
        single_gaussian_inv_covs = np.linalg.inv(betas_cov + 1e-5 * np.eye(betas_cov.shape[0]))  
        single_gaussian_precs = torch.tensor(np.linalg.cholesky(single_gaussian_inv_covs)).float()
        single_gaussian_means = torch.tensor(betas_mean).float()
        self.register_buffer('single_gaussian_precs', single_gaussian_precs)    # (20, 20)
        self.register_buffer('single_gaussian_means', single_gaussian_means)    # (20)
        use_ind_tch = torch.from_numpy(np.ones(single_gaussian_means.shape[0], dtype=bool)).float()   # .to(device)
        self.register_buffer('use_ind_tch', use_ind_tch)

    def forward(self, betas_smal_orig, use_singe_gaussian=False):      
        n_betas_smal = betas_smal_orig.shape[1]
        device = betas_smal_orig.device
        use_ind_tch_corrected = self.use_ind_tch * torch.cat((torch.ones_like(self.use_ind_tch[:n_betas_smal]), torch.zeros_like(self.use_ind_tch[n_betas_smal:])))        
        samples = torch.cat((betas_smal_orig, torch.zeros((betas_smal_orig.shape[0], self.single_gaussian_means.shape[0]-n_betas_smal)).float().to(device)), dim=1)
        mean_sub =  samples - self.single_gaussian_means.unsqueeze(0)
        single_gaussian_precs_corr = self.single_gaussian_precs * use_ind_tch_corrected[:, None] * use_ind_tch_corrected[None, :]
        res = torch.tensordot(mean_sub, single_gaussian_precs_corr, dims = ([1], [0]))
        res_final_mean_2 = torch.mean(res ** 2)
        return res_final_mean_2
