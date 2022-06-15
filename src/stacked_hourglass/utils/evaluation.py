# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose

import math
import torch
from kornia.geometry.subpix import dsnt     # kornia 0.4.0
import torch.nn.functional as F
from .transforms import transform_preds

__all__ = ['get_preds', 'get_preds_soft', 'calc_dists', 'dist_acc', 'accuracy', 'final_preds_untransformed',
           'final_preds', 'AverageMeter']

def get_preds(scores, return_maxval=False):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()    # values > 0
    preds *= pred_mask
    if return_maxval:
        return preds, maxval
    else:
        return preds


def get_preds_soft(scores, return_maxval=False, norm_coords=False, norm_and_unnorm_coords=False):
    ''' get predictions from score maps in torch Tensor
        predictions are made assuming a logit output map
        return type: torch.LongTensor
    '''

    # New: work on logit predictions
    scores_norm = dsnt.spatial_softmax2d(scores, temperature=torch.tensor(1))
    # maxval_norm, idx_norm = torch.max(scores_norm.view(scores.size(0), scores.size(1), -1), 2)
    # from unnormalized to normalized see:
    # from -1to1 to 0to64
    # see https://github.com/kornia/kornia/blob/b9ffe7efcba7399daeeb8028f10c22941b55d32d/kornia/utils/grid.py#L7 (line 40)
    # xs = (xs / (width - 1) - 0.5) * 2
    # ys = (ys / (height - 1) - 0.5) * 2

    device = scores.device

    if return_maxval:
        preds_normalized = dsnt.spatial_expectation2d(scores_norm, normalized_coordinates=True) 
        # grid_sample(input, grid, mode='bilinear', padding_mode='zeros')
        gs_input_single = scores_norm.reshape((-1, 1, scores_norm.shape[2], scores_norm.shape[3]))     # (120, 1, 64, 64)
        gs_input = scores_norm.reshape((-1, 1, scores_norm.shape[2], scores_norm.shape[3]))     # (120, 1, 64, 64)

        half_pad = 2
        gs_input_single_padded = F.pad(input=gs_input_single, pad=(half_pad, half_pad, half_pad, half_pad, 0, 0, 0, 0), mode='constant', value=0)
        gs_input_all = torch.zeros((gs_input_single.shape[0], 9, gs_input_single.shape[2], gs_input_single.shape[3])).to(device)
        ind_tot = 0
        for ind0 in [-1, 0, 1]:
            for ind1 in [-1, 0, 1]:
                gs_input_all[:, ind_tot, :, :] = gs_input_single_padded[:, 0, half_pad+ind0:-half_pad+ind0, half_pad+ind1:-half_pad+ind1]
                ind_tot +=1

        gs_grid = preds_normalized.reshape((-1, 2))[:, None, None, :]                           # (120, 1, 1, 2)
        gs_output_all = F.grid_sample(gs_input_all, gs_grid, mode='nearest', padding_mode='zeros', align_corners=True).reshape((gs_input_all.shape[0], gs_input_all.shape[1], 1))
        gs_output = gs_output_all.sum(axis=1)
        # scores_norm[0, :, :, :].max(axis=2)[0].max(axis=1)[0]
        # gs_output[0, :, 0]
        gs_output_resh = gs_output.reshape((scores_norm.shape[0], scores_norm.shape[1], 1))

        if norm_and_unnorm_coords:
            preds = dsnt.spatial_expectation2d(scores_norm, normalized_coordinates=False) + 1
            return preds, preds_normalized, gs_output_resh      
        elif norm_coords:
            return preds_normalized, gs_output_resh
        else:
            preds = dsnt.spatial_expectation2d(scores_norm, normalized_coordinates=False) + 1
            return preds, gs_output_resh
    else:
        if norm_coords:
            preds_normalized = dsnt.spatial_expectation2d(scores_norm, normalized_coordinates=True) 
            return preds_normalized
        else:
            preds = dsnt.spatial_expectation2d(scores_norm, normalized_coordinates=False) + 1
            return preds


def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dist, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1

def accuracy(output, target, idxs=None, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    if idxs is None:
        idxs = list(range(target.shape[-3]))
    preds   = get_preds_soft(output)     # get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]], thr=thr)
        if acc[i+1] >= 0:
            avg_acc = avg_acc + acc[i+1]
            cnt += 1

    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc

def final_preds_untransformed(output, res):
    coords = get_preds_soft(output)     # get_preds(output) # float type

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5

    if coords.dim() < 3:
        coords = coords.unsqueeze(0)

    coords -= 1  # Convert from 1-based to 0-based coordinates

    return coords

def final_preds(output, center, scale, res):
    coords = final_preds_untransformed(output, res)
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.unsqueeze(0)

    return preds


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
