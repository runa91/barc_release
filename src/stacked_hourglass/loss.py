import torch.nn as nn
from torch.nn.functional import mse_loss


def joints_mse_loss(output, target, target_weight=None):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.view((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.view((batch_size, num_joints, -1)).split(1, 1)

    loss = 0
    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx]
        heatmap_gt = heatmaps_gt[idx]
        if target_weight is None:
            loss += 0.5 * mse_loss(heatmap_pred, heatmap_gt, reduction='mean')
        else:
            loss += 0.5 * mse_loss(
                heatmap_pred.mul(target_weight[:, idx]),
                heatmap_gt.mul(target_weight[:, idx]),
                reduction='mean'
            )

    return loss / num_joints


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        if not self.use_target_weight:
            target_weight = None
        return joints_mse_loss(output, target, target_weight)
