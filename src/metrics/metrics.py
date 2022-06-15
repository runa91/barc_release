# code from: https://github.com/benjiebob/WLDO/blob/master/wldo_regressor/metrics.py


import torch
import torch.nn.functional as F
import numpy as np

IMG_RES = 256   # in WLDO it is 224

class Metrics():

    @staticmethod
    def PCK_thresh(
        pred_keypoints, gt_keypoints,
        gtseg, has_seg,
        thresh, idxs, biggs=False):

        pred_keypoints, gt_keypoints, gtseg = pred_keypoints[has_seg], gt_keypoints[has_seg], gtseg[has_seg]

        if idxs is None:
            idxs = list(range(pred_keypoints.shape[1]))

        idxs = np.array(idxs).astype(int)

        pred_keypoints = pred_keypoints[:, idxs]
        gt_keypoints = gt_keypoints[:, idxs]

        if biggs: 
            keypoints_gt = ((gt_keypoints + 1.0) * 0.5) * IMG_RES  
            dist = torch.norm(pred_keypoints - keypoints_gt[:, :, [1, 0]], dim = -1)
        else:
            keypoints_gt = gt_keypoints     # (0 to IMG_SIZE)
            dist = torch.norm(pred_keypoints - keypoints_gt[:, :, :2], dim = -1)

        seg_area = torch.sum(gtseg.reshape(gtseg.shape[0], -1), dim = -1).unsqueeze(-1)

        hits = (dist / torch.sqrt(seg_area)) < thresh
        total_visible = torch.sum(gt_keypoints[:, :, -1], dim = -1)
        pck = torch.sum(hits.float() * gt_keypoints[:, :, -1], dim = -1) / total_visible

        return pck

    @staticmethod
    def PCK(
        pred_keypoints, keypoints, 
        gtseg, has_seg, 
        thresh_range=[0.15],
        idxs:list=None,
        biggs=False):
        """Calc PCK with same method as in eval.
        idxs = optional list of subset of keypoints to index from
        """
        cumulative_pck = []
        for thresh in thresh_range:
            pck = Metrics.PCK_thresh(
                pred_keypoints, keypoints, 
                gtseg, has_seg, thresh, idxs, 
                biggs=biggs)
            cumulative_pck.append(pck)
        pck_mean = torch.stack(cumulative_pck, dim = 0).mean(dim=0)
        return pck_mean

    @staticmethod
    def IOU(synth_silhouettes, gt_seg, img_border_mask, mask):
        for i in range(mask.shape[0]):
            synth_silhouettes[i] *= mask[i]
        # Do not penalize parts of the segmentation outside the img range
        gt_seg = (gt_seg * img_border_mask) + synth_silhouettes * (1.0 - img_border_mask)
        intersection = torch.sum((synth_silhouettes * gt_seg).reshape(synth_silhouettes.shape[0], -1), dim = -1)
        union = torch.sum(((synth_silhouettes + gt_seg).reshape(synth_silhouettes.shape[0], -1) > 0.0).float(), dim = -1)
        acc_IOU_SCORE = intersection / union
        if torch.isnan(acc_IOU_SCORE).sum() > 0:
            import pdb; pdb.set_trace()
        return acc_IOU_SCORE