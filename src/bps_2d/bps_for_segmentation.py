
# code idea from https://github.com/sergeyprokudin/bps

import os
import numpy as np
from PIL import Image
import time
import scipy
import scipy.spatial
import pymp


#####################
QUERY_POINTS = np.asarray([30, 34, 31, 55, 29, 84, 35, 108, 34, 145, 29, 171, 27, 
    196, 29, 228, 58, 35, 61, 55, 57, 83, 56, 109, 63, 148, 58, 164, 57, 197, 60, 
    227, 81, 26, 87, 58, 85, 87, 89, 117, 86, 142, 89, 172, 84, 197, 88, 227, 113, 
    32, 116, 58, 112, 88, 118, 113, 109, 147, 114, 173, 119, 201, 113, 229, 139, 
    29, 141, 59, 142, 93, 139, 117, 146, 147, 141, 173, 142, 201, 143, 227, 170, 
    26, 173, 59, 166, 90, 174, 117, 176, 141, 169, 175, 167, 198, 172, 227, 198, 
    30, 195, 59, 204, 85, 198, 116, 195, 140, 198, 175, 194, 193, 199, 227, 221, 
    26, 223, 57, 227, 83, 227, 113, 227, 140, 226, 173, 230, 196, 228, 229]).reshape((64, 2))
#####################

class SegBPS():

    def __init__(self, query_points=QUERY_POINTS, size=256):
        self.size = size
        self.query_points = query_points
        row, col = np.indices((self.size, self.size))
        self.indices_rc = np.stack((row, col), axis=2)   # (256, 256, 2)
        self.pts_aranged = np.arange(64)
        return

    def _do_kdtree(self, combined_x_y_arrays, points):
        # see https://stackoverflow.com/questions/10818546/finding-index-of-nearest-
        #   point-in-numpy-arrays-of-x-and-y-coordinates
        mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
        dist, indexes = mytree.query(points)
        return indexes

    def calculate_bps_points(self, seg, thr=0.5, vis=False, out_path=None):
        # seg: input segmentation image of shape (256, 256) with values between 0 and 1
        query_val = seg[self.query_points[:, 0], self.query_points[:, 1]]
        pts_fg = self.pts_aranged[query_val>=thr]
        pts_bg = self.pts_aranged[query_val<thr]
        candidate_inds_bg = self.indices_rc[seg<thr]
        candidate_inds_fg = self.indices_rc[seg>=thr]
        if candidate_inds_bg.shape[0] == 0:
            candidate_inds_bg = np.ones((1, 2)) * 128        #  np.zeros((1, 2))
        if candidate_inds_fg.shape[0] == 0:
            candidate_inds_fg = np.ones((1, 2)) * 128        #  np.zeros((1, 2))
        # calculate nearest points
        all_nearest_points = np.zeros((64, 2))
        all_nearest_points[pts_fg, :] = candidate_inds_bg[self._do_kdtree(candidate_inds_bg, self.query_points[pts_fg, :]), :]
        all_nearest_points[pts_bg, :] = candidate_inds_fg[self._do_kdtree(candidate_inds_fg, self.query_points[pts_bg, :]), :]
        all_nearest_points_01 = all_nearest_points / 255.
        if vis:
            self.visualize_result(seg, all_nearest_points, out_path=out_path)
        return all_nearest_points_01

    def calculate_bps_points_batch(self, seg_batch, thr=0.5, vis=False, out_path=None):
        # seg_batch: input segmentation image of shape (bs, 256, 256) with values between 0 and 1
        bs = seg_batch.shape[0]
        all_nearest_points_01_batch = np.zeros((bs, self.query_points.shape[0], 2))
        for ind in range(0, bs):         # 0.25
            seg = seg_batch[ind, :, :]
            all_nearest_points_01 = self.calculate_bps_points(seg, thr=thr, vis=vis, out_path=out_path)
            all_nearest_points_01_batch[ind, :, :] = all_nearest_points_01
        return all_nearest_points_01_batch

    def visualize_result(self, seg, all_nearest_points, out_path=None):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        # img: (256, 256, 3) 
        img = (np.stack((seg, seg, seg), axis=2) * 155).astype(np.int)
        if out_path is None:
            ind_img = 0
            out_path = '../test_img' + str(ind_img) + '.png'
        fig, ax = plt.subplots()
        plt.imshow(img)     
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ratio_in_out = 1    # 255
        for idx, (y, x) in enumerate(self.query_points):
            x = int(x*ratio_in_out)     
            y = int(y*ratio_in_out)     
            plt.scatter([x], [y], marker="x", s=50)
            x2 = int(all_nearest_points[idx, 1])
            y2 = int(all_nearest_points[idx, 0])
            plt.scatter([x2], [y2], marker="o", s=50)
            plt.plot([x, x2], [y, y2])
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return





if __name__ == "__main__":
    ind_img = 2 # 4
    path_seg_top = '...../pytorch-stacked-hourglass/results/dogs_hg8_ks_24_v1/test/'
    path_seg = os.path.join(path_seg_top, 'seg_big_' + str(ind_img) + '.png')
    img = np.asarray(Image.open(path_seg))
    # min is 0.004, max is 0.9
    # low values are background, high values are foreground
    seg = img[:, :, 1] / 255.
    # calculate points
    bps = SegBPS()
    bps.calculate_bps_points(seg, thr=0.5, vis=False, out_path=None)


