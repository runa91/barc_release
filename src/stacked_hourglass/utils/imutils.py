# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose

import numpy as np

from .misc import to_numpy, to_torch
from .pilutil import imread, imresize
from kornia.geometry.subpix import dsnt
import torch

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(imread(img_path, mode='RGB'))

# =============================================================================
# Helpful functions generating groundtruth labelmap
# =============================================================================

def gaussian(shape=(7,7),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return to_torch(h).float()

def draw_labelmap_orig(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    # maximum value of the gaussian is 1
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img), 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
    return to_torch(img), 1



def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # real probability distribution: the sum of all values is 1 
    img = to_numpy(img)
    if not type == 'Gaussian':
        raise NotImplementedError

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img), 0

    # Generate gaussian
    # img_new = dsnt.render_gaussian2d(mean=torch.tensor([[-1, 0]]).float(), std=torch.tensor([[sigma, sigma]]).float(), size=(img.shape[0], img.shape[1]), normalized_coordinates=False)
    img_new = dsnt.render_gaussian2d(mean=torch.tensor([[pt[0], pt[1]]]).float(), \
        std=torch.tensor([[sigma, sigma]]).float(), \
            size=(img.shape[0], img.shape[1]), \
            normalized_coordinates=False)
    img_new = img_new[0, :, :]      # this is a torch image
    return img_new, 1


def draw_multiple_labelmaps(out_res, pts, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # real probability distribution: the sum of all values is 1 
    if not type == 'Gaussian':
        raise NotImplementedError

    # Generate gaussians
    n_pts = pts.shape[0]    
    imgs_new = dsnt.render_gaussian2d(mean=pts[:, :2], \
        std=torch.tensor([[sigma, sigma]]).float().repeat((n_pts, 1)), \
        size=(out_res[0], out_res[1]), \
        normalized_coordinates=False)       # shape: (n_pts, out_res[0], out_res[1])

    visibility_orig = imgs_new.sum(axis=2).sum(axis=1)   # shape: (n_pts)
    visibility = torch.zeros((n_pts, 1), dtype=torch.float32)
    visibility[visibility_orig>=0.99999] = 1.0

    # import pdb; pdb.set_trace()

    return imgs_new, visibility.int()