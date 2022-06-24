

import os
import glob
import numpy as np
import torch
import torch.utils.data as data

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.anipose_data_info import COMPLETE_DATA_INFO        
from stacked_hourglass.utils.imutils import load_image 
from stacked_hourglass.utils.transforms import crop, color_normalize
from stacked_hourglass.utils.pilutil import imresize 
from stacked_hourglass.utils.imutils import im_to_torch
from configs.dataset_path_configs import TEST_IMAGE_CROP_ROOT_DIR
from configs.data_info import COMPLETE_DATA_INFO_24


class ImgCrops(data.Dataset):
    DATA_INFO = COMPLETE_DATA_INFO_24
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]  

    def __init__(self, img_crop_folder='default', image_path=None, is_train=False, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', 
                 do_augment='default', shorten_dataset_to=None, dataset_mode='keyp_only'):
        assert is_train == False
        assert do_augment == 'default' or do_augment == False
        self.inp_res = inp_res
        if img_crop_folder == 'default':
            self.folder_imgs = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'test_image_crops') 
        else:
            self.folder_imgs = img_crop_folder
        name_list = glob.glob(os.path.join(self.folder_imgs, '*.png')) + glob.glob(os.path.join(self.folder_imgs, '*.jpg')) + glob.glob(os.path.join(self.folder_imgs, '*.jpeg'))
        name_list = sorted(name_list)
        self.test_name_list = [name.split('/')[-1] for name in name_list]
        print('len(dataset): ' + str(self.__len__()))

    def __getitem__(self, index):
        img_name = self.test_name_list[index]
        # load image
        img_path = os.path.join(self.folder_imgs, img_name)
        img = load_image(img_path)  # CxHxW
        # prepare image (cropping and color)
        img_max = max(img.shape[1], img.shape[2])
        img_padded = torch.zeros((img.shape[0], img_max, img_max))
        if img_max == img.shape[2]:
            start = (img_max-img.shape[1])//2
            img_padded[:, start:start+img.shape[1], :] = img
        else:
            start = (img_max-img.shape[2])//2
            img_padded[:, :, start:start+img.shape[2]] = img   
        img = img_padded
        img_prep = im_to_torch(imresize(img, [self.inp_res, self.inp_res], interp='bilinear'))   
        inp = color_normalize(img_prep, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)
        # add the following fields to make it compatible with stanext, most of them are fake
        target_dict = {'index': index, 'center' : -2, 'scale' : -2, 
            'breed_index': -2, 'sim_breed_index': -2,
            'ind_dataset': 1}
        target_dict['pts'] = np.zeros((self.DATA_INFO.n_keyp, 3))
        target_dict['tpts'] = np.zeros((self.DATA_INFO.n_keyp, 3))
        target_dict['target_weight'] = np.zeros((self.DATA_INFO.n_keyp, 1))
        target_dict['silh'] = np.zeros((self.inp_res, self.inp_res))
        return inp, target_dict


    def __len__(self):
        return len(self.test_name_list)   









