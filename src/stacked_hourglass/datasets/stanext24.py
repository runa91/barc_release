# 24 joints instead of 20!!


import gzip
import json
import os
import random
import math
import numpy as np
import torch
import torch.utils.data as data
from importlib_resources import open_binary
from scipy.io import loadmat
from tabulate import tabulate
import itertools
import json
from scipy import ndimage

from csv import DictReader
from pycocotools.mask import decode as decode_RLE

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.data_info import COMPLETE_DATA_INFO_24
from stacked_hourglass.utils.imutils import load_image, draw_labelmap, draw_multiple_labelmaps
from stacked_hourglass.utils.misc import to_torch
from stacked_hourglass.utils.transforms import shufflelr, crop, color_normalize, fliplr, transform
import stacked_hourglass.datasets.utils_stanext as utils_stanext 
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints
from configs.dog_breeds.dog_breed_class import COMPLETE_ABBREV_DICT, COMPLETE_SUMMARY_BREEDS, SIM_MATRIX_RAW, SIM_ABBREV_INDICES
from configs.dataset_path_configs import STANEXT_RELATED_DATA_ROOT_DIR


class StanExt(data.Dataset):
    DATA_INFO = COMPLETE_DATA_INFO_24

    # Suggested joints to use for keypoint reprojection error calculations 
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]      

    def __init__(self, image_path=None, is_train=True, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', 
                 do_augment='default', shorten_dataset_to=None, dataset_mode='keyp_only', V12=None, val_opt='test'):
        self.V12 = V12
        self.is_train = is_train    # training set or test set
        if do_augment == 'yes':
            self.do_augment = True
        elif do_augment == 'no':
            self.do_augment = False
        elif do_augment=='default':
            if self.is_train:
                self.do_augment = True
            else:
                self.do_augment = False
        else:
            raise ValueError
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.dataset_mode = dataset_mode
        if self.dataset_mode=='complete' or self.dataset_mode=='keyp_and_seg' or self.dataset_mode=='keyp_and_seg_and_partseg':
            self.calc_seg = True
        else:
            self.calc_seg = False
        self.val_opt = val_opt

        # create train/val split
        self.img_folder = utils_stanext.get_img_dir(V12=self.V12)
        self.train_dict, init_test_dict, init_val_dict = utils_stanext.load_stanext_json_as_dict(split_train_test=True, V12=self.V12)
        self.train_name_list = list(self.train_dict.keys())     # 7004
        if self.val_opt == 'test':
            self.test_dict = init_test_dict
            self.test_name_list = list(self.test_dict.keys())        
        elif self.val_opt == 'val':
            self.test_dict = init_val_dict
            self.test_name_list = list(self.test_dict.keys())       
        else:
            raise NotImplementedError

        # stanext breed dict (contains for each name a stanext specific index)
        breed_json_path = os.path.join(STANEXT_RELATED_DATA_ROOT_DIR, 'StanExt_breed_dict_v2.json')
        self.breed_dict = self.get_breed_dict(breed_json_path, create_new_breed_json=False) 
        self.train_name_list = sorted(self.train_name_list)
        self.test_name_list = sorted(self.test_name_list)
        random.seed(4)
        random.shuffle(self.train_name_list)
        random.shuffle(self.test_name_list)
        if shorten_dataset_to is not None:
            # sometimes it is useful to have a smaller set (validation speed, debugging)
            self.train_name_list = self.train_name_list[0 : min(len(self.train_name_list), shorten_dataset_to)]
            self.test_name_list = self.test_name_list[0 : min(len(self.test_name_list), shorten_dataset_to)]
            # special case for debugging: 12 similar images
            if shorten_dataset_to == 12:
                my_sample = self.test_name_list[2]
                for ind in range(0, 12):
                    self.test_name_list[ind] = my_sample
        print('len(dataset): ' + str(self.__len__()))

        # add results for eyes, whithers and throat as obtained through anipose -> they are used
        #   as pseudo ground truth at training time.
        self.path_anipose_out_root = os.path.join(STANEXT_RELATED_DATA_ROOT_DIR, 'animalpose_hg8_v0_results_on_StanExt')

        
    def get_data_sampler_info(self):
        # for custom data sampler
        if self.is_train:
            name_list = self.train_name_list
        else:
            name_list = self.test_name_list 
        info_dict = {'name_list': name_list,
                    'stanext_breed_dict': self.breed_dict,
                    'breeds_abbrev_dict': COMPLETE_ABBREV_DICT, 
                    'breeds_summary': COMPLETE_SUMMARY_BREEDS, 
                    'breeds_sim_martix_raw': SIM_MATRIX_RAW, 
                    'breeds_sim_abbrev_inds': SIM_ABBREV_INDICES
                    }
        return info_dict


    def get_breed_dict(self, breed_json_path, create_new_breed_json=False):
        if create_new_breed_json:
            breed_dict = {}
            breed_index = 0
            for img_name in self.train_name_list: 
                folder_name = img_name.split('/')[0]
                breed_name = folder_name.split(folder_name.split('-')[0] + '-')[1]
                if not (folder_name in breed_dict):
                    breed_dict[folder_name] = {
                        'breed_name': breed_name,
                        'index': breed_index}
                    breed_index += 1
            with open(breed_json_path, 'w', encoding='utf-8') as f: json.dump(breed_dict, f, ensure_ascii=False, indent=4)
        else:
            with open(breed_json_path) as json_file: breed_dict = json.load(json_file)
        return breed_dict


    def __getitem__(self, index):

        if self.is_train:
            name = self.train_name_list[index]
            data = self.train_dict[name]
        else:
            name = self.test_name_list[index]
            data = self.test_dict[name]

        sf = self.scale_factor
        rf = self.rot_factor

        img_path = os.path.join(self.img_folder, data['img_path'])
        try:
            anipose_res_path = os.path.join(self.path_anipose_out_root, data['img_path'].replace('.jpg', '.json'))
            with open(anipose_res_path) as f: anipose_data = json.load(f)
            anipose_thr = 0.2
            anipose_joints_0to24 = np.asarray(anipose_data['anipose_joints_0to24']).reshape((-1, 3))
            anipose_joints_0to24_scores = anipose_joints_0to24[:, 2]
            # anipose_joints_0to24_scores[anipose_joints_0to24_scores>anipose_thr] = 1.0
            anipose_joints_0to24_scores[anipose_joints_0to24_scores<anipose_thr] = 0.0
            anipose_joints_0to24[:, 2] = anipose_joints_0to24_scores
        except:
            # REMARK: This happens sometimes!!! maybe once every 10th image..?
            # print('no anipose eye keypoints!')
            anipose_joints_0to24 = np.zeros((24, 3))

        joints = np.concatenate((np.asarray(data['joints'])[:20, :], anipose_joints_0to24[20:24, :]), axis=0)
        joints[joints[:, 2]==0, :2] = 0     # avoid nan values
        pts = torch.Tensor(joints)

        # inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        # sf = scale * 200.0 / res[0]  # res[0]=256
        # center = center * 1.0 / sf
        # scale = scale / sf = 256 / 200
        # h = 200 * scale
        bbox_xywh = data['img_bbox']
        bbox_c = [bbox_xywh[0]+0.5*bbox_xywh[2], bbox_xywh[1]+0.5*bbox_xywh[3]]
        bbox_max = max(bbox_xywh[2], bbox_xywh[3])
        bbox_diag = math.sqrt(bbox_xywh[2]**2 + bbox_xywh[3]**2)
        # bbox_s = bbox_max / 200.      # the dog will fill the image -> bbox_max = 256
        # bbox_s = bbox_diag / 200.     # diagonal of the boundingbox will be 200
        bbox_s = bbox_max / 200. * 256. / 200.  # maximum side of the bbox will be 200
        c = torch.Tensor(bbox_c)
        s = bbox_s

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        # segmentation map (we reshape it to 3xHxW, such that we can do the 
        #   same transformations as with the image)
        if self.calc_seg:
            seg = torch.Tensor(utils_stanext.get_seg_from_entry(data)[None, :, :])
            seg = torch.cat(3*[seg])

        r = 0
        do_flip = False
        if self.do_augment:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            # Flip
            if random.random() <= 0.5:
                do_flip = True
                img = fliplr(img)
                if self.calc_seg:
                    seg = fliplr(seg)
                pts = shufflelr(pts, img.size(2), self.DATA_INFO.hflip_indices)
                c[0] = img.size(2) - c[0]
            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        img_border_mask = torch.all(inp > 1.0/256, dim = 0).unsqueeze(0).float()        # 1 is foreground
        inp = color_normalize(inp, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)
        if self.calc_seg:
            seg = crop(seg, c, s, [self.inp_res, self.inp_res], rot=r)

        # Generate ground truth
        tpts = pts.clone()
        target_weight = tpts[:, 2].clone().view(nparts, 1)
        
        target = torch.zeros(nparts, self.out_res, self.out_res)
        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r, as_int=False))
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis
        # NEW:
        '''target_new, vis_new = draw_multiple_labelmaps((self.out_res, self.out_res), tpts[:, :2]-1, self.sigma, type=self.label_type)
        target_weight_new = tpts[:, 2].clone().view(nparts, 1) * vis_new
        target_new[(target_weight_new==0).reshape((-1)), :, :] = 0'''
                        
        # --- Meta info
        this_breed = self.breed_dict[name.split('/')[0]]        # 120
        # add information about location within breed similarity matrix
        folder_name = name.split('/')[0]
        breed_name = folder_name.split(folder_name.split('-')[0] + '-')[1]
        abbrev = COMPLETE_ABBREV_DICT[breed_name]
        try:
            sim_breed_index = COMPLETE_SUMMARY_BREEDS[abbrev]._ind_in_xlsx_matrix 
        except: # some breeds are not in the xlsx file
            sim_breed_index = -1
        meta = {'index' : index, 'center' : c, 'scale' : s,
            'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 
            'breed_index': this_breed['index'], 'sim_breed_index': sim_breed_index,
            'ind_dataset': 0}   # ind_dataset=0 for stanext or stanexteasy or stanext 2
        meta2 = {'index' : index, 'center' : c, 'scale' : s,
            'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight, 
           'ind_dataset': 3} 

        # return different things depending on dataset_mode
        if self.dataset_mode=='keyp_only':
            # save_input_image_with_keypoints(inp, meta['tpts'], out_path='./test_input_stanext.png', ratio_in_out=self.inp_res/self.out_res)
            return inp, target, meta
        elif self.dataset_mode=='keyp_and_seg':
            meta['silh'] = seg[0, :, :]
            meta['name'] = name
            return inp, target, meta
        elif self.dataset_mode=='keyp_and_seg_and_partseg':
            # partseg is fake! this does only exist such that this dataset can be combined with an other datset that has part segmentations
            meta2['silh'] = seg[0, :, :]
            meta2['name'] = name
            fake_body_part_matrix = torch.ones((3, 256, 256)).long() * (-1)
            meta2['body_part_matrix'] = fake_body_part_matrix
            return inp, target, meta2
        elif self.dataset_mode=='complete':
            target_dict = meta
            target_dict['silh'] = seg[0, :, :]
            # NEW for silhouette loss
            target_dict['img_border_mask'] = img_border_mask
            target_dict['has_seg'] = True
            if target_dict['silh'].sum() < 1:
                if  ((not self.is_train) and self.val_opt == 'test'):
                    raise ValueError
                elif self.is_train:
                    print('had to replace training image')
                    replacement_index = max(0, index - 1)
                    inp, target_dict = self.__getitem__(replacement_index)                    
                else:
                    # There seem to be a few validation images without segmentation
                    # which would lead to nan in iou calculation
                    replacement_index = max(0, index - 1)
                    inp, target_dict = self.__getitem__(replacement_index)
            return inp, target_dict
        else:
            print('sampling error')
            import pdb; pdb.set_trace()
            raise ValueError


    def __len__(self):
        if self.is_train:
            return len(self.train_name_list) 
        else:
            return len(self.test_name_list)   


