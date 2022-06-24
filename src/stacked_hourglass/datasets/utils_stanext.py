
import os
from matplotlib import pyplot as plt
import glob
import json
import numpy as np
from scipy.io import loadmat
from csv import DictReader
from collections import OrderedDict
from pycocotools.mask import decode as decode_RLE

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.dataset_path_configs import IMG_V12_DIR, JSON_V12_DIR, STAN_V12_TRAIN_LIST_DIR, STAN_V12_VAL_LIST_DIR, STAN_V12_TEST_LIST_DIR


def get_img_dir(V12):
	if V12: 
		return IMG_V12_DIR
	else:
		return IMG_DIR

def get_seg_from_entry(entry):
	"""Given a .json entry, returns the binary mask as a numpy array"""
	rle = {
		"size": [entry['img_height'], entry['img_width']],
		"counts": entry['seg']}
	decoded = decode_RLE(rle)
	return decoded

def full_animal_visible(seg_data):
    if seg_data[0, :].sum() == 0 and seg_data[seg_data.shape[0]-1, :].sum() == 0 and seg_data[:, 0].sum() == 0 and seg_data[:, seg_data.shape[1]-1].sum() == 0:
        return True
    else:
        return False

def load_train_and_test_lists(train_list_dir=None , test_list_dir=None):
	""" returns sets containing names such as 'n02085620-Chihuahua/n02085620_5927.jpg' """
	# train data
	train_list_mat = loadmat(train_list_dir)
	train_list = []
	for ind in range(0, train_list_mat['file_list'].shape[0]):
		name = train_list_mat['file_list'][ind, 0][0]
		train_list.append(name)
	# test data
	test_list_mat = loadmat(test_list_dir)
	test_list = []
	for ind in range(0, test_list_mat['file_list'].shape[0]):
		name = test_list_mat['file_list'][ind, 0][0]
		test_list.append(name)
	return train_list, test_list



def _filter_dict(t_list, j_dict, n_kp_min=4):
	""" should only be used by load_stanext_json_as_dict() """
	out_dict = {}
	for sample in t_list:
		if sample in j_dict.keys():
			n_kp = np.asarray(j_dict[sample]['joints'])[:, 2].sum()
			if n_kp >= n_kp_min:
				out_dict[sample] = j_dict[sample]
	return out_dict

def load_stanext_json_as_dict(split_train_test=True, V12=True):	
	# load json into memory
	if V12:
		with open(JSON_V12_DIR) as infile: 
			json_data = json.load(infile)
		# with open(JSON_V12_DIR) as infile: json_data = json.load(infile, object_pairs_hook=OrderedDict)
	else:
		with open(JSON_DIR) as infile: 
			json_data = json.load(infile)
	# convert json data to a dictionary of img_path : all_data, for easy lookup
	json_dict = {i['img_path']: i for i in json_data}
	if split_train_test:
		if V12:
			train_list_numbers = np.load(STAN_V12_TRAIN_LIST_DIR)
			val_list_numbers = np.load(STAN_V12_VAL_LIST_DIR)
			test_list_numbers = np.load(STAN_V12_TEST_LIST_DIR)
			train_list = [json_data[i]['img_path'] for i in train_list_numbers]
			val_list = [json_data[i]['img_path'] for i in val_list_numbers]
			test_list = [json_data[i]['img_path'] for i in test_list_numbers]
			train_dict = _filter_dict(train_list, json_dict, n_kp_min=4)
			val_dict = _filter_dict(val_list, json_dict, n_kp_min=4)
			test_dict = _filter_dict(test_list, json_dict, n_kp_min=4)
			return train_dict, test_dict, val_dict
		else:
			train_list, test_list = load_train_and_test_lists(train_list_dir=STAN_ORIG_TRAIN_LIST_DIR , test_list_dir=STAN_ORIG_TEST_LIST_DIR) 
			train_dict = _filter_dict(train_list, json_dict)
			test_dict = _filter_dict(test_list, json_dict)
			return train_dict, test_dict, None
	else:
		return json_dict

def get_dog(json_dict, name, img_dir=None):	# (json_dict, name, img_dir=IMG_DIR)
    """ takes the name of a dog, and loads in all the relevant information as a dictionary:
			dict_keys(['img_path', 'img_width', 'img_height', 'joints', 'img_bbox', 
			'is_multiple_dogs', 'seg', 'img_data', 'seg_data']) 
			img_bbox: [x0, y0, width, height] """
    data = json_dict[name]
    # load img
    img_data = plt.imread(os.path.join(img_dir, data['img_path']))
    # load seg
    seg_data = get_seg_from_entry(data)
    # add to output
    data['img_data'] = img_data		# 0 to 255
    data['seg_data'] = seg_data		# 0: bg,   1: fg
    return data





