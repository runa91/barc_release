

import numpy as np
import os
import sys


# stanext dataset
# (1) path to stanext dataset
STAN_V12_ROOT_DIR = '...../StanfordExtra_V12/'
IMG_V12_DIR = os.path.join(STAN_V12_ROOT_DIR, 'StanExtV12_Images')		      
JSON_V12_DIR = os.path.join(STAN_V12_ROOT_DIR, 'labels', "StanfordExtra_v12.json")    
STAN_V12_TRAIN_LIST_DIR = os.path.join(STAN_V12_ROOT_DIR, 'labels', 'train_stanford_StanfordExtra_v12.npy')  
STAN_V12_VAL_LIST_DIR = os.path.join(STAN_V12_ROOT_DIR, 'labels', 'val_stanford_StanfordExtra_v12.npy')  
STAN_V12_TEST_LIST_DIR = os.path.join(STAN_V12_ROOT_DIR, 'labels', 'test_stanford_StanfordExtra_v12.npy')  
# (2) path to related data such as breed indices and prepared predictions for withers, throat and eye keypoints 
STANEXT_RELATED_DATA_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stanext_related_data')

# test image crop dataset
TEST_IMAGE_CROP_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'test_image_crops') 
