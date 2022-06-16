
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

abs_barc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',))

_C = CN()
_C.barc_dir = abs_barc_dir
_C.device = 'cuda'

## path settings
_C.paths = CN()
_C.paths.ROOT_OUT_PATH = abs_barc_dir + '/results/'
_C.paths.ROOT_CHECKPOINT_PATH = abs_barc_dir + '/checkpoint/'
_C.paths.MODELPATH_NORMFLOW = abs_barc_dir + '/checkpoint/cvpr_normflow_pret/rgbddog_v3_model.pt'

## parameter settings
_C.params = CN()
_C.params.ARCH = 'hg8'    
_C.params.STRUCTURE_POSE_NET = 'normflow'     # 'default'   # 'vae' 
_C.params.NF_VERSION = 3
_C.params.N_JOINTS = 35   
_C.params.N_KEYP = 24      #20    
_C.params.N_SEG = 2
_C.params.N_PARTSEG = 15
_C.params.UPSAMPLE_SEG = True
_C.params.ADD_PARTSEG = True   # partseg: for the CVPR paper this part of the network exists, but is not trained (no part labels in StanExt)
_C.params.N_BETAS = 30    # 10
_C.params.N_BETAS_LIMBS = 7
_C.params.N_BONES = 24
_C.params.N_BREEDS = 121      # 120 breeds plus background
_C.params.IMG_SIZE = 256
_C.params.SILH_NO_TAIL = False
_C.params.KP_THRESHOLD = None    
_C.params.ADD_Z_TO_3D_INPUT = False   
_C.params.N_SEGBPS = 64*2
_C.params.ADD_SEGBPS_TO_3D_INPUT = True
_C.params.FIX_FLENGTH = False   
_C.params.RENDER_ALL = True
_C.params.VLIN = 2    
_C.params.STRUCTURE_Z_TO_B = 'lin'
_C.params.N_Z_FREE = 64
_C.params.PCK_THRESH = 0.15    

## optimization settings
_C.optim = CN()
_C.optim.LR = 5e-4
_C.optim.SCHEDULE = [150, 175, 200]
_C.optim.GAMMA = 0.1
_C.optim.MOMENTUM = 0
_C.optim.WEIGHT_DECAY = 0
_C.optim.EPOCHS = 220
_C.optim.BATCH_SIZE = 12       # keep 12 (needs to be an even number, as we have a custom data sampler)
_C.optim.TRAIN_PARTS = 'all_without_shapedirs'

## dataset settings
_C.data = CN()
_C.data.DATASET = 'stanext24'
_C.data.V12 = True     
_C.data.SHORTEN_VAL_DATASET_TO = None        
_C.data.VAL_OPT = 'val'
_C.data.VAL_METRICS = 'no_loss'

# ---------------------------------------
def update_dependent_vars(cfg):    
    cfg.params.N_CLASSES = cfg.params.N_KEYP + cfg.params.N_SEG
    if cfg.params.VLIN == 0: 
        cfg.params.NUM_STAGE_COMB = 2
        cfg.params.NUM_STAGE_HEADS = 1  
        cfg.params.NUM_STAGE_HEADS_POSE = 1
        cfg.params.TRANS_SEP = False
    elif cfg.params.VLIN == 1:
        cfg.params.NUM_STAGE_COMB = 3              
        cfg.params.NUM_STAGE_HEADS = 1             
        cfg.params.NUM_STAGE_HEADS_POSE = 2        
        cfg.params.TRANS_SEP = False
    elif cfg.params.VLIN == 2:
        cfg.params.NUM_STAGE_COMB = 3              
        cfg.params.NUM_STAGE_HEADS = 1             
        cfg.params.NUM_STAGE_HEADS_POSE = 2        
        cfg.params.TRANS_SEP = True
    else:
        raise NotImplementedError
    if cfg.params.STRUCTURE_Z_TO_B == '1dconv':
        cfg.params.N_Z = cfg.params.N_BETAS + cfg.params.N_BETAS_LIMBS
    else:
        cfg.params.N_Z = cfg.params.N_Z_FREE
    return


update_dependent_vars(_C)
global _cfg_global 
_cfg_global = _C.clone()


def get_cfg_defaults():
    # Get a yacs CfgNode object with default values as defined within this file.
    # Return a clone so that the defaults will not be altered.
    return _C.clone()

def update_cfg_global_with_yaml(cfg_yaml_file):    
    _cfg_global.merge_from_file(cfg_yaml_file)
    update_dependent_vars(_cfg_global)
    return 

def get_cfg_global_updated():
    # return _cfg_global.clone()
    return _cfg_global

