from dataclasses import dataclass
from typing import List
import json
import numpy as np
import os
import sys

STATISTICS_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'statistics')
STATISTICS_PATH = os.path.join(STATISTICS_DATA_DIR, 'statistics_modified_v1.json')

@dataclass
class DataInfo:
    rgb_mean: List[float]
    rgb_stddev: List[float]
    joint_names: List[str]
    hflip_indices: List[int]
    n_joints: int
    n_keyp: int
    n_bones: int
    n_betas: int
    image_size: int
    trans_mean: np.ndarray
    trans_std: np.ndarray
    flength_mean: np.ndarray
    flength_std: np.ndarray
    pose_rot6d_mean: np.ndarray
    keypoint_weights: List[float]

# SMAL samples 3d statistics
#   statistics like mean values were calculated once when the project was started and they were not changed afterwards anymore
def load_statistics(statistics_path):
    with open(statistics_path) as f:
        statistics = json.load(f)
    '''new_pose_mean = [[[np.round(val, 2) for val in sublst] for sublst in sublst_big] for sublst_big in statistics['pose_mean']]
    statistics['pose_mean'] = new_pose_mean
    j_out = json.dumps(statistics, indent=4)    #, sort_keys=True)
    with open(self.statistics_path, 'w') as file: file.write(j_out)'''
    new_statistics = {'trans_mean': np.asarray(statistics['trans_mean']),
                    'trans_std': np.asarray(statistics['trans_std']),       
                    'flength_mean': np.asarray(statistics['flength_mean']),
                    'flength_std': np.asarray(statistics['flength_std']),  
                    'pose_mean': np.asarray(statistics['pose_mean']),
                    }
    new_statistics['pose_rot6d_mean'] = new_statistics['pose_mean'][:, :, :2].reshape((-1, 6))
    return new_statistics
STATISTICS = load_statistics(STATISTICS_PATH)


############################################################################
# for StanExt (original number of keypoints, 20 not 24)

# for keypoint names see: https://github.com/benjiebob/StanfordExtra/blob/master/keypoint_definitions.csv
StanExt_JOINT_NAMES = [
    'Left_front_leg_paw', 'Left_front_leg_middle_joint', 'Left_front_leg_top', 
    'Left_rear_leg_paw', 'Left_rear_leg_middle_joint', 'Left_rear_leg_top',
    'Right_front_leg_paw', 'Right_front_leg_middle_joint', 'Right_front_leg_top',
    'Right_rear_leg_paw', 'Right_rear_leg_middle_joint', 'Right_rear_leg_top',
    'Tail_start', 'Tail_end', 'Base_of_left_ear', 'Base_of_right_ear', 
    'Nose', 'Chin', 'Left_ear_tip', 'Right_ear_tip']

KEYPOINT_WEIGHTS = [3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 2, 2, 3, 1, 2, 2]      

COMPLETE_DATA_INFO = DataInfo(
    rgb_mean=[0.4404, 0.4440, 0.4327],      # not sure
    rgb_stddev=[0.2458, 0.2410, 0.2468],    # not sure
    joint_names=StanExt_JOINT_NAMES,
    hflip_indices=[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 15, 14, 16, 17, 19, 18],
    n_joints = 35,
    n_keyp = 20,    # 25,
    n_bones = 24,
    n_betas = 30,       # 10,
    image_size = 256,
    trans_mean = STATISTICS['trans_mean'],
    trans_std = STATISTICS['trans_std'],
    flength_mean = STATISTICS['flength_mean'],
    flength_std = STATISTICS['flength_std'],
    pose_rot6d_mean = STATISTICS['pose_rot6d_mean'],
    keypoint_weights = KEYPOINT_WEIGHTS
    )


############################################################################
# new for StanExt24

# ..., 'Left_eye', 'Right_eye', 'Withers', 'Throat']       # the last 4 keypoints are in the animal_pose dataset, but not StanfordExtra
StanExt_JOINT_NAMES_24 = [
    'Left_front_leg_paw', 'Left_front_leg_middle_joint', 'Left_front_leg_top', 
    'Left_rear_leg_paw', 'Left_rear_leg_middle_joint', 'Left_rear_leg_top',
    'Right_front_leg_paw', 'Right_front_leg_middle_joint', 'Right_front_leg_top',
    'Right_rear_leg_paw', 'Right_rear_leg_middle_joint', 'Right_rear_leg_top',
    'Tail_start', 'Tail_end', 'Base_of_left_ear', 'Base_of_right_ear', 
    'Nose', 'Chin', 'Left_ear_tip', 'Right_ear_tip',
    'Left_eye', 'Right_eye', 'Withers', 'Throat']

KEYPOINT_WEIGHTS_24 = [3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 2, 2, 3, 1, 2, 2, 1, 1, 0, 0]     

COMPLETE_DATA_INFO_24 = DataInfo(
    rgb_mean=[0.4404, 0.4440, 0.4327],      # not sure
    rgb_stddev=[0.2458, 0.2410, 0.2468],    # not sure
    joint_names=StanExt_JOINT_NAMES_24,
    hflip_indices=[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 15, 14, 16, 17, 19, 18, 21, 20, 22, 23],
    n_joints = 35,
    n_keyp = 24,    # 20,    # 25,
    n_bones = 24,
    n_betas = 30,       # 10,
    image_size = 256,
    trans_mean = STATISTICS['trans_mean'],
    trans_std = STATISTICS['trans_std'],
    flength_mean = STATISTICS['flength_mean'],
    flength_std = STATISTICS['flength_std'],
    pose_rot6d_mean = STATISTICS['pose_rot6d_mean'],
    keypoint_weights = KEYPOINT_WEIGHTS_24
    )


