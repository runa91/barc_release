'''
Adjusted version of other PyTorch implementation of the SMAL/SMPL model
see:
    1.) https://github.com/silviazuffi/smalst/blob/master/smal_model/smal_torch.py
    2.) https://github.com/benjiebob/SMALify/blob/master/smal_model/smal_torch.py
'''

import os
import pickle as pkl
import json
import numpy as np
from smpl_webuser.serialization import load_model
import pickle as pkl

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.SMAL_configs import SMAL_DATA_DIR, SYMMETRY_INDS_FILE

# model_dir = 'smalst/smpl_models/'
# FILE_DIR = os.path.dirname(os.path.realpath(__file__))
model_dir = SMAL_DATA_DIR       # os.path.join(FILE_DIR, '..', 'smpl_models/')
symmetry_inds_file = SYMMETRY_INDS_FILE     # os.path.join(FILE_DIR, '..', 'smpl_models/symmetry_inds.json')
with open(symmetry_inds_file) as f:
    symmetry_inds_dict = json.load(f)
LEFT_INDS = np.asarray(symmetry_inds_dict['left_inds'])
RIGHT_INDS = np.asarray(symmetry_inds_dict['right_inds'])
CENTER_INDS = np.asarray(symmetry_inds_dict['center_inds'])


def get_symmetry_indices():
    sym_dict = {'left': LEFT_INDS,
                'right': RIGHT_INDS,
                'center': CENTER_INDS}
    return sym_dict

def verify_symmetry(shapedirs, center_inds=CENTER_INDS, left_inds=LEFT_INDS, right_inds=RIGHT_INDS):
    # shapedirs: (3889, 3, n_sh)
    assert (shapedirs[center_inds, 1, :] == 0.0).all()
    assert (shapedirs[right_inds, 1, :] == -shapedirs[left_inds, 1, :]).all()
    return

def from_shapedirs_to_shapedirs_half(shapedirs, center_inds=CENTER_INDS, left_inds=LEFT_INDS, right_inds=RIGHT_INDS, verify=False):
    # shapedirs: (3889, 3, n_sh)
    # shapedirs_half: (2012, 3, n_sh)
    selected_inds = np.concatenate((center_inds, left_inds), axis=0)
    shapedirs_half = shapedirs[selected_inds, :, :]
    if verify: 
        verify_symmetry(shapedirs)
    else:
        shapedirs_half[:center_inds.shape[0], 1, :] = 0.0
    return shapedirs_half

def from_shapedirs_half_to_shapedirs(shapedirs_half, center_inds=CENTER_INDS, left_inds=LEFT_INDS, right_inds=RIGHT_INDS):
    # shapedirs_half: (2012, 3, n_sh)
    # shapedirs: (3889, 3, n_sh)
    shapedirs = np.zeros((center_inds.shape[0] + 2*left_inds.shape[0], 3, shapedirs_half.shape[2]))
    shapedirs[center_inds, :, :] = shapedirs_half[:center_inds.shape[0], :, :]
    shapedirs[left_inds, :, :] = shapedirs_half[center_inds.shape[0]:, :, :]
    shapedirs[right_inds, :, :] = shapedirs_half[center_inds.shape[0]:, :, :]
    shapedirs[right_inds, 1, :] = - shapedirs_half[center_inds.shape[0]:, 1, :]
    return shapedirs

def align_smal_template_to_symmetry_axis(v, subtract_mean=True):
    # These are the indexes of the points that are on the symmetry axis
    I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 55, 119, 120, 163, 209, 210, 211, 213, 216, 227, 326, 395, 452, 578, 910, 959, 964, 975, 976, 977, 1172, 1175, 1176, 1178, 1194, 1243, 1739, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1870, 1919, 1960, 1961, 1965, 1967, 2003]
    if subtract_mean:
        v = v - np.mean(v)
    y = np.mean(v[I,1])
    v[:,1] = v[:,1] - y
    v[I,1] = 0
    left_inds = LEFT_INDS
    right_inds = RIGHT_INDS
    center_inds = CENTER_INDS
    v[right_inds, :] = np.array([1,-1,1])*v[left_inds, :]

    try:
        assert(len(left_inds) == len(right_inds))
    except:
        import pdb; pdb.set_trace()

    return v, left_inds, right_inds, center_inds

def load_smal_model(model_name='my_smpl_00781_4_all.pkl'):
    model_path = os.path.join(model_dir, model_name)

    model = load_model(model_path)
    v = align_smal_template_to_symmetry_axis(model.r.copy())
    return v, model.f

def get_horse_template(model_name='my_smpl_00781_4_all.pkl', data_name='my_smpl_data_00781_4_all.pkl'):
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)
    nBetas = len(model.betas.r)
    data_path = os.path.join(model_dir, data_name)     # os.path.join(model_dir, 'my_smpl_data_00781_4_all.pkl')
    # data = pkl.load(open(data_path))
    try:
        with open(data_path, 'r') as f:
            data = pkl.load(f)
    except (UnicodeDecodeError, TypeError) as e:
        with open(data_path, 'rb') as file:
            u = pkl._Unpickler(file)
            u.encoding = 'latin1'
            data = u.load()
    # Select average zebra/horse
    betas = data['cluster_means'][2][:nBetas]
    model.betas[:] = betas
    v = model.r.copy()
    return v

def get_dog_template(model_name='my_smpl_00781_4_all.pkl', data_name='my_smpl_data_00781_4_all.pkl'):
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)
    nBetas = len(model.betas.r)
    data_path = os.path.join(model_dir, data_name)     # os.path.join(model_dir, 'my_smpl_data_00781_4_all.pkl')
    # data = pkl.load(open(data_path))
    try:
        with open(data_path, 'r') as f:
            data = pkl.load(f)
    except (UnicodeDecodeError, TypeError) as e:
        with open(data_path, 'rb') as file:
            u = pkl._Unpickler(file)
            u.encoding = 'latin1'
            data = u.load()
    # Select average dog
    betas = data['cluster_means'][1][:nBetas]
    model.betas[:] = betas
    v = model.r.copy()
    return v

def get_neutral_template(model_name='my_smpl_00781_4_all.pkl', data_name='my_smpl_data_00781_4_all.pkl'):
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)
    nBetas = len(model.betas.r)
    data_path = os.path.join(model_dir, data_name)     # os.path.join(model_dir, 'my_smpl_data_00781_4_all.pkl')
    # data = pkl.load(open(data_path))
    try:
        with open(data_path, 'r') as f:
            data = pkl.load(f)
    except (UnicodeDecodeError, TypeError) as e:
        with open(data_path, 'rb') as file:
            u = pkl._Unpickler(file)
            u.encoding = 'latin1'
            data = u.load()
    v = model.r.copy()
    return v


