from __future__ import print_function, absolute_import, division
import os
import onnx
import onnxsim
import torch.optim
from collections import OrderedDict

import argparse
import os.path
import os.path
from distutils.util import strtobool
import torch
import torch.backends.cudnn
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../', 'src'))
from stacked_hourglass.datasets.stanext24 import StanExt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../', 'src'))
from combined_model.model_shape_v7 import ModelImageTo3d_withshape_withproj
from configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated

def main(args):
    args.workers = 16
    args.model_file_complete = 'barc_complete/model_best.pth.tar'
    args.config = 'barc_cfg_test.yaml'
    args.save_images = True

    # load configs
    #   step 1: load default configs
    #   step 2: load updates from .yaml file
    path_config = os.path.join(get_cfg_defaults().barc_dir, 'src', 'configs', args.config)
    update_cfg_global_with_yaml(path_config)
    cfg = get_cfg_global_updated()

    # Select the hardware device to use for training.
    if not torch.cuda.is_available() and not cfg.device == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    path_model_file_complete = os.path.join(cfg.paths.ROOT_CHECKPOINT_PATH, args.model_file_complete)

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    # prepare complete model
    model = ModelImageTo3d_withshape_withproj(
        num_stage_comb=cfg.params.NUM_STAGE_COMB, num_stage_heads=cfg.params.NUM_STAGE_HEADS, \
        num_stage_heads_pose=cfg.params.NUM_STAGE_HEADS_POSE, trans_sep=cfg.params.TRANS_SEP, \
        arch=cfg.params.ARCH, n_joints=cfg.params.N_JOINTS, n_classes=cfg.params.N_CLASSES, \
        n_keyp=cfg.params.N_KEYP, n_bones=cfg.params.N_BONES, n_betas=cfg.params.N_BETAS,
        n_betas_limbs=cfg.params.N_BETAS_LIMBS, \
        n_breeds=cfg.params.N_BREEDS, n_z=cfg.params.N_Z, image_size=cfg.params.IMG_SIZE, \
        silh_no_tail=cfg.params.SILH_NO_TAIL, thr_keyp_sc=cfg.params.KP_THRESHOLD,
        add_z_to_3d_input=cfg.params.ADD_Z_TO_3D_INPUT,
        n_segbps=cfg.params.N_SEGBPS, add_segbps_to_3d_input=cfg.params.ADD_SEGBPS_TO_3D_INPUT,
        add_partseg=cfg.params.ADD_PARTSEG, n_partseg=cfg.params.N_PARTSEG, \
        fix_flength=cfg.params.FIX_FLENGTH, structure_z_to_betas=cfg.params.STRUCTURE_Z_TO_B,
        structure_pose_net=cfg.params.STRUCTURE_POSE_NET,
        nf_version=cfg.params.NF_VERSION)

    # load trained model
    print(path_model_file_complete)
    assert os.path.isfile(path_model_file_complete)
    print('Loading model weights from file: {}'.format(path_model_file_complete))

    checkpoint_complete = torch.load(path_model_file_complete)
    state_dict_complete = checkpoint_complete['state_dict']
    model.load_state_dict(state_dict_complete, strict=False)
    model = model.to(device)
    model.eval()

    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, 256, 256),
    )
    data_info = StanExt.DATA_INFO
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}

    output_path = './barc.onnx'
    torch.onnx.export(
        model,
        args=(dummy_input, norm_dict),
        f=output_path,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )
    logger.log("finished exporting onnx ")

    logger.log("start simplifying onnx ")
    input_data = {"data": dummy_input.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(output_path, input_data=input_data)
    if flag:
        onnx.save(model_sim, output_path)
        logger.log("simplify onnx successfully")
    else:
        logger.log("simplify onnx failed")


if __name__ == '__main__':

    # use as follows:
    # python scripts/test_image_to_3d_withshape.py --workers 12 --save-images True --config barc_cfg_test.yaml --model-file-complete=barc_new_v2/model_best.pth.tar

    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--model-file-complete', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('-cg', '--config', default='barc_cfg_test.yaml', type=str, metavar='PATH',
                        help='name of config file (default: barc_cfg_test.yaml within src/configs folder)')
    parser.add_argument('--save-images', default='True', type=lambda x: bool(strtobool(x)),
                        help='bool indicating if images should be saved')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--metrics', '-m', metavar='METRICS', default='all',
                        choices=['all', None],
                        help='model architecture')
    main(parser.parse_args())