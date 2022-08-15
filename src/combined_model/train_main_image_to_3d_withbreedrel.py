
import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.parallel
from tqdm import tqdm
import os
import pathlib
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import trimesh

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds, get_preds, get_preds_soft
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints, save_input_image
from metrics.metrics import Metrics
from configs.SMAL_configs import EVAL_KEYPOINTS, KEYPOINT_GROUPS


# ---------------------------------------------------------------------------------------------------------------------------
def do_training_epoch(train_loader, model, loss_module, device, data_info, optimiser, quiet=False, acc_joints=None, weight_dict=None):
    losses = AverageMeter()
    losses_keyp = AverageMeter()
    losses_silh = AverageMeter()
    losses_shape = AverageMeter()
    losses_pose = AverageMeter()
    losses_class = AverageMeter()
    losses_breed = AverageMeter()
    losses_partseg = AverageMeter()
    accuracies = AverageMeter()
    # Put the model in training mode.
    model.train()
    # prepare progress bar
    iterable = enumerate(train_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Train', total=len(train_loader), ascii=True, leave=False)
        iterable = progress
    # information for normalization 
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}
    # prepare variables, put them on the right device
    for i, (input, target_dict) in iterable:
        batch_size = input.shape[0]
        for key in target_dict.keys(): 
            if key == 'breed_index':
                target_dict[key] = target_dict[key].long().to(device)
            elif key in ['index', 'pts', 'tpts', 'target_weight', 'silh', 'silh_distmat_tofg', 'silh_distmat_tobg', 'sim_breed_index', 'img_border_mask']:
                target_dict[key] = target_dict[key].float().to(device)
            elif key == 'has_seg':
                target_dict[key] = target_dict[key].to(device)
            else:
                pass
        input = input.float().to(device)

        # ----------------------- do training step -----------------------
        assert model.training, 'model must be in training mode.'
        with torch.enable_grad():
            # ----- forward pass -----  
            output, output_unnorm, output_reproj = model(input, norm_dict=norm_dict)        
            # ----- loss -----
            loss, loss_dict = loss_module(output_reproj=output_reproj, 
                target_dict=target_dict, 
                weight_dict=weight_dict)
            # ----- backward pass and parameter update -----
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        # ----------------------------------------------------------------

        # prepare losses for progress bar
        bs_fake = 1     # batch_size
        losses.update(loss_dict['loss'], bs_fake)
        losses_keyp.update(loss_dict['loss_keyp_weighted'], bs_fake)
        losses_silh.update(loss_dict['loss_silh_weighted'], bs_fake)
        losses_shape.update(loss_dict['loss_shape_weighted'], bs_fake)
        losses_pose.update(loss_dict['loss_poseprior_weighted'], bs_fake)   
        losses_class.update(loss_dict['loss_class_weighted'], bs_fake)
        losses_breed.update(loss_dict['loss_breed_weighted'], bs_fake)
        losses_partseg.update(loss_dict['loss_partseg_weighted'], bs_fake)
        acc = - loss_dict['loss_keyp_weighted']     # this will be used to keep track of the 'best model'
        accuracies.update(acc, bs_fake)
        # Show losses as part of the progress bar.
        if progress is not None:
            my_string = 'Loss: {loss:0.4f}, loss_keyp: {loss_keyp:0.4f}, loss_silh: {loss_silh:0.4f}, loss_partseg: {loss_partseg:0.4f}, loss_shape: {loss_shape:0.4f}, loss_pose: {loss_pose:0.4f}, loss_class: {loss_class:0.4f}, loss_breed: {loss_breed:0.4f}'.format(
                loss=losses.avg,
                loss_keyp=losses_keyp.avg,
                loss_silh=losses_silh.avg,
                loss_shape=losses_shape.avg,
                loss_pose=losses_pose.avg,
                loss_class=losses_class.avg,
                loss_breed=losses_breed.avg,
                loss_partseg=losses_partseg.avg
            )
            progress.set_postfix_str(my_string)

    return my_string, accuracies.avg       


# ---------------------------------------------------------------------------------------------------------------------------
def do_validation_epoch(val_loader, model, loss_module, device, data_info, flip=False, quiet=False, acc_joints=None, save_imgs_path=None, weight_dict=None, metrics=None, val_opt='default', test_name_list=None, render_all=False, pck_thresh=0.15, len_dataset=None):
    losses = AverageMeter()
    losses_keyp = AverageMeter()
    losses_silh = AverageMeter()
    losses_shape = AverageMeter()
    losses_pose = AverageMeter()
    losses_class = AverageMeter()
    losses_breed = AverageMeter()
    losses_partseg = AverageMeter()
    accuracies = AverageMeter()
    if save_imgs_path is not None:
        pathlib.Path(save_imgs_path).mkdir(parents=True, exist_ok=True) 
    # Put the model in evaluation mode.
    model.eval()
    # prepare progress bar
    iterable = enumerate(val_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Valid', total=len(val_loader), ascii=True, leave=False)
        iterable = progress
    # summarize information for normalization 
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}
    batch_size = val_loader.batch_size
    # prepare variables, put them on the right device
    my_step = 0
    for i, (input, target_dict) in iterable:
        curr_batch_size = input.shape[0]
        for key in target_dict.keys(): 
            if key == 'breed_index':
                target_dict[key] = target_dict[key].long().to(device)
            elif key in ['index', 'pts', 'tpts', 'target_weight', 'silh', 'silh_distmat_tofg', 'silh_distmat_tobg', 'sim_breed_index', 'img_border_mask']:
                target_dict[key] = target_dict[key].float().to(device)
            elif key == 'has_seg':
                target_dict[key] = target_dict[key].to(device)
            else:
                pass
        input = input.float().to(device)

        # ----------------------- do validation step -----------------------
        with torch.no_grad():
            # ----- forward pass -----  
            # output: (['pose', 'flength', 'trans', 'keypoints_norm', 'keypoints_scores'])
            # output_unnorm: (['pose_rotmat', 'flength', 'trans', 'keypoints'])
            # output_reproj: (['vertices_smal', 'torch_meshes', 'keyp_3d', 'keyp_2d', 'silh', 'betas', 'pose_rot6d', 'dog_breed', 'shapedirs', 'z', 'flength_unnorm', 'flength'])
            # target_dict: (['index', 'center', 'scale', 'pts', 'tpts', 'target_weight', 'breed_index', 'sim_breed_index', 'ind_dataset', 'silh'])
            output, output_unnorm, output_reproj = model(input, norm_dict=norm_dict)        
            # ----- loss -----
            if metrics == 'no_loss':
                loss, loss_dict = loss_module(output_reproj=output_reproj, 
                    target_dict=target_dict, 
                    weight_dict=weight_dict)
        # ----------------------------------------------------------------

        if i == 0:
            if len_dataset is None:
                len_data = val_loader.batch_size * len(val_loader)  # 1703
            else:
                len_data = len_dataset
            if metrics == 'all' or metrics == 'no_loss':
                pck = np.zeros((len_data))
                pck_by_part = {group:np.zeros((len_data)) for group in KEYPOINT_GROUPS}
                acc_sil_2d = np.zeros(len_data)

                all_betas = np.zeros((len_data, output_reproj['betas'].shape[1]))
                all_betas_limbs = np.zeros((len_data, output_reproj['betas_limbs'].shape[1]))
                all_z = np.zeros((len_data, output_reproj['z'].shape[1]))
                all_pose_rotmat = np.zeros((len_data, output_unnorm['pose_rotmat'].shape[1], 3, 3))
                all_flength = np.zeros((len_data, output_unnorm['flength'].shape[1]))
                all_trans = np.zeros((len_data, output_unnorm['trans'].shape[1]))
                all_breed_indices = np.zeros((len_data))
                all_image_names = []        # len_data * [None]

        index = i  
        ind_img = 0
        if save_imgs_path is not None:
            # render predicted 3d models
            visualizations = model.render_vis_nograd(vertices=output_reproj['vertices_smal'],
                                                    focal_lengths=output_unnorm['flength'],
                                                    color=0)        # color=2)
            for ind_img in range(len(target_dict['index'])):    
                try: 
                    if test_name_list is not None:
                        img_name = test_name_list[int(target_dict['index'][ind_img].cpu().detach().numpy())].replace('/', '_')
                        img_name = img_name.split('.')[0]
                    else:
                        img_name = str(index) + '_' + str(ind_img)
                    # save image with predicted keypoints
                    out_path = save_imgs_path + '/keypoints_pred_' + img_name + '.png'
                    pred_unp = (output['keypoints_norm'][ind_img, :, :] + 1.) / 2 * (data_info.image_size - 1)
                    pred_unp_maxval = output['keypoints_scores'][ind_img, :, :]
                    pred_unp_prep = torch.cat((pred_unp, pred_unp_maxval), 1)
                    inp_img = input[ind_img, :, :, :].detach().clone()
                    save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_path, threshold=0.1, print_scores=True, ratio_in_out=1.0)    # threshold=0.3
                    # save predicted 3d model (front view)
                    pred_tex = visualizations[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                    pred_tex_max = np.max(pred_tex, axis=2)
                    out_path = save_imgs_path + '/tex_pred_' + img_name + '.png'
                    plt.imsave(out_path, pred_tex)
                    input_image = input[ind_img, :, :, :].detach().clone()
                    for t, m, s in zip(input_image, data_info.rgb_mean, data_info.rgb_stddev): t.add_(m)
                    input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0) 
                    im_masked = cv2.addWeighted(input_image_np,0.2,pred_tex,0.8,0)
                    im_masked[pred_tex_max<0.01, :] = input_image_np[pred_tex_max<0.01, :]
                    out_path = save_imgs_path + '/comp_pred_' + img_name + '.png'
                    plt.imsave(out_path, im_masked)
                    # save predicted 3d model (side view)
                    vertices_cent = output_reproj['vertices_smal'] - output_reproj['vertices_smal'].mean(dim=1)[:, None, :]
                    roll = np.pi / 2 * torch.ones(1).float().to(device)
                    pitch = np.pi / 2 * torch.ones(1).float().to(device)
                    tensor_0 = torch.zeros(1).float().to(device)
                    tensor_1 = torch.ones(1).float().to(device)
                    RX = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0]), torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)
                    RY = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)
                    vertices_rot = (torch.matmul(RY, vertices_cent.reshape((-1, 3))[:, :, None])).reshape((curr_batch_size, -1, 3))
                    vertices_rot[:, :, 2] = vertices_rot[:, :, 2] + torch.ones_like(vertices_rot[:, :, 2]) * 20     # 18     # *16

                    visualizations_rot = model.render_vis_nograd(vertices=vertices_rot,
                                                            focal_lengths=output_unnorm['flength'],
                                                            color=0)        # 2)
                    pred_tex = visualizations_rot[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                    pred_tex_max = np.max(pred_tex, axis=2)
                    out_path = save_imgs_path + '/rot_tex_pred_' + img_name + '.png'
                    plt.imsave(out_path, pred_tex)
                    if render_all:
                        # save input image
                        inp_img = input[ind_img, :, :, :].detach().clone()
                        out_path = save_imgs_path + '/image_' + img_name + '.png'
                        save_input_image(inp_img, out_path)
                        # save mesh
                        V_posed = output_reproj['vertices_smal'][ind_img, :, :].detach().cpu().numpy()
                        Faces = model.smal.f
                        mesh_posed = trimesh.Trimesh(vertices=V_posed, faces=Faces, process=False)
                        mesh_posed.export(save_imgs_path + '/mesh_posed_' + img_name + '.obj')
                except: 
                    print('dont save an image')

        if metrics == 'all' or metrics == 'no_loss':
            # prepare a dictionary with all the predicted results
            preds = {}
            preds['betas'] = output_reproj['betas'].cpu().detach().numpy()
            preds['betas_limbs'] = output_reproj['betas_limbs'].cpu().detach().numpy()
            preds['z'] = output_reproj['z'].cpu().detach().numpy()
            preds['pose_rotmat'] = output_unnorm['pose_rotmat'].cpu().detach().numpy()
            preds['flength'] = output_unnorm['flength'].cpu().detach().numpy()
            preds['trans'] = output_unnorm['trans'].cpu().detach().numpy()
            preds['breed_index'] = target_dict['breed_index'].cpu().detach().numpy().reshape((-1))
            img_names = []
            for ind_img2 in range(0, output_reproj['betas'].shape[0]):
                if test_name_list is not None:
                    img_name2 = test_name_list[int(target_dict['index'][ind_img2].cpu().detach().numpy())].replace('/', '_')
                    img_name2 = img_name2.split('.')[0]
                else:
                    img_name2 = str(index) + '_' + str(ind_img2)
                img_names.append(img_name2)
            preds['image_names'] = img_names
            # prepare keypoints for PCK calculation - predicted as well as ground truth
            pred_keypoints_norm = output['keypoints_norm']   # -1 to 1
            pred_keypoints_256 = output_reproj['keyp_2d']
            pred_keypoints = pred_keypoints_256
            gt_keypoints_256 = target_dict['tpts'][:, :, :2] / 64. * (256. - 1)
            gt_keypoints_norm = gt_keypoints_256 / 256 / 0.5 - 1
            gt_keypoints = torch.cat((gt_keypoints_256, target_dict['tpts'][:, :, 2:3]), dim=2)     # gt_keypoints_norm
            # prepare silhouette for IoU calculation - predicted as well as ground truth
            has_seg = target_dict['has_seg']
            img_border_mask = target_dict['img_border_mask'][:, 0, :, :]
            gtseg = target_dict['silh']
            synth_silhouettes = output_reproj['silh'][:, 0, :, :]       # output_reproj['silh']
            synth_silhouettes[synth_silhouettes>0.5] = 1
            synth_silhouettes[synth_silhouettes<0.5] = 0
            # calculate PCK as well as IoU (similar to WLDO)
            preds['acc_PCK'] = Metrics.PCK(
                pred_keypoints, gt_keypoints, 
                gtseg, has_seg, idxs=EVAL_KEYPOINTS,
                thresh_range=[pck_thresh],       # [0.15],
            )
            preds['acc_IOU'] = Metrics.IOU(
                synth_silhouettes, gtseg, 
                img_border_mask, mask=has_seg
            )
            for group, group_kps in KEYPOINT_GROUPS.items():
                preds[f'{group}_PCK'] = Metrics.PCK(
                    pred_keypoints, gt_keypoints, gtseg, has_seg, 
                    thresh_range=[pck_thresh],       # [0.15],
                    idxs=group_kps
                )
            # add results for all images in this batch to lists
            curr_batch_size = pred_keypoints_256.shape[0]
            if not (preds['acc_PCK'].data.cpu().numpy().shape == (pck[my_step * batch_size:my_step * batch_size + curr_batch_size]).shape):
                import pdb; pdb.set_trace()
            pck[my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['acc_PCK'].data.cpu().numpy()
            acc_sil_2d[my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['acc_IOU'].data.cpu().numpy()
            for part in pck_by_part:
                pck_by_part[part][my_step * batch_size:my_step * batch_size + curr_batch_size] = preds[f'{part}_PCK'].data.cpu().numpy()
            all_betas[my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['betas'] 
            all_betas_limbs[my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['betas_limbs'] 
            all_z[my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['z'] 
            all_pose_rotmat[my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['pose_rotmat'] 
            all_flength[my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['flength'] 
            all_trans[my_step * batch_size:my_step * batch_size + curr_batch_size, ...] = preds['trans'] 
            all_breed_indices[my_step * batch_size:my_step * batch_size + curr_batch_size] = preds['breed_index']
            all_image_names.extend(preds['image_names'])
            # update progress bar
            if progress is not None:
                my_string = "PCK: {0:.2f}, IOU: {1:.2f}".format(
                    pck[:(my_step * batch_size + curr_batch_size)].mean(),
                    acc_sil_2d[:(my_step * batch_size + curr_batch_size)].mean())
                progress.set_postfix_str(my_string)
        else:
            # measure accuracy and record loss
            bs_fake = 1     # batch_size
            losses.update(loss_dict['loss'], bs_fake)
            losses_keyp.update(loss_dict['loss_keyp_weighted'], bs_fake)
            losses_silh.update(loss_dict['loss_silh_weighted'], bs_fake)
            losses_shape.update(loss_dict['loss_shape_weighted'], bs_fake)
            losses_pose.update(loss_dict['loss_poseprior_weighted'], bs_fake)  
            losses_class.update(loss_dict['loss_class_weighted'], bs_fake)
            losses_breed.update(loss_dict['loss_breed_weighted'], bs_fake)
            losses_partseg.update(loss_dict['loss_partseg_weighted'], bs_fake)
            acc = - loss_dict['loss_keyp_weighted']     # this will be used to keep track of the 'best model'
            accuracies.update(acc, bs_fake)
            # Show losses as part of the progress bar.
            if progress is not None:
                my_string = 'Loss: {loss:0.4f}, loss_keyp: {loss_keyp:0.4f}, loss_silh: {loss_silh:0.4f}, loss_partseg: {loss_partseg:0.4f}, loss_shape: {loss_shape:0.4f}, loss_pose: {loss_pose:0.4f}, loss_class: {loss_class:0.4f}, loss_breed: {loss_breed:0.4f}'.format(
                    loss=losses.avg,
                    loss_keyp=losses_keyp.avg,
                    loss_silh=losses_silh.avg,
                    loss_shape=losses_shape.avg,
                    loss_pose=losses_pose.avg,
                    loss_class=losses_class.avg,
                    loss_breed=losses_breed.avg,
                    loss_partseg=losses_partseg.avg
                )
                progress.set_postfix_str(my_string)
        my_step += 1
    if metrics == 'all':
        summary = {'pck': pck, 'acc_sil_2d': acc_sil_2d, 'pck_by_part':pck_by_part,
                    'betas': all_betas, 'betas_limbs': all_betas_limbs, 'z': all_z, 'pose_rotmat': all_pose_rotmat,
                    'flenght': all_flength, 'trans': all_trans, 'image_names': all_image_names, 'breed_indices': all_breed_indices}
        return my_string, summary    
    elif metrics == 'no_loss':
        return my_string, np.average(np.asarray(acc_sil_2d))
    else:
        return my_string, accuracies.avg       


# ---------------------------------------------------------------------------------------------------------------------------
def do_visual_epoch(val_loader, model, device, data_info, flip=False, quiet=False, acc_joints=None, save_imgs_path=None, weight_dict=None, metrics=None, val_opt='default', test_name_list=None, render_all=False, pck_thresh=0.15):
    if save_imgs_path is not None:
        pathlib.Path(save_imgs_path).mkdir(parents=True, exist_ok=True) 

    # Put the model in evaluation mode.
    model.eval()

    iterable = enumerate(val_loader)

    # information for normalization 
    norm_dict = {
        'pose_rot6d_mean': torch.from_numpy(data_info.pose_rot6d_mean).float().to(device),
        'trans_mean': torch.from_numpy(data_info.trans_mean).float().to(device),
        'trans_std': torch.from_numpy(data_info.trans_std).float().to(device),
        'flength_mean': torch.from_numpy(data_info.flength_mean).float().to(device),
        'flength_std': torch.from_numpy(data_info.flength_std).float().to(device)}

    for i, (input, target_dict) in iterable:
        batch_size = input.shape[0]
        input = input.float().to(device)

        # ----------------------- do visualization step -----------------------
        with torch.no_grad():
            output, output_unnorm, output_reproj = model(input, norm_dict=norm_dict)        

        index = i  
        ind_img = 0
        if save_imgs_path is not None:
            for ind_img in range(batch_size): #  range(min(12, batch_size)):     # range(12):    # [0]:  #range(0, batch_size):

                try: 
                    if test_name_list is not None:
                        img_name = test_name_list[int(target_dict['index'][ind_img].cpu().detach().numpy())].replace('/', '_')
                        img_name = img_name.split('.')[0]
                    else:
                        img_name = str(index) + '_' + str(ind_img)
                    visualizations = model.render_vis_nograd(vertices=output_reproj['vertices_smal'],
                                                            focal_lengths=output_unnorm['flength'],
                                                            color=0)    #0: light blue   2: white)
                    # save image with predicted keypoints
                    out_path = save_imgs_path + '/keypoints_pred_' + img_name + '.png'
                    pred_unp = (output['keypoints_norm'][ind_img, :, :] + 1.) / 2 * (data_info.image_size - 1)
                    pred_unp_maxval = output['keypoints_scores'][ind_img, :, :]
                    pred_unp_prep = torch.cat((pred_unp, pred_unp_maxval), 1)
                    inp_img = input[ind_img, :, :, :].detach().clone()
                    save_input_image_with_keypoints(inp_img, pred_unp_prep, out_path=out_path, threshold=0.1, print_scores=True, ratio_in_out=1.0)    # threshold=0.3
                    # save predicted 3d model
                    #   (1) front view
                    pred_tex = visualizations[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                    pred_tex_max = np.max(pred_tex, axis=2)
                    out_path = save_imgs_path + '/tex_pred_' + img_name + '.png'
                    plt.imsave(out_path, pred_tex)
                    input_image = input[ind_img, :, :, :].detach().clone()
                    for t, m, s in zip(input_image, data_info.rgb_mean, data_info.rgb_stddev): t.add_(m)
                    input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0) 
                    im_masked = cv2.addWeighted(input_image_np,0.2,pred_tex,0.8,0)
                    im_masked[pred_tex_max<0.01, :] = input_image_np[pred_tex_max<0.01, :]
                    out_path = save_imgs_path + '/comp_pred_' + img_name + '.png'
                    plt.imsave(out_path, im_masked)
                    #   (2) side view
                    vertices_cent = output_reproj['vertices_smal'] - output_reproj['vertices_smal'].mean(dim=1)[:, None, :]
                    roll = np.pi / 2 * torch.ones(1).float().to(device)
                    pitch = np.pi / 2 * torch.ones(1).float().to(device)
                    tensor_0 = torch.zeros(1).float().to(device)
                    tensor_1 = torch.ones(1).float().to(device)
                    RX = torch.stack([torch.stack([tensor_1, tensor_0, tensor_0]), torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)
                    RY = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)
                    vertices_rot = (torch.matmul(RY, vertices_cent.reshape((-1, 3))[:, :, None])).reshape((batch_size, -1, 3))
                    vertices_rot[:, :, 2] = vertices_rot[:, :, 2] + torch.ones_like(vertices_rot[:, :, 2]) * 20     # 18     # *16
                    visualizations_rot = model.render_vis_nograd(vertices=vertices_rot,
                                                            focal_lengths=output_unnorm['flength'],
                                                            color=0)    # 2)
                    pred_tex = visualizations_rot[ind_img, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
                    pred_tex_max = np.max(pred_tex, axis=2)
                    out_path = save_imgs_path + '/rot_tex_pred_' + img_name + '.png'
                    plt.imsave(out_path, pred_tex)
                    render_all = True
                    if render_all:
                        # save input image 
                        inp_img = input[ind_img, :, :, :].detach().clone()
                        out_path = save_imgs_path + '/image_' + img_name + '.png'
                        save_input_image(inp_img, out_path)
                        # save posed mesh
                        V_posed = output_reproj['vertices_smal'][ind_img, :, :].detach().cpu().numpy()
                        Faces = model.smal.f
                        mesh_posed = trimesh.Trimesh(vertices=V_posed, faces=Faces, process=False)
                        mesh_posed.export(save_imgs_path + '/mesh_posed_' + img_name + '.obj')
                except:
                    print('pass...')
    return