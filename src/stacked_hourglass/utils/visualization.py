
# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

# import stacked_hourglass.datasets.utils_stanext as utils_stanext 
# COLORS, labels = utils_stanext.load_keypoint_labels_and_colours()
COLORS = ['#d82400', '#d82400', '#d82400', '#fcfc00', '#fcfc00', '#fcfc00', '#48b455', '#48b455', '#48b455', '#0090aa', '#0090aa', '#0090aa', '#d848ff', '#d848ff', '#fc90aa', '#006caa', '#d89000', '#d89000', '#fc90aa', '#006caa', '#ededed', '#ededed', '#a9d08e', '#a9d08e']
RGB_MEAN = [0.4404, 0.4440, 0.4327]
RGB_STD = [0.2458, 0.2410, 0.2468]



def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_input_image_with_keypoints(img, tpts, out_path='./test_input.png', colors=COLORS, rgb_mean=RGB_MEAN, rgb_std=RGB_STD, ratio_in_out=4., threshold=0.3, print_scores=False):
    """ 
    img has shape (3, 256, 256) and is a torch tensor
    pts has shape (20, 3) and is a torch tensor
    -> this function is tested with the mpii dataset and the results look ok
    """
    # reverse color normalization
    for t, m, s in zip(img, rgb_mean, rgb_std): t.add_(m)       # inverse to transforms.color_normalize()
    img_np = img.detach().cpu().numpy().transpose(1, 2, 0) 
    # tpts_np = tpts.detach().cpu().numpy()
    # plot image
    fig, ax = plt.subplots()
    plt.imshow(img_np)     # plt.imshow(im)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    # plot all visible keypoints
    #import pdb; pdb.set_trace()

    for idx, (x, y, v) in enumerate(tpts):
        if v > threshold:
            x = int(x*ratio_in_out)
            y = int(y*ratio_in_out)
            plt.scatter([x], [y], c=[colors[idx]], marker="x", s=50)
            if print_scores:
                txt = '{:2.2f}'.format(v.item())
                plt.annotate(txt, (x, y))        # , c=colors[idx])

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    plt.close()
    return



def save_input_image(img, out_path, colors=COLORS, rgb_mean=RGB_MEAN, rgb_std=RGB_STD):
    for t, m, s in zip(img, rgb_mean, rgb_std): t.add_(m)       # inverse to transforms.color_normalize()
    img_np = img.detach().cpu().numpy().transpose(1, 2, 0) 
    plt.imsave(out_path, img_np)
    return

######################################################################
def get_bodypart_colors():
    # body colors
    n_body = 8
    c = np.arange(1, n_body + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gist_rainbow)
    cmap.set_array([])
    body_cols = []
    for i in range(0, n_body):
        body_cols.append(cmap.to_rgba(i + 1))
    # head colors
    n_blue = 5 
    c = np.arange(1, n_blue + 1)
    norm = mpl.colors.Normalize(vmin=c.min()-1, vmax=c.max()+1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])
    head_cols = []
    for i in range(0, n_body):
        head_cols.append(cmap.to_rgba(i + 1))
    # torso colors
    n_blue = 2
    c = np.arange(1, n_blue + 1)
    norm = mpl.colors.Normalize(vmin=c.min()-1, vmax=c.max()+1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
    cmap.set_array([])
    torso_cols = []
    for i in range(0, n_body):
        torso_cols.append(cmap.to_rgba(i + 1))
    return body_cols, head_cols, torso_cols
body_cols, head_cols, torso_cols = get_bodypart_colors()
tbp_dict = {'full_body': [0, 8], 
            'head': [8, 13], 
            'torso': [13, 15]}

def save_image_with_part_segmentation(partseg_big, seg_big, input_image_np, ind_img, out_path_seg=None, out_path_seg_overlay=None, thr=0.3):
    soft_max = torch.nn.Softmax(dim=0)
    # create dit with results
    tbp_dict_res = {}
    for ind_tbp, part in enumerate(['full_body', 'head', 'torso']):
        partseg_tbp = partseg_big[:, tbp_dict[part][0]:tbp_dict[part][1], :, :]
        segm_img_pred = soft_max((partseg_tbp[ind_img, :, :, :]))   # [1, :, :]
        m_v, m_i = segm_img_pred.max(axis=0)
        tbp_dict_res[part] = {
            'inds': tbp_dict[part],
            'seg_probs': segm_img_pred,
            'seg_max_inds': m_i,
            'seg_max_values': m_v}
    # create output_image
    partseg_image = np.zeros((256, 256, 3))
    for ind_sp in range(0, 5):
        # partseg_image[tbp_dict_res['head']['seg_max_inds']==ind_sp, :] = head_cols[ind_sp][0:3]  
        mask_a = tbp_dict_res['full_body']['seg_max_inds']==1
        mask_b = tbp_dict_res['head']['seg_max_inds']==ind_sp
        partseg_image[mask_a*mask_b, :] = head_cols[ind_sp][0:3]  
    for ind_sp in range(0, 2):
        # partseg_image[tbp_dict_res['torso']['seg_max_inds']==ind_sp, :] = torso_cols[ind_sp][0:3]
        mask_a = tbp_dict_res['full_body']['seg_max_inds']==2
        mask_b = tbp_dict_res['torso']['seg_max_inds']==ind_sp
        partseg_image[mask_a*mask_b, :] = torso_cols[ind_sp][0:3]  
    for ind_sp in range(0, 8):
        if (not ind_sp == 1) and (not ind_sp == 2): # head and torso
            partseg_image[tbp_dict_res['full_body']['seg_max_inds']==ind_sp, :] = body_cols[ind_sp][0:3]  
    partseg_image[soft_max((seg_big[ind_img, :, :, :]))[1, :, :]<thr, :] = 0
    # save images
    if out_path_seg is not None:
        plt.imsave(out_path_seg, partseg_image)
    if out_path_seg_overlay is not None:
        partseg_image[soft_max((seg_big[ind_img, :, :, :]))[1, :, :]<thr, :] = input_image_np[soft_max((seg_big[ind_img, :, :, :]))[1, :, :]<thr, :]
        im_masked_partseg = cv2.addWeighted(input_image_np.astype(np.float32),0.5,partseg_image.astype(np.float32),0.5,0)
        plt.imsave(out_path_seg_overlay, im_masked_partseg)

    return


def save_image_with_part_segmentation_from_gt_annotation(partseg_annots, out_path, ind_img=0):
    # partseg_annots: (bs, 3, 256, 256)    
    # import pdb; pdb.set_trace()
    annots = partseg_annots[ind_img, :, :, :]
    partseg_image = np.zeros((256, 256, 3))
    for ind_sp in range(0, 8):
        partseg_image[annots[0, :, :]==ind_sp, :] = body_cols[ind_sp][0:3] 
    for ind_sp in range(0, 5):
        partseg_image[annots[1, :, :]==ind_sp, :] = head_cols[ind_sp][0:3]  
    for ind_sp in range(0, 2):
        partseg_image[annots[2, :, :]==ind_sp, :] = torso_cols[ind_sp][0:3]  
    plt.imsave(out_path, partseg_image.astype(np.float32))
    return


def save_image_from_prepared_partseg(partseg_init, out_path):
    # partseg_init: (256, 256, 11)
    # partseg_init = output_reproj['partseg_images_hg_nograd'][0, :, :, :].detach().cpu().numpy()
    # out_path = '/ps/scratch/nrueegg/new_projects/Animals/dog_project/pytorch-stacked-hourglass/debugging_output/partseg_hg_0.png'
    partseg = np.argmax(partseg_init, axis=2)
    partseg_image = np.zeros((256, 256, 3))
    for ind in range(partseg_init.shape[2]):
        if ind ==  0: # head
            partseg_image[partseg==ind, :] = np.asarray(head_cols[0][0:3])
        elif ind < 7:
            partseg_image[partseg==ind, :] = np.asarray(body_cols[ind+1][0:3])
        else:   # 7 to 10
            partseg_image[partseg==ind, :] = np.asarray(head_cols[ind-6][0:3])
    partseg_image[partseg_init.sum(axis=2)==0, :] = 0
    plt.imsave(out_path, partseg_image)
    return


