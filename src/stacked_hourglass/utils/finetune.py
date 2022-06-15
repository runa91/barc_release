# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose

import torch
from torch.nn import Conv2d, ModuleList


def change_hg_outputs(model, indices):
    """Change the output classes of the model.

    Args:
        model: The model to modify.
        indices: An array of indices describing the new model outputs. For example, [3, 4, None]
                 will modify the model to have 3 outputs, the first two of which have parameters
                 copied from the fourth and fifth outputs of the original model.
    """
    with torch.no_grad():
        new_n_outputs = len(indices)
        new_score = ModuleList()
        for conv in model.score:
            new_conv = Conv2d(conv.in_channels, new_n_outputs, conv.kernel_size, conv.stride)
            new_conv = new_conv.to(conv.weight.device, conv.weight.dtype)
            for i, index in enumerate(indices):
                if index is not None:
                    new_conv.weight[i] = conv.weight[index]
                    new_conv.bias[i] = conv.bias[index]
            new_score.append(new_conv)
        model.score = new_score
        new_score_ = ModuleList()
        for conv in model.score_:
            new_conv = Conv2d(new_n_outputs, conv.out_channels, conv.kernel_size, conv.stride)
            new_conv = new_conv.to(conv.weight.device, conv.weight.dtype)
            for i, index in enumerate(indices):
                if index is not None:
                    new_conv.weight[:, i] = conv.weight[:, index]
            new_conv.bias = conv.bias
            new_score_.append(new_conv)
        model.score_ = new_score_
