# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose

import torch
from torch.nn.functional import interpolate


def _resize(tensor, size, mode='bilinear'):
    """Resize the image.

    Args:
        tensor (torch.Tensor): The image tensor to be resized.
        size (tuple of int): Size of the resized image (height, width).
        mode (str): The pixel sampling interpolation mode to be used.

    Returns:
        Tensor: The resized image tensor.
    """
    assert len(size) == 2

    # If the tensor is already the desired size, return it immediately.
    if tensor.shape[-2] == size[0] and tensor.shape[-1] == size[1]:
        return tensor

    if not tensor.is_floating_point():
        dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        tensor = _resize(tensor, size, mode)
        return tensor.to(dtype)

    out_shape = (*tensor.shape[:-2], *size)
    if tensor.ndimension() < 3:
        raise Exception('tensor must be at least 2D')
    elif tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndimension() > 4:
        tensor = tensor.view(-1, *tensor.shape[-3:])
    align_corners = None
    if mode in {'linear', 'bilinear', 'trilinear'}:
        align_corners = False
    resized = interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
    return resized.view(*out_shape)


def _crop(tensor, t, l, h, w, padding_mode='constant', fill=0):
    """Crop the image, padding out-of-bounds regions.

    Args:
        tensor (torch.Tensor): The image tensor to be cropped.
        t (int): Top pixel coordinate.
        l (int): Left pixel coordinate.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
        padding_mode (str): Padding mode (currently "constant" is the only valid option).
        fill (float): Fill value to use with constant padding.

    Returns:
        Tensor: The cropped image tensor.
    """
    # If the _crop region is wholly within the image, simply narrow the tensor.
    if t >= 0 and l >= 0 and t + h <= tensor.size(-2) and l + w <= tensor.size(-1):
        return tensor[..., t:t+h, l:l+w]

    if padding_mode == 'constant':
        result = torch.full((*tensor.size()[:-2], h, w), fill,
                            device=tensor.device, dtype=tensor.dtype)
    else:
        raise Exception('_crop only supports "constant" padding currently.')

    sx1 = l
    sy1 = t
    sx2 = l + w
    sy2 = t + h
    dx1 = 0
    dy1 = 0

    if sx1 < 0:
        dx1 = -sx1
        w += sx1
        sx1 = 0

    if sy1 < 0:
        dy1 = -sy1
        h += sy1
        sy1 = 0

    if sx2 >= tensor.size(-1):
        w -= sx2 - tensor.size(-1)

    if sy2 >= tensor.size(-2):
        h -= sy2 - tensor.size(-2)

    # Copy the in-bounds sub-area of the _crop region into the result tensor.
    if h > 0 and w > 0:
        src = tensor.narrow(-2, sy1, h).narrow(-1, sx1, w)
        dst = result.narrow(-2, dy1, h).narrow(-1, dx1, w)
        dst.copy_(src)

    return result


def calculate_fit_contain_output_area(in_height, in_width, out_height, out_width):
    ih, iw = in_height, in_width
    k = min(out_width / iw, out_height / ih)
    oh = round(k * ih)
    ow = round(k * iw)
    y_off = (out_height - oh) // 2
    x_off = (out_width - ow) // 2
    return y_off, x_off, oh, ow


def fit(tensor, size, fit_mode='cover', resize_mode='bilinear', *, fill=0):
    """Fit the image within the given spatial dimensions.

    Args:
        tensor (torch.Tensor): The image tensor to be fit.
        size (tuple of int): Size of the output (height, width).
        fit_mode (str): 'fill', 'contain', or 'cover'. These behave in the same way as CSS's
                        `object-fit` property.
        fill (float): padding value (only applicable in 'contain' mode).

    Returns:
        Tensor: The resized image tensor.
    """
    if fit_mode == 'fill':
        return _resize(tensor, size, mode=resize_mode)
    elif fit_mode == 'contain':
        y_off, x_off, oh, ow = calculate_fit_contain_output_area(*tensor.shape[-2:], *size)
        resized = _resize(tensor, (oh, ow), mode=resize_mode)
        result = tensor.new_full((*tensor.size()[:-2], *size), fill)
        result[..., y_off:y_off + oh, x_off:x_off + ow] = resized
        return result
    elif fit_mode == 'cover':
        ih, iw = tensor.shape[-2:]
        k = max(size[-1] / iw, size[-2] / ih)
        oh = round(k * ih)
        ow = round(k * iw)
        resized = _resize(tensor, (oh, ow), mode=resize_mode)
        y_trim = (oh - size[-2]) // 2
        x_trim = (ow - size[-1]) // 2
        result = _crop(resized, y_trim, x_trim, size[-2], size[-1])
        return result
    raise ValueError('Invalid fit_mode: ' + repr(fit_mode))
