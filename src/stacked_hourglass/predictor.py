
# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose

import torch
from stacked_hourglass.utils.evaluation import final_preds_untransformed
from stacked_hourglass.utils.imfit import fit, calculate_fit_contain_output_area
from stacked_hourglass.utils.transforms import color_normalize, fliplr, flip_back


def _check_batched(images):
    if isinstance(images, (tuple, list)):
        return True
    if images.ndimension() == 4:
        return True
    return False


class HumanPosePredictor:
    def __init__(self, model, device=None, data_info=None, input_shape=None):
        """Helper class for predicting 2D human pose joint locations.

        Args:
            model: The model for generating joint heatmaps.
            device: The computational device to use for inference.
            data_info: Specifications of the data (defaults to ``Mpii.DATA_INFO``).
            input_shape: The input dimensions of the model (height, width).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        model.to(device)
        self.model = model
        self.device = device

        if data_info is None:
            raise ValueError
            # self.data_info = Mpii.DATA_INFO
        else:
            self.data_info = data_info

        # Input shape ordering: H, W
        if input_shape is None:
            self.input_shape = (256, 256)
        elif isinstance(input_shape, int):
            self.input_shape = (input_shape, input_shape)
        else:
            self.input_shape = input_shape

    def do_forward(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

    def prepare_image(self, image):
        was_fixed_point = not image.is_floating_point()
        image = torch.empty_like(image, dtype=torch.float32).copy_(image)
        if was_fixed_point:
            image /= 255.0
        if image.shape[-2:] != self.input_shape:
            image = fit(image, self.input_shape, fit_mode='contain')
        image = color_normalize(image, self.data_info.rgb_mean, self.data_info.rgb_stddev)
        return image

    def estimate_heatmaps(self, images, flip=False):
        is_batched = _check_batched(images)
        raw_images = images if is_batched else images.unsqueeze(0)
        input_tensor = torch.empty((len(raw_images), 3, *self.input_shape),
                                   device=self.device, dtype=torch.float32)
        for i, raw_image in enumerate(raw_images):
            input_tensor[i] = self.prepare_image(raw_image)
        heatmaps = self.do_forward(input_tensor)[-1].cpu()
        if flip:
            flip_input = fliplr(input_tensor)
            flip_heatmaps = self.do_forward(flip_input)[-1].cpu()
            heatmaps += flip_back(flip_heatmaps, self.data_info.hflip_indices)
            heatmaps /= 2
        if is_batched:
            return heatmaps
        else:
            return heatmaps[0]

    def estimate_joints(self, images, flip=False):
        """Estimate human joint locations from input images.

        Images are expected to be centred on a human subject and scaled reasonably.

        Args:
            images: The images to estimate joint locations for. Can be a single image or a list
                    of images.
            flip (bool): If set to true, evaluates on flipped versions of the images as well and
                         averages the results.

        Returns:
            The predicted human joint locations in image pixel space.
        """
        is_batched = _check_batched(images)
        raw_images = images if is_batched else images.unsqueeze(0)
        heatmaps = self.estimate_heatmaps(raw_images, flip=flip).cpu()
        # final_preds_untransformed compares the first component of shape with x and second with y
        # This relates to the image Width, Height (Heatmap has shape Height, Width)
        coords = final_preds_untransformed(heatmaps, heatmaps.shape[-2:][::-1])
        # Rescale coords to pixel space of specified images.
        for i, image in enumerate(raw_images):
            # When returning to original image space we need to compensate for the fact that we are
            # used fit_mode='contain' when preparing the images for inference.
            y_off, x_off, height, width = calculate_fit_contain_output_area(*image.shape[-2:], *self.input_shape)
            coords[i, :, 1] *= self.input_shape[-2] / heatmaps.shape[-2]
            coords[i, :, 1] -= y_off
            coords[i, :, 1] *= image.shape[-2] / height
            coords[i, :, 0] *= self.input_shape[-1] / heatmaps.shape[-1]
            coords[i, :, 0] -= x_off
            coords[i, :, 0] *= image.shape[-1] / width
        if is_batched:
            return coords
        else:
            return coords[0]
