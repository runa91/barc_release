import argparse
import os.path

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.mpii import Mpii, print_mpii_validation_accuracy
from stacked_hourglass.train import do_validation_epoch


def main(args):
    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    pretrained = not args.model_file

    if pretrained:
        print('No model weights file specified, using pretrained weights instead.')

    # Create the model, downloading pretrained weights if necessary.
    if args.arch == 'hg1':
        model = hg1(pretrained=pretrained)
    elif args.arch == 'hg2':
        model = hg2(pretrained=pretrained)
    elif args.arch == 'hg8':
        model = hg8(pretrained=pretrained)
    else:
        raise Exception('unrecognised model architecture: ' + args.model)
    model = model.to(device)

    if not pretrained:
        assert os.path.isfile(args.model_file)
        print('Loading model weights from file: {}'.format(args.model_file))
        checkpoint = torch.load(args.model_file)
        state_dict = checkpoint['state_dict']
        if sorted(state_dict.keys())[0].startswith('module.'):
            model = DataParallel(model)
        model.load_state_dict(state_dict)

    # Initialise the MPII validation set dataloader.
    val_dataset = Mpii(args.image_path, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Generate predictions for the validation set.
    _, _, predictions = do_validation_epoch(val_loader, model, device, Mpii.DATA_INFO, args.flip)

    # Report PCKh for the predictions.
    print('\nFinal validation PCKh scores:\n')
    print_mpii_validation_accuracy(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--image-path', required=True, type=str,
                        help='path to MPII Human Pose images')
    parser.add_argument('--arch', metavar='ARCH', default='hg1',
                        choices=['hg1', 'hg2', 'hg8'],
                        help='model architecture')
    parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--batch-size', default=6, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--flip', dest='flip', action='store_true',
                        help='flip the input during validation')

    main(parser.parse_args())
