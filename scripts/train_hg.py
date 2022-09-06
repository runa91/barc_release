import argparse
import os

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import sys,os
sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../','src'))

# from src.src_test2 import Test2
from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.stanext24 import StanExt
# from src.stacked_hourglass.datasets.mpii import Mpii

from stacked_hourglass.train import do_training_epoch, do_validation_epoch
from stacked_hourglass.utils.logger import Logger
from stacked_hourglass.utils.misc import save_checkpoint, adjust_learning_rate

from configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated

def main(args):
    # Select the hardware device to use for inference.
    path_config = os.path.join(get_cfg_defaults().barc_dir, 'src', 'configs', args.config)
    update_cfg_global_with_yaml(path_config)
    cfg = get_cfg_global_updated()
    print(f"cfg {cfg}")
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations by default.
    torch.set_grad_enabled(False)

    # create checkpoint dir
    os.makedirs(args.checkpoint, exist_ok=True)
    if args.arch == 'hg1':
        model = hg1(pretrained=False,num_classes=cfg.params.N_CLASSES, num_partseg=cfg.params.N_PARTSEG, upsample_seg=cfg.params.UPSAMPLE_SEG, add_partseg=cfg.params.ADD_PARTSEG)
        # model = hg1(pretrained=False, upsample_seg=True,add_partseg=True)
    elif args.arch == 'hg2':
        # model = hg2(pretrained=False)
        model = hg2(pretrained=False,num_classes=cfg.params.N_CLASSES, num_partseg=cfg.params.N_PARTSEG, upsample_seg=cfg.params.UPSAMPLE_SEG, add_partseg=cfg.params.ADD_PARTSEG)
    elif args.arch == 'hg8':
        # model = hg8(pretrained=False)
        model = hg8(pretrained=False,num_classes=cfg.params.N_CLASSES, num_partseg=cfg.params.N_PARTSEG, upsample_seg=cfg.params.UPSAMPLE_SEG, add_partseg=cfg.params.ADD_PARTSEG)
    else:
        raise Exception('unrecognised model architecture: ' + args.arch)

    model = DataParallel(model).to(device)

    optimizer = RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)

    best_acc = 0

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    # create data loader
    # StanExt
    train_dataset = StanExt(args.image_path, is_train=True, inp_res=args.input_shape,V12=True,dataset_mode='keyp_and_seg')
    # train_dataset = Mpii(args.image_path, is_train=True, inp_res=args.input_shape)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_dataset = StanExt(args.image_path, is_train=False, inp_res=args.input_shape,V12=True,dataset_mode='keyp_and_seg')
    # val_dataset = Mpii(args.image_path, is_train=False, inp_res=args.input_shape)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # train and eval
    lr = args.lr
    for epoch in trange(args.start_epoch, args.epochs, desc='Overall', ascii=True):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)

        # train for one epoch
        train_loss, train_acc = do_training_epoch(train_loader, model, device, StanExt.DATA_INFO ,
                                                  optimizer,quiet=False ,
                                                  acc_joints=StanExt.ACC_JOINTS)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = do_validation_epoch(val_loader, model, device,
                                                                 StanExt.DATA_INFO, False,
                                                                 acc_joints=StanExt.ACC_JOINTS)

        # print metrics
        tqdm.write(f'[{epoch + 1:3d}/{args.epochs:3d}] lr={lr:0.2e} '
                   f'train_loss={train_loss:0.4f} train_acc={100 * train_acc:0.2f} '
                   f'valid_loss={valid_loss:0.4f} valid_acc={100 * valid_acc:0.2f}')

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])
        logger.plot_to_file(os.path.join(args.checkpoint, 'log.svg'), ['Train Acc', 'Val Acc'])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a stacked hourglass model.')
    # Dataset setting
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')
    parser.add_argument('-cg', '--config', default='barc_cfg_train.yaml', type=str, metavar='PATH',
                        help='name of config file (default: barc_cfg_train.yaml within src/configs folder)')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg8',
                        choices=['hg1', 'hg2', 'hg8'],
                        help='model architecture')
    # Training strategy
    parser.add_argument('--input_shape', default=256, type=int, nargs='+',
                        help='Input shape of the model. Given as: (H, W)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    main(parser.parse_args())
