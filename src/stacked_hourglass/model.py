# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose
# Hourglass network inserted in the pre-activated Resnet
# Use lr=0.01 for current version
# (c) YANG, Wei


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


__all__ = ['HourglassNet', 'hg']


model_urls = {
    'hg1': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg1-ce125879.pth',
    'hg2': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg2-15e342d9.pth',
    'hg8': 'https://github.com/anibali/pytorch-stacked-hourglass/releases/download/v0.0.0/bearpaw_hg8-90e5d470.pth',
}


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, upsample_seg=False, add_partseg=False, num_partseg=None):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample_seg = upsample_seg
        self.add_partseg = add_partseg

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

        if self.add_partseg:
            self.hg_ps = (Hourglass(block, num_blocks, self.num_feats, 4))
            self.res_ps = (self._make_residual(block, self.num_feats, num_blocks))
            self.fc_ps = (self._make_fc(ch, ch))
            self.score_ps = (nn.Conv2d(ch, num_partseg, kernel_size=1, bias=True))
            self.ups_upsampling_ps = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


        if self.upsample_seg:
            self.ups_upsampling = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.ups_conv0 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3,
                        bias=True)
            self.ups_bn1 = nn.BatchNorm2d(32)
            self.ups_conv1 = nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3,
                        bias=True)
            self.ups_bn2 = nn.BatchNorm2d(16+2)
            self.ups_conv2 = nn.Conv2d(16+2, 16, kernel_size=5, stride=1, padding=2,
                        bias=True)
            self.ups_bn3 = nn.BatchNorm2d(16)
            self.ups_conv3 = nn.Conv2d(16, 2, kernel_size=5, stride=1, padding=2,
                        bias=True)



    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x_in):
        out = []
        out_seg = []
        out_partseg = []
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            if i == self.num_stacks - 1:
                if self.add_partseg:
                    y_ps = self.hg_ps(x)
                    y_ps = self.res_ps(y_ps)
                    y_ps = self.fc_ps(y_ps)
                    score_ps = self.score_ps(y_ps)
                    out_partseg.append(score_ps[:, :, :, :])
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            if self.upsample_seg:
                out.append(score[:, :-2, :, :])
                out_seg.append(score[:, -2:, :, :])
            else:
                out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        if self.upsample_seg:
            # PLAN: add a residual to the upsampled version of the segmentation image
            # upsample predicted segmentation
            seg_score = score[:, -2:, :, :]
            seg_score_256 = self.ups_upsampling(seg_score)
            # prepare input image

            ups_img = self.ups_conv0(x_in)

            ups_img = self.ups_bn1(ups_img)
            ups_img = self.relu(ups_img)
            ups_img = self.ups_conv1(ups_img)

            # import pdb; pdb.set_trace()

            ups_conc = torch.cat((seg_score_256, ups_img), 1)

            # ups_conc = self.ups_bn2(ups_conc)
            ups_conc = self.relu(ups_conc)
            ups_conc = self.ups_conv2(ups_conc)

            ups_conc = self.ups_bn3(ups_conc)
            ups_conc = self.relu(ups_conc)
            correction = self.ups_conv3(ups_conc)

            seg_final = seg_score_256 + correction

            if self.add_partseg:
                partseg_final = self.ups_upsampling_ps(score_ps)
                out_dict = {'out_list_kp': out,
                            'out_list_seg': out,
                            'seg_final': seg_final,
                            'out_list_partseg': out_partseg,
                            'partseg_final': partseg_final
                            }
                return out_dict
            else:
                out_dict = {'out_list_kp': out,
                            'out_list_seg': out,
                            'seg_final': seg_final
                            }
                return out_dict

        return out     


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'], upsample_seg=kwargs['upsample_seg'],
                         add_partseg=kwargs['add_partseg'], num_partseg=kwargs['num_partseg'])
    return model


def _hg(arch, pretrained, progress, **kwargs):
    model = hg(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def hg1(pretrained=False, progress=True, num_blocks=1, num_classes=16, upsample_seg=False, add_partseg=False, num_partseg=None):
    return _hg('hg1', pretrained, progress, num_stacks=1, num_blocks=num_blocks,
               num_classes=num_classes, upsample_seg=upsample_seg,
               add_partseg=add_partseg, num_partseg=num_partseg)


def hg2(pretrained=False, progress=True, num_blocks=1, num_classes=16, upsample_seg=False, add_partseg=False, num_partseg=None):
    return _hg('hg2', pretrained, progress, num_stacks=2, num_blocks=num_blocks,
               num_classes=num_classes, upsample_seg=upsample_seg,
               add_partseg=add_partseg, num_partseg=num_partseg)

def hg4(pretrained=False, progress=True, num_blocks=1, num_classes=16, upsample_seg=False, add_partseg=False, num_partseg=None):
    return _hg('hg4', pretrained, progress, num_stacks=4, num_blocks=num_blocks,
               num_classes=num_classes, upsample_seg=upsample_seg,
               add_partseg=add_partseg, num_partseg=num_partseg)

def hg8(pretrained=False, progress=True, num_blocks=1, num_classes=16, upsample_seg=False, add_partseg=False, num_partseg=None):
    return _hg('hg8', pretrained, progress, num_stacks=8, num_blocks=num_blocks,
               num_classes=num_classes, upsample_seg=upsample_seg,
               add_partseg=add_partseg, num_partseg=num_partseg)
