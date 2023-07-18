
import torch
import torch.nn as nn
import math
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from torchvision.ops import DeformConv2d
from PIL import Image
from torch.cuda import amp
from models.common import SAM, ECA_block, ECA
from torchvision import models

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


def split(x, groups):
    out = x.chunk(groups, dim=1)

    return out

def shuffle( x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    return out


class SGBottleneck(nn.Module):
    def __init__(self, c1, c2, s=1, use_ca=False):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1, None, 1)
        self.cv2 = Conv(c1, c1, 1, 1, None, 1)
        self.dw = DWConv(c_, c_, 3, s, False)
        self.g = Conv(c_, c_, 5, 1, None, c_)
        self.out = Conv(3 * c_, c2, 1, 1)
        self.ca = CoordAttention(c_, c_)
        self.s = s
        self.use_ca = use_ca


    def forward(self, x):
        if self.s == 1:
            x1, x2 = split(x, 2)
            x3 = self.cv2(x1)
            x4 = self.dw(x3)
            x5 = self.cv2(x4)
            if self.use_ca:
                x5 = self.ca(x5)
            # x3 = self.cv2(self.dw(self.cv2(x1)))
            x6 = self.g(x3)
            y = torch.cat([x2, x5, x6], dim=1)
        else:
            x1 = self.cv2(self.dw(self.cv1(x)))
            if self.use_ca:
                x1 = self.ca(x1)
            x2 = self.cv1(self.dw(x))
            x3 = self.g(x1)
            y = torch.cat([x1, x2, x3], dim=1)

        out = shuffle(y, 3)

        return self.out(out)




if __name__ == '__main__':
    x = torch.randn(1, 64, 40, 40)    # b, c, h, w
    eca = SAM()
    y = eca(x)
    print(y.shape)

import ntpath
import os
import glob
import shutil

# imgpathin = r'E:\BaiduNetdiskDownload\CUHK_Occlusion_Dataset\CUHK_Occlusion_Dataset\JPEG'
# imgout = r'E:\BaiduNetdiskDownload\CUHK_Occlusion_Dataset\CUHK_Occlusion_Dataset\JPEGImages'
# for subdir in os.listdir(imgpathin):
#     print(subdir)
#     file_path = os.path.join(imgpathin, subdir)
#     for subdir1 in os.listdir(file_path):
#         print(subdir1)
#         file_path1 = os.path.join(file_path, subdir1)
#         for jpg_file in os.listdir(file_path1):
#             src = os.path.join(file_path1, jpg_file)
#             new_name = str(subdir + "_" + subdir1 + "_" + jpg_file)
#             dst = os.path.join(imgout, new_name)
#             os.rename(src, dst)

