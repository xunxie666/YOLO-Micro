import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU(), d=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False, dilation=d)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class CEM(nn.Module):
    def __init__(self, c1):
        super(CEM, self).__init__()
        c_ = 2048
        c_1 = 512
        self.cv = Conv(c1, 256, 1, 1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_1, 1, 1)
        self.cv3 = Conv(c_1, int(0.5 * c_1), 3, 1, p=3, d=3)
        self.cv4 = Conv(int(c_ + 0.5 * c_1), c_1, 1, 1)
        self.cv5 = Conv(c_1, int(0.5 * c_1), 3, 1, p=6, d=6)
        self.cv6 = Conv(c_ + c_1, c_1, 1, 1)
        self.cv7 = Conv(c_1, int(0.5 * c_1), 3, 1, p=12, d=12)
        self.cv8 = Conv(int(c_ + 1.5 * c_1), c_1, 1, 1)
        self.cv9 = Conv(c_1, int(0.5 * c_1), 3, 1, p=18, d=18)
        self.cv10 = Conv(c_ + 2 * c_1, c_1, 1, 1)
        self.cv11 = Conv(c_1, int(0.5 * c_1), 3, 1, p=24, d=24)
        self.cv12 = Conv(3 * c_1, int(0.5 * c_1), 1, 1)

    def forward(self, x):
        x_ = F.adaptive_avg_pool2d(x, (1, 1))
        x_ = self.cv(x_)
        x_up = F.interpolate(x_, size=[416, 416], mode='bilinear')
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        x3 = self.cv3(x2)
        concat1 = torch.concat([x3, x1], 1)
        x4 = self.cv4(concat1)
        x5 = self.cv5(x4)
        concat2 = torch.concat([x5, concat1], 1)
        x6 = self.cv6(concat2)
        x7 = self.cv7(x6)
        concat3 = torch.concat([x7, concat2], 1)
        x8 = self.cv8(concat3)
        x9 = self.cv9(x8)
        concat4 = torch.concat([x9, concat3], 1)
        x10 = self.cv10(concat4)
        x11 = self.cv11(x10)
        concat5 = torch.concat([x3, x5, x7, x9, x11, x_up], 1)


        return self.cv12(concat5)

# class Backbone(nn.Module):
#     def __init__(self, cout):
#         super().__init__()
#         self.model = nn.Sequential(
#             Conv(3, cout, 3, 1),
#             CEM(cout)
#         )
#
#     def forword(self, x):
#         x = self.model(x)
#         return x

class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)

        self.mp = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)

        return torch.cat([x_2, x_1], 1)


if __name__ == '__main__':
    image = torch.randn([1, 3, 224, 224])
    # cem = CEM(2048)
    # cem.eval()
    # y = cem(image)
    x = Conv(3, 10, k=(6, 4), s=1, p=0)
    x1 = x(image)
    maxLayer = nn.MaxPool2d(2, 2)
    model = maxLayer(x1)
    print(model.shape)

    # summary(cem, (3, 416, 416))

