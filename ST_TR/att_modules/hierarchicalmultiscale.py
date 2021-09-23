import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat
from .multi_scale_cross_attn import MultiScaleCrossAttn
from .cross_attention import Residual, PreNorm, FeedForward
import math
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class HierarchicalMultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        # self.msca1 = MultiScaleCrossAttn(in_channels=3, hdim=64, heads=4, downscaling_factors=[16, 4, 7], window_sizes=[7, 4, 4])
        # self.msca2 = MultiScaleCrossAttn(in_channels=64, hdim=256, heads=4, downscaling_factors=[3, 2], window_sizes=[2, 3])
        # conv1
        # first layer: channel 3 -> 32
        # self.conv = conv3x3(3, 32)
        # self.bn = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv1 = nn.Sequential(self.conv, self.bn, self.relu, self.pool)

        self.downscaling_factors_big = [7, 2, 2, 2, 2]
        self.msca1 = MultiScaleCrossAttn(in_channels=3, hdim=32, heads=1, downscaling_factors=[7, 2, 4], window_sizes=[2, 7, 7])
        self.msca2 = MultiScaleCrossAttn(in_channels=32, hdim=64, heads=1, downscaling_factors=[2, 4], window_sizes=[4, 2])
        self.msca3 = MultiScaleCrossAttn(in_channels=64, hdim=128, heads=2, downscaling_factors=[2, 4], window_sizes=[4, 2])
        self.msca4 = MultiScaleCrossAttn(in_channels=128, hdim=256, heads=4, downscaling_factors=[2, 1], window_sizes=[1, 2])
        self.msca5 = MultiScaleCrossAttn(in_channels=256, hdim=512, heads=8, downscaling_factors=[2, 1], window_sizes=[1, 2])
        self.layers = nn.ModuleList([self.msca1, self.msca2, self.msca3, self.msca4, self.msca5])

    def forward(self, x):
        """
        x: [bs, 3, 112, 112]
        """
        for i in range(len(self.layers)):
            bs, _, h, w = x.size()
            x = self.layers[i](x)
            nw, nh = h // self.downscaling_factors_big[i], w // self.downscaling_factors_big[i]
            x = x.reshape(bs, nh, nw, x.size(-1)).permute(0, 3, 1, 2)
        return x.squeeze()

if __name__ == "__main__":
    model = HierarchicalMultiScale().cuda()

    # for i in range(100):
    x = torch.randn(6, 3, 112, 112).cuda()
    out = model(x)
    print(out.shape)