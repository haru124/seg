"""
src/seg/models/nn/basic.py
---------------------------
Helper modules used across the segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    '_ConvBNReLU', '_ConvBNPReLU', '_ConvBN', '_BNPReLU',
    '_DepthwiseConv', 'InvertedResidual', '_PSPModule'
]


class _ConvBNReLU(nn.Module):
    """Conv + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn   = norm_layer(out_channels)
        self.relu = nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _ConvBNPReLU(nn.Module):
    """Conv + BatchNorm + PReLU"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=False)
        self.bn    = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class _ConvBN(nn.Module):
    """Conv + BatchNorm (no activation)"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn   = norm_layer(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class _BNPReLU(nn.Module):
    """BatchNorm + PReLU"""
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.bn    = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.bn(x))


class _DepthwiseConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, stride,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels, 3, stride=stride,
                        padding=1, groups=in_channels, norm_layer=norm_layer),
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer),
        )

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2 style)"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        assert stride in [1, 2]
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        hidden = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(_ConvBNReLU(in_channels, hidden, 1,
                                      relu6=True, norm_layer=norm_layer))
        layers.extend([
            _ConvBNReLU(hidden, hidden, 3, stride=stride, padding=1,
                        groups=hidden, relu6=True, norm_layer=norm_layer),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            norm_layer(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class _PSPModule(nn.Module):
    """Pyramid Scene Parsing Module"""
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), **kwargs):
        super().__init__()
        out_channels = in_channels // 4
        self.avgpools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in sizes])
        self.convs    = nn.ModuleList([
            _ConvBNReLU(in_channels, out_channels, 1, **kwargs) for _ in sizes
        ])

    def forward(self, x):
        size  = x.shape[-2:]
        feats = [x]
        for pool, conv in zip(self.avgpools, self.convs):
            feats.append(F.interpolate(conv(pool(x)), size,
                                       mode='bilinear', align_corners=True))
        return torch.cat(feats, dim=1)