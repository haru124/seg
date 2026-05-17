'''
"""
src/seg/models/backbones/mobilenetv2.py
----------------------------------------
MobileNetV2 backbone - lightweight and fast.
Perfect for edge devices or quick experimentation.

Uses inverted residual blocks (expansion -> depthwise conv -> projection)
instead of traditional residual blocks.
"""

import torch
import torch.nn as nn

__all__ = ['MobileNetV2', 'get_mobilenet_v2']


class ConvBNReLU(nn.Sequential):
    """Conv + BatchNorm + ReLU (or ReLU6)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None,
                 groups=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_ch),
            activation,
        )


class InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetV2).
    Expands to hidden_dim channels, does depthwise, then projects back.
    """
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Expansion phase (pointwise)
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        
        # Depthwise phase
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3,
                                 stride=stride, groups=hidden_dim, norm_layer=norm_layer))
        
        # Projection phase (pointwise, no activation)
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 backbone for semantic segmentation.
    
    Architecture: linear bottleneck + inverted residual blocks
    Output stride: 32 (by default)
    """
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None,
                 norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()

        if inverted_residual_setting is None:
            # (expansion_ratio, output_channels, num_blocks, stride)
            inverted_residual_setting = [
                (1, 16, 1, 1),
                (6, 24, 2, 2),
                (6, 32, 3, 2),
                (6, 64, 4, 2),
                (6, 96, 3, 1),
                (6, 160, 3, 2),
                (6, 320, 1, 1),
            ]

        # First layer: 32 channels
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        features = [ConvBNReLU(3, input_channel, kernel_size=3, stride=2, norm_layer=norm_layer)]

        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t, norm_layer))
                input_channel = output_channel

        # Last layer
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1, norm_layer=norm_layer))
        features.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_mobilenet_v2(use_pretrained_backbone=True, **kwargs):
    """
    Build MobileNetV2 backbone.
    
    Args:
        C : (not supported in this implementation)
        **kwargs   : passed to MobileNetV2 constructor
    
    Returns:
        MobileNetV2 instance
    """
    model = MobileNetV2(**kwargs)
    if use_pretrained_backbone:
        print("[MobileNetV2] Pretrained weights not available in this implementation.")
    return model
'''


"""
src/seg/models/backbones/mobilenetv2.py
----------------------------------------
Thin wrapper around torchvision MobileNetV2.
Note: AdaptiveAvgPool2d is NOT inside self.features — kept only for
      classification forward, so backbone slicing works correctly.
"""

from torchvision.models import MobileNetV2
from torchvision.models import mobilenet_v2 as _tv_mobilenet_v2

__all__ = ['MobileNetV2', 'get_mobilenet_v2']


def get_mobilenet_v2(use_pretrained_weights=False, **kwargs):
    """
    Returns a torchvision MobileNetV2 (no pretrained weights loaded here).
    kwargs are ignored — torchvision MobileNetV2 doesn't accept norm_layer
    in older versions, so we drop extra kwargs safely.
    """
    model = _tv_mobilenet_v2(weights=None)
    return model