"""
src/seg/models/nn/jpu.py
------------------------
Joint Pyramid Upsampling (JPU) - optional feature fusion module.

NOT used in the basic DeepLabV3+ implementation, but available for
experimentation. It provides better multi-scale feature integration
by combining features from multiple resolution levels with multiple
dilation rates.

Reference: "Fast, Accurate and Lightweight Super-Resolution with 
           Cascading Residual Network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['JPU', 'SeparableConv2d']


class SeparableConv2d(nn.Module):
    """Separable convolution = depthwise + pointwise"""
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding,
                              dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    """
    Joint Pyramid Upsampling.
    
    Takes 3 feature maps of increasing stride and fuses them with
    multiple dilated convolutions.
    
    Args:
        in_channels : list of [in_ch_low, in_ch_mid, in_ch_high]
        width       : output channels for each feature map (default 512)
    """
    def __init__(self, in_channels, width=512, norm_layer=nn.BatchNorm2d, **kwargs):
        super(JPU, self).__init__()
        
        # Project each input to the same width
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True),
        )
        
        # Multi-scale fusion with different dilations
        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=1, dilation=1, bias=False),
            norm_layer(width),
            nn.ReLU(True),
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=2, dilation=2, bias=False),
            norm_layer(width),
            nn.ReLU(True),
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=4, dilation=4, bias=False),
            norm_layer(width),
            nn.ReLU(True),
        )
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=8, dilation=8, bias=False),
            norm_layer(width),
            nn.ReLU(True),
        )

    def forward(self, *inputs):
        """
        Args:
            inputs : (low_res, mid_res, high_res) feature maps
        
        Returns:
            (low_res, mid_res, high_res, fused_features)
        """
        # Project and align to high-res (smallest spatial size)
        feats = [
            self.conv5(inputs[-1]),  # highest stride
            self.conv4(inputs[-2]),
            self.conv3(inputs[-3]),  # lowest stride
        ]
        
        # Upsample to smallest spatial dimension (conv3's output)
        size = feats[-1].size()[2:]
        feats[-2] = F.interpolate(feats[-2], size, mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(feats[-3], size, mode='bilinear', align_corners=True)
        
        # Concatenate all features
        feat = torch.cat(feats, dim=1)  # (3*width, H, W)
        
        # Apply multi-scale dilated convolutions in parallel
        feat = torch.cat([
            self.dilation1(feat),
            self.dilation2(feat),
            self.dilation3(feat),
            self.dilation4(feat),
        ], dim=1)  # (4*width, H, W)
        
        return inputs[0], inputs[1], inputs[2], feat