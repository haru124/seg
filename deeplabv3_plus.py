"""
src/seg/models/deeplabv3_plus.py
---------------------------------
DeepLabV3+ for semantic segmentation.

Supports three backbone families:
    resnet18 / resnet34 / resnet50 / resnet101   (standard ResNet)
    resnet18_v1b … resnet101_v1b                 (dilated ResNetV1b)
    mobilenet_v2                                 (MobileNetV2)

The original repo's version only wired Xception.
This version maps all three backbone families correctly.

Architecture overview:
    1. Backbone  → low-level features (C1, stride-4) + high-level (C4, stride-8/16)
    2. ASPP      → multi-scale context from C4
    3. Decoder   → upsample ASPP output, cat with C1, refine, upsample to input size
    4. Aux head  → auxiliary loss from C3 (optional, weight=0.4)

Reference:
    "Encoder-Decoder with Atrous Separable Convolution for Semantic
     Image Segmentation" — Chen et al., ECCV 2018

Output (during training):
    dict {"out": (N, C, H, W), "aux": (N, C, H, W)}  if aux=True
    dict {"out": (N, C, H, W)}                         if aux=False

Output (during eval):
    dict {"out": (N, C, H, W)}   — aux branch is suppressed at eval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.seg.models.nn.basic import _ConvBNReLU


# ══════════════════════════════════════════════════════════════════════
# ASPP — Atrous Spatial Pyramid Pooling
# ══════════════════════════════════════════════════════════════════════

class _ASPPConv(nn.Module):
    """Single atrous convolution branch in ASPP."""

    def __init__(self, in_channels, out_channels, dilation,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _ASPPPooling(nn.Module):
    """Global average pooling branch in ASPP."""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.block(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.

    Args:
        in_channels  : channels of the backbone feature map fed into ASPP
        atrous_rates : dilation rates for the 3 atrous conv branches
                       output_stride=16 → (6, 12, 18)
                       output_stride=8  → (12, 24, 36)
        out_channels : channels of each branch output (default 256)
    """

    def __init__(self, in_channels, atrous_rates=(6, 12, 18),
                 out_channels=256, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # 1×1 conv branch
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        # Three atrous conv branches
        self.b1 = _ASPPConv(in_channels, out_channels, atrous_rates[0], norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, atrous_rates[1], norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, atrous_rates[2], norm_layer)
        # Global pooling branch
        self.b4 = _ASPPPooling(in_channels, out_channels, norm_layer)

        # Project the 5 concatenated branches to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        feats = [self.b0(x), self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        x = torch.cat(feats, dim=1)
        return self.project(x)


# ══════════════════════════════════════════════════════════════════════
# Decoder head
# ══════════════════════════════════════════════════════════════════════

class _DeepLabHead(nn.Module):
    """
    Decoder: upsample ASPP output, concatenate with low-level features,
    refine with two 3×3 convs, output class logits.

    c1_channels : channels in the low-level feature map (C1 from backbone)
                  ResNet family  → 64 (BasicBlock) or 256 (Bottleneck layer1 out)
                  MobileNetV2   → 24  (inverted residual block 2 output)
    """

    def __init__(self, num_classes, c1_channels=256,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Reduce low-level features to 48 channels (paper default)
        self.c1_reduce = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)

        # Refine the concatenated features
        self.refine = nn.Sequential(
            _ConvBNReLU(256 + 48, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),   # final classification layer
        )

    def forward(self, aspp_feat, c1_feat):
        """
        aspp_feat : (N, 256, H/8, W/8) or (N, 256, H/16, W/16) from ASPP
        c1_feat   : (N, c1_channels, H/4, W/4) low-level backbone features
        """
        # Target spatial size = low-level feature size (stride-4)
        size = c1_feat.shape[-2:]

        c1 = self.c1_reduce(c1_feat)
        x  = F.interpolate(aspp_feat, size=size, mode="bilinear", align_corners=False)
        x  = torch.cat([x, c1], dim=1)   # (N, 256+48, H/4, W/4)
        return self.refine(x)


# ══════════════════════════════════════════════════════════════════════
# Auxiliary head (used only during training)
# ══════════════════════════════════════════════════════════════════════

class _AuxHead(nn.Module):
    """
    Lightweight auxiliary classifier on C3 features.
    Helps gradient flow into earlier backbone layers.
    """

    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════
# DeepLabV3+ — main model
# ══════════════════════════════════════════════════════════════════════

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with swappable backbone.

    Args:
        num_classes       : number of output segmentation classes (19 for Cityscapes)
        backbone_name     : see BACKBONE_CONFIGS below
        output_stride     : 16 (default, faster) or 8 (higher resolution ASPP)
        aux               : include auxiliary loss branch during training
        pretrained_base   : load ImageNet weights for backbone
        norm_layer        : normalisation layer (default: BatchNorm2d)
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        output_stride: int = 16,
        aux: bool = True,
        pretrained_base: bool = True,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.aux = aux

        # Build backbone and get channel info
        self.backbone, c1_channels, c3_channels, c4_channels = \
            _build_backbone(backbone_name, output_stride,
                            pretrained_base, norm_layer)

        # ASPP rates depend on output_stride
        aspp_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)

        self.aspp = ASPP(c4_channels, aspp_rates, out_channels=256,
                         norm_layer=norm_layer)
        self.head = _DeepLabHead(num_classes, c1_channels, norm_layer)

        if aux:
            self.aux_head = _AuxHead(c3_channels, num_classes, norm_layer)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Extract multi-scale features from backbone
        c1, c3, c4 = self.backbone(x)

        # Main decoder path
        aspp_out = self.aspp(c4)
        out = self.head(aspp_out, c1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        result = {"out": out}

        # Auxiliary branch (only active during training)
        if self.aux and self.training:
            aux_out = self.aux_head(c3)
            aux_out = F.interpolate(aux_out, size=input_size,
                                    mode="bilinear", align_corners=False)
            result["aux"] = aux_out

        return result


# ══════════════════════════════════════════════════════════════════════
# Backbone wrappers
# ══════════════════════════════════════════════════════════════════════

class _ResNetBackbone(nn.Module):
    """
    Wraps standard ResNet (resnet.py) to expose (c1, c3, c4) feature maps.

    c1 : output of layer1  (stride 4, 256 channels for Bottleneck, 64 for Basic)
    c3 : output of layer3  (stride 16 or 8 depending on output_stride)
    c4 : output of layer4  (stride 32 or 8 depending on output_stride)
    """

    def __init__(self, resnet):
        super().__init__()
        self.stem    = nn.Sequential(resnet.conv1, resnet.bn1,
                                     resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.layer1(x)
        x  = self.layer2(c1)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c1, c3, c4


class _MobileNetV2Backbone(nn.Module):
    """
    Wraps MobileNetV2 features list to expose (c1, c3, c4).

    MobileNetV2 inverted residual blocks (cumulative):
      features[0]   : initial conv  (stride 2, 32ch)
      features[1]   : block 1       (stride 1, 16ch)
      features[2:4] : block 2–3     (stride 2, 24ch)  ← c1 (stride 4)
      features[4:7] : block 4–6     (stride 2, 32ch)
      features[7:11]: block 7–10    (stride 2, 64ch)
      features[11:14]: block 11–13  (stride 1, 96ch)  ← c3 (stride 16)
      features[14:17]: block 14–16  (stride 2→1, 160/320ch) ← c4
      features[17]  : last conv     (1280ch)
    """

    def __init__(self, mobilenet):
        super().__init__()
        f = mobilenet.features

        # Stride 4 features (low-level, 24 channels)
        self.stage1 = nn.Sequential(*f[:4])

        # Stride 8 features
        self.stage2 = nn.Sequential(*f[4:7])

        # Stride 16 features (96 channels) — used as c3 for aux
        self.stage3 = nn.Sequential(*f[7:14])

        # Stride 16 (with dilated conv) → 320 channels for ASPP
        self.stage4 = nn.Sequential(*f[14:])

    def forward(self, x):
        c1 = self.stage1(x)   # stride 4,  24ch
        x  = self.stage2(c1)  # stride 8
        c3 = self.stage3(x)   # stride 16, 96ch
        c4 = self.stage4(c3)  # stride 16, 320ch (last conv is 1280)
        return c1, c3, c4


# ══════════════════════════════════════════════════════════════════════
# Backbone factory
# ══════════════════════════════════════════════════════════════════════

def _build_backbone(name: str, output_stride: int,
                    pretrained: bool, norm_layer):
    """
    Instantiate the requested backbone and return:
        (backbone_module, c1_channels, c3_channels, c4_channels)
    """

    dilated = (output_stride != 32)   # use dilated convs in layers 3 & 4

    # ── ResNet family (standard, from resnet.py) ──────────────────────
    if name in ("resnet18", "resnet34", "resnet50", "resnet101"):
        from src.seg.models.backbones.resnet import (
            resnet18, resnet34, resnet50, resnet101,
        )
        _builders = {
            "resnet18":  resnet18,
            "resnet34":  resnet34,
            "resnet50":  resnet50,
            "resnet101": resnet101,
        }
        base = _builders[name](pretrained=pretrained, norm_layer=norm_layer)

        # Apply dilation to layer3/layer4 for output_stride < 32
        if dilated:
            _make_dilated(base.layer3, stride=1, dilation=2)
            _make_dilated(base.layer4, stride=1, dilation=4)

        backbone = _ResNetBackbone(base)

        is_bottleneck = name in ("resnet50", "resnet101")
        c1_ch = 256 if is_bottleneck else 64
        c3_ch = 1024 if is_bottleneck else 256
        c4_ch = 2048 if is_bottleneck else 512

        return backbone, c1_ch, c3_ch, c4_ch

    # ── ResNetV1b family (dilated by default) ─────────────────────────
    if name in ("resnet18_v1b", "resnet34_v1b", "resnet50_v1b", "resnet101_v1b"):
        from src.seg.models.backbones.resnetv1b import (
            resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b,
        )
        _builders = {
            "resnet18_v1b":  resnet18_v1b,
            "resnet34_v1b":  resnet34_v1b,
            "resnet50_v1b":  resnet50_v1b,
            "resnet101_v1b": resnet101_v1b,
        }
        base = _builders[name](pretrained=pretrained,
                               dilated=dilated, norm_layer=norm_layer)
        backbone = _ResNetBackbone(base)

        is_bottleneck = name in ("resnet50_v1b", "resnet101_v1b")
        c1_ch = 256 if is_bottleneck else 64
        c3_ch = 1024 if is_bottleneck else 256
        c4_ch = 2048 if is_bottleneck else 512

        return backbone, c1_ch, c3_ch, c4_ch

    # ── MobileNetV2 ───────────────────────────────────────────────────
    if name == "mobilenet_v2":
        from src.seg.models.backbones.mobilenetv2 import get_mobilenet_v2
        base = get_mobilenet_v2(pretrained=False, norm_layer=norm_layer)
        # MobileNetV2 doesn't have an ImageNet pretrained URL in the repo;
        # set pretrained=False and load weights manually if available.
        backbone = _MobileNetV2Backbone(base)
        # c1: 24ch (stride 4), c3: 96ch (stride 16), c4: 1280ch
        return backbone, 24, 96, 1280

    raise ValueError(
        f"Unknown backbone '{name}'. "
        "Supported: resnet18, resnet34, resnet50, resnet101, "
        "resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b, mobilenet_v2"
    )


def _make_dilated(layer, stride: int, dilation: int):
    """
    Patch a ResNet layer in-place: replace strides with dilation so
    the feature map stays the same spatial size.
    Used to achieve output_stride=8 or 16 from a standard ResNet.
    """
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (stride, stride)
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding  = (dilation, dilation)


# ══════════════════════════════════════════════════════════════════════
# Public factory
# ══════════════════════════════════════════════════════════════════════

def get_segmentation_model(
    num_classes: int = 19,
    backbone: str = "resnet50",
    output_stride: int = 16,
    aux: bool = True,
    pretrained_base: bool = True,
    norm_layer=nn.BatchNorm2d,
) -> DeepLabV3Plus:
    """
    Build a DeepLabV3+ model ready for Cityscapes training.

    Args:
        num_classes    : number of classes (19 for Cityscapes)
        backbone       : backbone name (see _build_backbone for options)
        output_stride  : 16 (default) or 8
        aux            : use auxiliary loss branch
        pretrained_base: load ImageNet backbone weights
        norm_layer     : normalisation layer

    Returns:
        DeepLabV3Plus instance
    """
    model = DeepLabV3Plus(
        num_classes     = num_classes,
        backbone_name   = backbone,
        output_stride   = output_stride,
        aux             = aux,
        pretrained_base = pretrained_base,
        norm_layer      = norm_layer,
    )
    return model
