"""
src/seg/models/deeplabv3_plus.py
---------------------------------
DeepLabV3+ for semantic segmentation.

Backbones supported:
    resnet18 / resnet34 / resnet50 / resnet101  (via torchvision)
    mobilenet_v2                                (via torchvision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from src.seg.models.nn.basic import _ConvBNReLU
from src.seg.models.nn.jpu import JPU


# ══════════════════════════════════════════════════════════════════════
# ASPP
# ══════════════════════════════════════════════════════════════════════

class _ASPPConv(nn.Module):
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
        return F.interpolate(self.block(x), size=size,
                             mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18),
                 out_channels=256, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b1 = _ASPPConv(in_channels, out_channels, atrous_rates[0], norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, atrous_rates[1], norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, atrous_rates[2], norm_layer)
        self.b4 = _ASPPPooling(in_channels, out_channels, norm_layer)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        feats = [self.b0(x), self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return self.project(torch.cat(feats, dim=1))


# ══════════════════════════════════════════════════════════════════════
# Decoder
# ══════════════════════════════════════════════════════════════════════

class _DeepLabHead(nn.Module):
    def __init__(self, num_classes, c1_channels=256, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.c1_reduce = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
        self.refine = nn.Sequential(
            _ConvBNReLU(256 + 48, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, aspp_feat, c1_feat):
        size = c1_feat.shape[-2:]
        c1   = self.c1_reduce(c1_feat)
        x    = F.interpolate(aspp_feat, size=size, mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, c1], dim=1))


# ══════════════════════════════════════════════════════════════════════
# Auxiliary head
# ══════════════════════════════════════════════════════════════════════

class _AuxHead(nn.Module):
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
# DeepLabV3+
# ══════════════════════════════════════════════════════════════════════

class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        output_stride: int = 16,
        aux: bool = True,
        pretrained_base: bool = True,
        norm_layer=nn.BatchNorm2d,
        use_jpu: bool = False,
        backbone_weights_path: str = None,
    ):
        super().__init__()
        self.aux     = aux
        self.use_jpu = use_jpu

        self.backbone, c1_ch, c3_ch, c4_ch = _build_backbone(
            backbone_name, output_stride,
            pretrained_base, norm_layer, backbone_weights_path,
        )

        if use_jpu:
            self.jpu = JPU(
                in_channels=[c1_ch, c3_ch, c4_ch],
                width=512,
                norm_layer=norm_layer,
            )
            aspp_in = 512 * 4
        else:
            aspp_in = c4_ch

        aspp_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(aspp_in, aspp_rates, out_channels=256, norm_layer=norm_layer)
        self.head = _DeepLabHead(num_classes, c1_ch, norm_layer)

        if aux:
            self.aux_head = _AuxHead(c3_ch, num_classes, norm_layer)

    def forward(self, x):
        input_size = x.shape[-2:]
        c1, c3, c4 = self.backbone(x)

        if self.use_jpu:
            c1, c3, c4, jpu_out = self.jpu(c1, c3, c4)
            aspp_out = self.aspp(jpu_out)
        else:
            aspp_out = self.aspp(c4)

        out = self.head(aspp_out, c1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        result = {"out": out}

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
    Wraps a torchvision ResNet to expose (c1, c3, c4).
    torchvision ResNet has: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
    """
    def __init__(self, resnet):
        super().__init__()
        # torchvision uses 'relu' not 'relu1'
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1,
                                    resnet.relu,  resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.layer1(x)
        x  = self.layer2(c1)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c1, c3, c4


class _MobileNetV2Backbone(nn.Module):
    """
    Wraps torchvision MobileNetV2 features to expose (c1, c3, c4).

    torchvision MobileNetV2.features layout:
      [0]      ConvBNReLU  stride=2  32ch
      [1]      IR t=1      stride=1  16ch
      [2-3]    IR t=6      stride=2  24ch   ← c1 (stride 4)
      [4-6]    IR t=6      stride=2  32ch
      [7-10]   IR t=6      stride=2  64ch
      [11-13]  IR t=6      stride=1  96ch   ← c3 (stride 16)
      [14-16]  IR t=6      stride=2  160ch
      [17]     IR t=6      stride=1  320ch
      [18]     ConvBNReLU  1x1       1280ch ← c4
    Note: NO AdaptiveAvgPool in features (that's in classifier only)
    """
    def __init__(self, mobilenet):
        super().__init__()
        f = mobilenet.features          # nn.Sequential of 19 modules [0..18]
        self.stage1 = nn.Sequential(*f[:4])    # → stride 4,  24ch
        self.stage2 = nn.Sequential(*f[4:7])   # → stride 8,  32ch
        self.stage3 = nn.Sequential(*f[7:14])  # → stride 16, 96ch
        self.stage4 = nn.Sequential(*f[14:])   # → stride 32, 1280ch

    def forward(self, x):
        c1 = self.stage1(x)
        x  = self.stage2(c1)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        return c1, c3, c4


# ══════════════════════════════════════════════════════════════════════
# Backbone factory
# ══════════════════════════════════════════════════════════════════════

import torch.utils.model_zoo as model_zoo


def _build_backbone(name, output_stride, pretrained, norm_layer,
                    backbone_weights_path=None):

    # ── ResNet family ──────────────────────────────────────────────────
    if name in ("resnet18", "resnet34", "resnet50", "resnet101"):

        # Import from torchvision directly — guaranteed architecture match
        from torchvision.models import (
            resnet18  as tv_r18,
            resnet34  as tv_r34,
            resnet50  as tv_r50,
            resnet101 as tv_r101,
        )
        _tv = {"resnet18": tv_r18, "resnet34": tv_r34,
               "resnet50": tv_r50, "resnet101": tv_r101}

        # Build with NO weights first
        base = _tv[name](weights=None)

        # Load weights
        if backbone_weights_path and Path(backbone_weights_path).exists():
            print(f"[Backbone] Loading weights from {backbone_weights_path}")
            state_dict = torch.load(backbone_weights_path,
                                    map_location="cpu", weights_only=True)
            # state_dict may be wrapped — torchvision saves bare state_dicts
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = base.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Backbone] Missing keys ({len(missing)}): {missing[:5]} ...")
            if unexpected:
                print(f"[Backbone] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        elif pretrained:
            from src.seg.models.backbones.resnet import model_urls
            print(f"[Backbone] Downloading pretrained {name}")
            base.load_state_dict(model_zoo.load_url(model_urls[name]))

        backbone = _ResNetBackbone(base)

        is_bottleneck = name in ("resnet50", "resnet101")
        c1_ch = 256 if is_bottleneck else 64
        c3_ch = 1024 if is_bottleneck else 128
        c4_ch = 2048 if is_bottleneck else 512

        return backbone, c1_ch, c3_ch, c4_ch

    # ── MobileNetV2 ────────────────────────────────────────────────────
    if name == "mobilenet_v2":
        from torchvision.models import mobilenet_v2 as tv_mv2

        base = tv_mv2(weights=None)

        if backbone_weights_path and Path(backbone_weights_path).exists():
            print(f"[Backbone] Loading weights from {backbone_weights_path}")
            state_dict = torch.load(backbone_weights_path,
                                    map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            base.load_state_dict(state_dict, strict=False)
        elif pretrained:
            url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
            print("[Backbone] Downloading MobileNetV2 pretrained weights")
            base.load_state_dict(model_zoo.load_url(url))

        backbone = _MobileNetV2Backbone(base)
        # c1: 24ch (stride 4), c3: 96ch (stride 16), c4: 1280ch (stride 32)
        return backbone, 24, 96, 1280

    raise ValueError(
        f"Unknown backbone '{name}'. "
        "Supported: resnet18, resnet34, resnet50, resnet101, mobilenet_v2"
    )


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
    use_jpu: bool = False,
    backbone_weights_path: str = None,
) -> DeepLabV3Plus:
    return DeepLabV3Plus(
        num_classes           = num_classes,
        backbone_name         = backbone,
        output_stride         = output_stride,
        aux                   = aux,
        pretrained_base       = pretrained_base,
        norm_layer            = norm_layer,
        use_jpu               = use_jpu,
        backbone_weights_path = backbone_weights_path,
    )