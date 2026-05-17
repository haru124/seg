"""
src/seg/models/deeplabv3_plus.py
---------------------------------
DeepLabV3+ for semantic segmentation.

Backbones supported:
    resnet18 / resnet34 / resnet50 / resnet101  (via torchvision)
    mobilenet_v2                                (via torchvision)

output_stride controls backbone dilation:
    output_stride=32 : no dilation  (standard ResNet, fastest, lowest memory)
    output_stride=16 : layer4 dilated  (default, good balance)
    output_stride=8  : layer3+4 dilated  (best quality, most memory)

Architecture (original DeepLabV3+):
    backbone (with dilation) → C1 (stride 4) + C4 (stride 8 or 16)
    → ASPP → Decoder (upsample + concat C1 + refine) → output
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
        x    = F.interpolate(aspp_feat, size=size,
                             mode="bilinear", align_corners=False)
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
# Dilation helper — applied AFTER loading pretrained weights
# ══════════════════════════════════════════════════════════════════════

def _apply_dilation_to_layer(layer, stride, dilation):
    """
    Patch a ResNet layer IN-PLACE to use dilation instead of striding.

    This is applied AFTER loading pretrained weights so the weights
    themselves don't change — only the stride/dilation/padding changes.
    The 3x3 conv in each block gets dilation applied; the 1x1 convs
    and downsample are unaffected (they don't have spatial extent).

    Args:
        layer   : resnet.layer3 or resnet.layer4
        stride  : new stride for the first block (1 = no downsampling)
        dilation: dilation rate for all 3x3 convs in this layer
    """
    for i, block in enumerate(layer):
        # Fix the downsampling shortcut in the FIRST block only
        if i == 0 and block.downsample is not None:
            # Change stride in the 1x1 downsample conv
            block.downsample[0].stride = (stride, stride)

        # Fix every conv in the block
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    # This is the spatial conv — apply dilation
                    if i == 0:
                        m.stride  = (stride, stride)
                    m.dilation = (dilation, dilation)
                    m.padding  = (dilation, dilation)  # padding = dilation to preserve size


# ══════════════════════════════════════════════════════════════════════
# DeepLabV3+
# ══════════════════════════════════════════════════════════════════════

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with pretrained ResNet backbone and optional dilation.

    output_stride controls the effective stride of the backbone output:
        32 → no dilation (fastest, but coarse features)
        16 → layer4 dilated with rate=2  (default, recommended)
        8  → layer3 dilated rate=2, layer4 dilated rate=4  (best, more memory)

    use_jpu: replace ASPP input with JPU-fused features (FastFCN style)
             mutually exclusive with the standard dilated backbone path
             but can be combined if needed
    """
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        output_stride: int = 16,
        aux: bool = True,
        use_pretrained_backbone: bool = True,
        norm_layer=nn.BatchNorm2d,
        use_jpu: bool = False,
        backbone_weights_path: str = None,
    ):
        super().__init__()
        self.aux      = aux
        self.use_jpu  = use_jpu

        self.backbone, c1_ch, c3_ch, c4_ch = _build_backbone(
            backbone_name, output_stride,
            use_pretrained_backbone, norm_layer, backbone_weights_path,
        )

        # JPU fuses c1+c3+c4 into a richer feature map before ASPP
        if use_jpu:
            self.jpu = JPU(
                in_channels=[c1_ch, c3_ch, c4_ch],
                width=512,
                norm_layer=norm_layer,
            )
            aspp_in = 512 * 4   # JPU outputs 4 dilated branches of 512ch each
        else:
            aspp_in = c4_ch     # standard path: ASPP takes C4 directly

        # ASPP rates depend on output_stride
        # output_stride=8  → rates (12, 24, 36) — larger rates for larger receptive field
        # output_stride=16 → rates (6, 12, 18)
        # output_stride=32 → rates (6, 12, 18)  — no backbone dilation, ASPP does the work
        if output_stride == 8:
            aspp_rates = (12, 24, 36)
        else:
            aspp_rates = (6, 12, 18)

        self.aspp = ASPP(aspp_in, aspp_rates, out_channels=256,
                         norm_layer=norm_layer)
        self.head = _DeepLabHead(num_classes, c1_ch, norm_layer)

        if aux:
            self.aux_head = _AuxHead(c3_ch, num_classes, norm_layer)

    def forward(self, x):
        input_size   = x.shape[-2:]
        c1, c3, c4   = self.backbone(x)

        if self.use_jpu:
            c1, c3, c4, jpu_out = self.jpu(c1, c3, c4)
            aspp_out = self.aspp(jpu_out)
        else:
            aspp_out = self.aspp(c4)

        out = self.head(aspp_out, c1)
        out = F.interpolate(out, size=input_size,
                            mode="bilinear", align_corners=False)
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
    Wraps torchvision ResNet to expose (c1, c3, c4) feature maps.

    C1 = layer1 output  (stride 4)   — low-level, used in decoder
    C3 = layer3 output  (stride 8 or 16 depending on dilation)
    C4 = layer4 output  (stride 8, 16, or 32 depending on dilation)
    """
    def __init__(self, resnet):
        super().__init__()
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
    Wraps torchvision MobileNetV2 to expose (c1, c3, c4).
    No dilation support (MobileNetV2 is used as-is).
    """
    def __init__(self, mobilenet):
        super().__init__()
        f = mobilenet.features
        self.stage1 = nn.Sequential(*f[:4])    # stride 4,  24ch  → c1
        self.stage2 = nn.Sequential(*f[4:7])   # stride 8,  32ch
        self.stage3 = nn.Sequential(*f[7:14])  # stride 16, 96ch  → c3
        self.stage4 = nn.Sequential(*f[14:])   # stride 32, 1280ch → c4

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


def _build_backbone(name, output_stride, use_pretrained_backbone, norm_layer,
                    backbone_weights_path=None):

    # ── ResNet family ──────────────────────────────────────────────────
    if name in ("resnet18", "resnet34", "resnet50", "resnet101"):

        from torchvision.models import (
            resnet18  as tv_r18,
            resnet34  as tv_r34,
            resnet50  as tv_r50,
            resnet101 as tv_r101,
        )
        _tv = {"resnet18": tv_r18, "resnet34": tv_r34,
               "resnet50": tv_r50, "resnet101": tv_r101}

        # Step 1: Build architecture with NO weights
        base = _tv[name](weights=None)

        # Step 2: Load pretrained weights BEFORE applying dilation
        #         (weights are for standard stride=2 conv — load them first,
        #          then change stride/dilation/padding in-place)
        
        if use_pretrained_backbone:

            # ---------------------------------------------------------
            # Case 1: local pretrained weights
            # ---------------------------------------------------------
            if (
                backbone_weights_path is not None
                and Path(backbone_weights_path).exists()
            ):

                print(f"[Backbone] Loading local weights: {backbone_weights_path}")

                state_dict = torch.load(
                    backbone_weights_path,
                    map_location="cpu",
                    weights_only=False,
                )

                # checkpoint wrapper support
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                missing, unexpected = base.load_state_dict(
                    state_dict,
                    strict=False,
                )

                if missing:
                    print(f"[Backbone] Missing keys ({len(missing)}): {missing[:3]} ...")

                if unexpected:
                    print(f"[Backbone] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")

            # ---------------------------------------------------------
            # Case 2: download torchvision pretrained weights
            # ---------------------------------------------------------
            else:
                from src.seg.models.backbones.resnet import model_urls
                print(f"[Backbone] Downloading torchvision pretrained {name}")
                state_dict = model_zoo.load_url(model_urls[name])
                base.load_state_dict(state_dict)

        # -------------------------------------------------------------
        # Case 3: random initialization
        # -------------------------------------------------------------
        else:
            print("[Backbone] Training backbone from scratch")
        # Step 3: Apply dilation IN-PLACE AFTER weight loading
        #
        # output_stride=32: no dilation (standard ResNet)
        #   layer3: stride=2 → H/16,  layer4: stride=2 → H/32
        #
        # output_stride=16: dilate layer4 only
        #   layer3: stride=2 → H/16 (unchanged)
        #   layer4: stride=1, dilation=2 → H/16 (same as layer3 output)
        #
        # output_stride=8: dilate layer3 AND layer4
        #   layer3: stride=1, dilation=2 → H/8 (same as layer2 output)
        #   layer4: stride=1, dilation=4 → H/8 (same spatial size)
        #
        # The weights themselves don't need to change — only stride/dilation/padding.
        # This is the standard trick used in all DeepLab papers.

        if output_stride == 16:
            print("[Backbone] Applying dilation: layer4 (stride=1, dilation=2)")
            _apply_dilation_to_layer(base.layer4, stride=1, dilation=2)

        elif output_stride == 8:
            print("[Backbone] Applying dilation: layer3 (dilation=2), layer4 (dilation=4)")
            _apply_dilation_to_layer(base.layer3, stride=1, dilation=2)
            _apply_dilation_to_layer(base.layer4, stride=1, dilation=4)

        elif output_stride == 32:
            print("[Backbone] No dilation (output_stride=32, standard ResNet)")
            # No changes needed

        else:
            raise ValueError(f"output_stride must be 8, 16, or 32. Got {output_stride}")

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
                                    map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            base.load_state_dict(state_dict, strict=False)
        elif pretrained:
            url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
            print("[Backbone] Downloading MobileNetV2 pretrained weights")
            base.load_state_dict(model_zoo.load_url(url))

        backbone = _MobileNetV2Backbone(base)
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