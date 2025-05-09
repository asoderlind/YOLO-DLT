# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import CARAFEPlusPlusUpsample, Conv, DWConv, GhostConv, LightConv, MalleConv, RepConv, RFAConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CAC2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.ca = CoordinateAttention(channels=(2 + n) * self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y: torch.Tensor | list[torch.Tensor] = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.ca(y)
        return self.cv2(y)


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class MalleBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv2 = MalleConv(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RFABottleneck(Bottleneck):
    """RFA Bottleneck with RFA Convolution."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv2 = RFAConv(c_, c2, k[1], 1)


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class MalleC3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else MalleBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class RepC3k2(C3k2):
    """RepC3k2 module with RepConv blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            RepC3k(self.c, self.c, 2, shortcut, g) if c3k else RepBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class CARepC3k2(CAC2f):
    """RepC3k2 module with RepConv blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, e, g, shortcut)
        self.m = nn.ModuleList(
            RepC3k(self.c, self.c, 2, shortcut, g) if c3k else RepBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class RFAC3k2(C3k2):
    """RFA module with RepConv blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the RFAC3k2 module with RepConv blocks for efficient feature extraction."""
        super().__init__(c1, c2, n, c3k, e, g, shortcut)

        self.m = nn.ModuleList(
            RFAC3k(self.c, self.c, 2, shortcut, g) if c3k else RFABottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class EnhancedC3k2(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # Add parallel illumination-aware branch
        self.light_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c1 // 4, c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Original path
        y = super().forward(x)
        # Enhanced path - emphasizes bright regions
        light_mask = self.light_attn(x)
        return y * (1 + light_mask)  # Soft feature boosting


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepC3k(C3):
    """RepC3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the RepC3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RFAC3k(C3):
    """RFA module with RepConv blocks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the RFAC3k module with RepConv blocks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RFABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # type: ignore[import-untyped] # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class SimSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=nn.ReLU())
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class FEM(nn.Module):
    """
    Feature Enhancement Module (FEM) block.
    Adapted from https://www.mdpi.com/2227-7390/12/1/124
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        stride: int = 1,
        groups: int = 1,
        dilation_1: int = 1,
        dilation_2: int = 3,
        dilation_3: int = 5,
    ):
        super().__init__()
        self.conv1 = Conv(
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            p=autopad(kernel_size, padding, dilation_1),
            s=stride,
            g=groups,
            d=dilation_1,
            act=nn.ReLU(),
        )
        self.conv2 = Conv(
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            p=autopad(kernel_size, padding, dilation_2),
            s=stride,
            g=groups,
            d=dilation_2,
            act=nn.ReLU(),
        )
        self.conv3 = Conv(
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            p=autopad(kernel_size, padding, dilation_3),
            s=stride,
            g=groups,
            d=dilation_3,
            act=nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FEM block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, in_c, h, w]
        Returns:
            out (torch.Tensor): Output tensor with shape [b, out_c, h, w]
        """

        # branch pooled features through 3 convolutions
        return (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class EfficientFEM(nn.Module):
    """
    Efficient Feature Enhancement Module (FEM) block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        kernel_size_1=3,
        kernel_size_2=5,
        kernel_size_3=7,
    ):
        super().__init__()
        self.conv1 = DWConv(
            c1=in_channels,
            c2=out_channels,
            k=kernel_size_1,
            s=stride,
            act=nn.ReLU(),
        )
        self.conv2 = DWConv(
            c1=in_channels,
            c2=out_channels,
            k=kernel_size_2,
            s=stride,
            act=nn.ReLU(),
        )
        self.conv3 = DWConv(
            c1=in_channels,
            c2=out_channels,
            k=kernel_size_3,
            s=stride,
            act=nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FEM block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, in_c, h, w]
        Returns:
            out (torch.Tensor): Output tensor with shape [b, out_c, h, w]
        """

        # branch pooled features through 3 convolutions
        return (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class SE(nn.Module):
    """
    Squeeze and Excitation (SE) block.
    Adapted from https://arxiv.org/abs/1709.01507v4
    """

    def __init__(self, in_channels: int, reduction: int = 16, skip: bool = False) -> None:
        super().__init__()

        if in_channels < reduction:
            raise ValueError("The number of input channels must be greater than the reduction factor.")

        # squeeze operation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        excitation_hidden = in_channels // reduction
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, excitation_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(excitation_hidden, in_channels, bias=False),
            nn.Sigmoid(),
        )

        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Squeeze-and-Excitation (SE) block.

        The SE block adaptively recalibrates channel-wise feature responses by:
        1. Squeeze: Aggregating spatial information via global average pooling
        2. Excitation: Generating channel-specific weights through a bottleneck FC network
        3. Scale: Applying these weights to selectively emphasize important channels

        This channel attention mechanism helps the network focus on informative features
        and suppress less useful ones, which is particularly valuable for detecting objects
        in challenging conditions such as low-light environments.

        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, channels, height, width]

        Returns:
            out (torch.Tensor): Channel-recalibrated feature tensor of same shape as input

        Shape:
            - Input: (B, C, H, W)
            - Output: (B, C, H, W)
        """

        b, c, _, _ = x.size()
        z: torch.Tensor = self.avg_pool(x)  # [b, c, 1, 1]
        # squeeze since linear expects [b, c]
        z = z.view(b, c)  # [b, c]
        z = self.excitation(z)  # [b, c]
        # expand since we want to multiply with x
        z = z.view(b, c, 1, 1)
        return x + x * z if self.skip else x * z

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) block.
    Adapted from https://arxiv.org/abs/1910.03151.
    Code from https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """

    def __init__(self, channel: int, gamma: int = 2, b: int = 1, skip: bool = False) -> None:
        """
        Initializes the ECA block with the specified channel dimension and gamma value.

        Args:
            channel (int): Number of input channels.
            gamma (int): Scaling factor for the kernel size. Default is 2.
            b (int): Bias term for kernel size calculation. Default is 1.
        """
        super().__init__()
        # Calculate kernel size based on channel dimension
        t = int(abs((math.log(channel, 2) + b) / gamma))
        # Ensure kernel size is odd
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Efficient Channel Attention (ECA) block.

        The ECA module applies channel attention through these steps:
        1. Global Average Pooling to squeeze spatial dimensions
        2. 1D convolution across channels to capture channel dependencies
        3. Sigmoid activation to generate attention weights
        4. Channel-wise multiplication with input features

        Key differences from SE block:
        - Uses 1D convolution instead of fully connected layers
        - Adaptive kernel size based on channel dimension
        - No dimension reduction, preserving channel information

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, channels, height, width]

        Shape:
            - Input: (B, C, H, W)
            - After pooling: (B, C, 1, 1)
            - After conv: (B, C, 1, 1)
            - Output: (B, C, H, W) (same as input)

        Examples:
            >>> x = torch.randn(1, 64, 32, 32)
            >>> eca = ECA(64)
            >>> output = eca(x)
            >>> print(output.shape)
            torch.Size([1, 64, 32, 32])
        """
        y: torch.Tensor = self.avg_pool(x)  # [b, c, 1, 1]

        # prepare for 1D convolution
        y = y.squeeze(-1)  # [b, c, 1]
        y = y.transpose(-1, -2)  # [b, 1, c]

        # apply 1D convolution across channels
        y = self.conv(y)  # [b, 1, c]

        # restore original shape
        y = y.transpose(-1, -2)  # [b, c, 1]
        y = y.unsqueeze(-1)  # [b, c, 1, 1]

        # generate attention weights
        y = self.sigmoid(y)

        return x + x * y if self.skip else x * y

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class SimAM(torch.nn.Module):
    """
    Simple Attention Module (SimAM) block.
    Adapted from https://proceedings.mlr.press/v139/yang21o.html.
    Code based on https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py.
    """

    def __init__(self, e_lambda: float = 1e-4, skip: bool = False) -> None:
        """
        Initializes the Simple Attention Module (SimAM) block with the specified lambda value.

        Args:
            e_lambda (float): Small constant value to prevent division by zero. Default is 1e-4.
        """
        super().__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SimAM attention mechanism to input feature maps.

        The method calculates attention weights based on how distinctive each value is from
        the channel mean. Values that differ significantly from their channel's mean receive
        higher attention weights. This is inspired by spatial suppression in neuroscience, where
        distinctive neurons have higher importance.

        e*_t = 4(σ² + λ)/((t - μ)² + 2σ² + 2λ)
        importance = 1 / e*_t = (t - μ)²/(4(σ² + λ)) + 0.5
        (importance has done some reshuffling, but is still equal to 1 / e*_t)

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            out (torch.Tensor): Tensor of same shape as input with attention weights applied
        """
        _, _, h, w = x.size()

        # Bessel's correction since we're estimating variance from a sample
        sample_size = w * h - 1

        # Calculate squared difference from mean for each value: (t - μ)²
        diff_from_mean_squared = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # [b, c, h, w]

        # Calculate the variance for each channel: σ²
        variance = diff_from_mean_squared.sum(dim=[2, 3], keepdim=True) / sample_size  # [b, c, 1, 1]

        # Calculate importance scores using simplified formula
        # This is proportional to the inverse of the energy function
        importance_scores = diff_from_mean_squared / (4 * (variance + self.e_lambda)) + 0.5  # [b, c, h, w]

        # Apply activation (typically sigmoid) and multiply with input
        return x + x * self.activation(importance_scores) if self.skip else x * self.activation(importance_scores)


class BiFPNAdd(nn.Module):
    """
    Add operation in BiFPN that replaces the original concat operation.
    """

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        """
        Initializes the BiFPNAdd module with the specified number of inputs.

        Args:
            num_inputs (int): Number of input feature maps to be fused
            eps (float): Small constant value to prevent division by zero. Default is 1e-4.
        """
        super().__init__()
        self.eps = eps
        # One weight per input feature map
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))  # [num_inputs]

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the BiFPNAdd module performing weighted feature fusion.

        This method implements learnable weighted feature fusion as described in the
        EfficientDet paper (https://arxiv.org/abs/1911.09070). The key steps are:

        1. Apply ReLU to weights to ensure they are non-negative
        2. Normalize weights so they sum to 1 (with epsilon for numerical stability)
        3. Stack input feature maps along a new dimension
        4. Perform weighted summation using Einstein summation notation

        The weighted fusion allows the network to learn the relative importance
        of each input feature map rather than treating them equally.

        Args:
            inputs (list[torch.Tensor]): List of input tensors with identical shapes [B, C, H, W]

        Returns:
            output (torch.Tensor): Fused feature map with shape [B, C, H, W]

        Shape:
            - Inputs: List of tensors, each with shape [batch_size, channels, height, width]
            - Weights: [num_inputs]
            - Output: [batch_size, channels, height, width]
        """
        weights = F.relu(self.weights)
        normalized_weights = weights / (torch.sum(weights) + self.eps)

        # Stack inputs along a new dimension
        stacked_inputs = torch.stack(inputs, dim=0)  # shape: [num_inputs, B, C, H, W]

        # Use einsum for efficient weighted sum
        return torch.einsum("i,ibchw->bchw", normalized_weights, stacked_inputs)


class FA(nn.Module):
    """
    Implements the Feature Attention (FA) block
    Adapted from: https://www.mdpi.com/2072-4292/16/23/4493
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        """
        Initializes the Feature Attention (FA) block with the specified input channels and reduction factor.

        Args:
            in_channels (int): Number of input channels
            reduction (int): Reduction factor for the number of channels. Default is 16.
        """
        super().__init__()

        self.hidden_channels = in_channels // reduction

        self.downconv = Conv(in_channels, self.hidden_channels, 1, 1)
        self.upconv = Conv(self.hidden_channels, in_channels, 1, 1)

        self.CBR1 = Conv(self.hidden_channels, self.hidden_channels, 3, act=nn.ReLU())
        self.CBS1 = Conv(self.hidden_channels, self.hidden_channels, 3)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.CBR2 = Conv(self.hidden_channels, self.hidden_channels, 3, act=nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Feature Attention (FA) block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            out (torch.Tensor): Output tensor of shape [batch_size, channels, height, width]
        """
        x_down: torch.Tensor = self.downconv(x)  # [b, c_hidden, h, w]
        F1: torch.Tensor = self.CBS1(self.CBR1(x_down) + x_down)  # [b, c_hidden, h, w]
        F2: torch.Tensor = F1 * self.attention(F1)  # [b, c_hidden, h, w]
        F3: torch.Tensor = self.CBR2(F2) * F2  # [b, c_hidden, h, w]
        return self.upconv(F3) + x  # [b, c, h, w]


class SPPFCSP(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    # Implemented from https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py
    def __init__(self, c1, c2, k: int = 5, e: float = 0.5):
        super().__init__()
        hidden_ch = int(c2 * e)
        self.cv1 = Conv(c1, hidden_ch, k=1, s=1)
        self.cv2 = Conv(c1, hidden_ch, k=1, s=1)
        self.cv3 = Conv(hidden_ch, hidden_ch, k=3, s=1)
        self.cv4 = Conv(hidden_ch, hidden_ch, k=1, s=1)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * hidden_ch, hidden_ch, 1, 1)
        self.cv6 = Conv(hidden_ch, hidden_ch, 3, 1)
        self.cv7 = Conv(2 * hidden_ch, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_skip = self.cv2(x)
        y0: list[torch.Tensor] = [self.cv4(self.cv3(self.cv1(x)))]
        y0.extend(self.m(y0[-1]) for _ in range(3))
        y1 = torch.cat(y0, 1)
        y2: torch.Tensor = self.cv6(self.cv5(y1))
        y_out = self.cv7(torch.cat((y_skip, y2), dim=1))
        return y_out


class SimSPPFCSP(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    # Implemented from https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py
    def __init__(self, c1, c2, k: int = 5, e: float = 0.5):
        super().__init__()
        hidden_ch = int(c2 * e)
        act = nn.ReLU()
        self.cv1 = Conv(c1, hidden_ch, k=1, s=1, act=act)
        self.cv2 = Conv(c1, hidden_ch, k=1, s=1, act=act)
        self.cv3 = Conv(hidden_ch, hidden_ch, k=3, s=1, act=act)
        self.cv4 = Conv(hidden_ch, hidden_ch, k=1, s=1, act=act)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * hidden_ch, hidden_ch, 1, 1, act=act)
        self.cv6 = Conv(hidden_ch, hidden_ch, 3, 1, act=act)
        self.cv7 = Conv(2 * hidden_ch, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_skip = self.cv2(x)
        y0: list[torch.Tensor] = [self.cv4(self.cv3(self.cv1(x)))]
        y0.extend(self.m(y0[-1]) for _ in range(3))
        y1 = torch.cat(y0, 1)
        y2: torch.Tensor = self.cv6(self.cv5(y1))
        y_out = self.cv7(torch.cat((y_skip, y2), dim=1))
        return y_out


class Transpose(nn.Module):
    """Normal Transpose, default for upsampling"""

    def __init__(self, c1, c2, k=2, s=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=c1, out_channels=c2, kernel_size=k, stride=s, bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class BiC(nn.Module):
    """Bi Concat Block in PAN"""

    def __init__(self, c1: list[int], c2: int, channel_reduction: int = 1, adaptive_upsample: bool = True) -> None:
        """
        0: left -> below
        1: above -> left
        2: below -> above
        """
        super().__init__()
        hidden_ch = c2 // channel_reduction
        self.cv1 = Conv(c1[1], hidden_ch, k=1, s=1)  # left
        self.cv2 = Conv(c1[2], hidden_ch, k=1, s=1)  # above
        self.cv3 = Conv(hidden_ch * 3, c2, k=1, s=1)  # output

        if adaptive_upsample:
            self.upsample = Transpose(c1=c1[0], c2=hidden_ch)
        else:
            self.upsample = nn.Sequential(Conv(c1=c1[0], c2=hidden_ch), nn.Upsample(scale_factor=2, mode="nearest"))  # type: ignore[assignment]
        self.downsample = Conv(c1=hidden_ch, c2=hidden_ch, k=3, s=2)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


class BiC_AFR(nn.Module):
    def __init__(self, c1: list[int], c2: int, channel_reduction: int = 1, skip: bool = False) -> None:
        super().__init__()
        hidden_ch = c2 // channel_reduction
        self.cv1 = Conv(c1[1], hidden_ch, k=1)  # left
        self.cv2 = Conv(c1[2], hidden_ch, k=1)  # above
        self.upsample = Transpose(c1[0], hidden_ch)
        self.downsample = Conv(hidden_ch, hidden_ch, k=3, s=2)

        # AFR additions
        self.afr = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_ch * 3, hidden_ch // 4, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch // 4, hidden_ch * 3, 1),
            nn.Sigmoid(),  # per-channel weights
        )
        self.skip = skip
        self.cv3 = Conv(hidden_ch * 3, c2, k=1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        concat = torch.cat((x0, x1, x2), dim=1)
        weights = self.afr(concat)  # learned weights
        return self.cv3(concat * weights + concat) if self.skip else self.cv3(concat * weights)  # weighted fusion


class CARAFEBiC(nn.Module):
    """Bi Concat Block in PAN"""

    def __init__(self, c1: list[int], c2: int, channel_reduction: int = 1, adaptive_upsample: bool = True) -> None:
        """
        0: left -> below
        1: above -> left
        2: below -> above
        """
        super().__init__()
        hidden_ch = c2 // channel_reduction
        self.cv1 = Conv(c1[1], hidden_ch, k=1, s=1)  # left
        self.cv2 = Conv(c1[2], hidden_ch, k=1, s=1)  # above
        self.cv3 = Conv(hidden_ch * 3, c2, k=1, s=1)  # output

        if adaptive_upsample:
            # self.upsample = Transpose(c1=c1[0], c2=hidden_ch)
            self.upsample = nn.Sequential(
                Conv(c1=c1[0], c2=hidden_ch), CARAFEPlusPlusUpsample(channels=hidden_ch, scale_factor=2)
            )
        else:
            self.upsample = nn.Sequential(Conv(c1=c1[0], c2=hidden_ch), nn.Upsample(scale_factor=2, mode="nearest"))  # type: ignore[assignment]
        self.downsample = Conv(c1=hidden_ch, c2=hidden_ch, k=3, s=2)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


class QKVLinear(nn.Module):
    """Linear projections for query, key, value"""

    def __init__(self, dim: int, qk_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, qk_dim)
        self.k_proj = nn.Linear(dim, qk_dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, n_win^2, h, w, dim]
                where n_win^2 is the number of windows, h and w are height and width,
        Returns:
        """
        q = self.q_proj(x)  # [B, n_win^2, h, w, qk_dim]
        k = self.k_proj(x)  # [B, n_win^2, h, w, qk_dim]
        v = self.v_proj(x)  # [B, n_win^2, h, w, dim]
        return q, torch.cat([k, v], dim=-1)  # [B, n_win^2, h, w, dim + qk_dim]


class TopkRouting(nn.Module):
    """Router based on topk routing"""

    def __init__(self, qk_dim: int, topk: int, qk_scale: float = 0.0):
        super().__init__()
        self.topk = topk
        self.qk_scale = qk_scale or qk_dim**-0.5

    def forward(self, q_win: torch.Tensor, k_win: torch.Tensor) -> tuple:
        """
        Args:
            q_win: query tokens, (n, p^2, c_qk)
            k_win: key tokens, (n, p^2, c_qk)

        Returns:
            r_weight: routing weights, (n, p^2, topk)
            r_idx: routing indices, (n, p^2, topk)
        """
        # Window-based dot-product attention scores
        attn = (q_win * self.qk_scale) @ k_win.transpose(-2, -1)  # (n, p^2, p^2)

        # Get top-k routing weights and indices
        r_weight, r_idx = torch.topk(attn, k=self.topk, dim=-1)  # (n, p^2, topk)

        # Normalize routing weights
        r_weight = F.softmax(r_weight, dim=-1)

        return r_weight, r_idx


class KVGather(nn.Module):
    """Gather key/value pairs based on routing indices"""

    def __init__(self):
        super().__init__()

    def forward(self, r_idx: torch.Tensor, r_weight: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r_idx: (n, p^2, topk)
            r_weight: (n, p^2, topk)
            kv: (n, p^2, h_kv*w_kv, c_kv)

        Returns:
            kv_sel: (n, p^2, topk, h_kv*w_kv, c_kv)
        """
        n, p2, topk = r_idx.shape
        _, _, hw_kv, c_kv = kv.shape

        # Gather selected kv pairs
        kv_sel = torch.gather(
            kv.unsqueeze(2).expand(-1, -1, topk, -1, -1),
            dim=1,
            index=r_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hw_kv, c_kv),
        )

        # Apply routing weights
        kv_sel = kv_sel * r_weight.unsqueeze(-1).unsqueeze(-1)

        return kv_sel


def _grid2seq(x: torch.Tensor, region_size: tuple[int, int], num_heads: int) -> tuple[torch.Tensor, int, int]:
    """
    Convert grid features to sequence format for attention computation

    Args:
        x: BCHW tensor
        region size: (region_h, region_w)
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum("bmdhpwq->bmhwpqd", x).flatten(2, 3).flatten(-3, -2)  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w


def _seq2grid(x: torch.Tensor, region_h: int, region_w: int, region_size: tuple[int, int]) -> torch.Tensor:
    """
    Convert sequence format back to grid format

    Args:
        x: (bs, nhead, nregion, reg_size, head_dim)
    Return:
        x: (bs, C, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum("bmhwpqd->bmdhpwq", x).reshape(
        bs, nhead * head_dim, region_h * region_size[0], region_w * region_size[1]
    )
    return x


def regional_routing_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    region_graph: torch.LongTensor,
    region_size: tuple[int, int],
    kv_region_size: tuple[int, int] | None = None,
    auto_pad=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Efficient implementation of regional routing attention

    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    _, _, Hq, Wq = query.size()
    _, _, Hk, Wk = key.size()

    if auto_pad:
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if q_pad_b > 0 or q_pad_r > 0:
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))  # zero padding

        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if kv_pad_r > 0 or kv_pad_b > 0:
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))  # zero padding

    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values with efficient broadcasting approach
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).expand(
        -1, -1, -1, -1, kv_region_size, head_dim
    )
    key_g = torch.gather(
        key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).expand(-1, -1, query.size(2), -1, -1, -1),
        dim=3,
        index=broadcasted_region_graph,
    )  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(
        value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).expand(-1, -1, query.size(2), -1, -1, -1),
        dim=3,
        index=broadcasted_region_graph,
    )  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)

    # token-to-token attention
    # (bs, nhead, q_nregion, reg_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, reg_size, topk*kv_region_size)
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)

    # (bs, nhead, q_nregion, reg_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, reg_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)

    # to BCHW format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


class BiLevelRoutingAttention(nn.Module):
    """Bi-Level Routing Attention with NCHW input format (efficient implementation)"""

    def __init__(self, dim, num_heads=8, n_win=7, topk=4, side_dwconv=5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim must be divisible by num_heads!"
        self.head_dim = self.dim // num_heads
        self.scale = self.dim**-0.5

        # Local enhancement (LEPE)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, padding=side_dwconv // 2, groups=dim)

        # Regional routing settings
        self.topk = topk
        self.n_win = n_win

        # Projections
        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        N, C, H, W = x.size()
        region_size = (max(1, H // self.n_win), max(1, W // self.n_win))

        # Linear projection
        qkv = self.qkv_linear(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Region-to-region routing
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        q_r = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
        k_r = k_r.flatten(2, 3)  # nc(hw)
        a_r = q_r @ k_r  # n(hw)(hw)
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k
        idx_r = idx_r.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Token-to-token attention
        output, _ = regional_routing_attention_torch(
            query=q, key=k, value=v, scale=self.scale, region_graph=idx_r, region_size=region_size
        )

        # Add local enhancement and apply output projection
        output = output + self.lepe(v)
        output = self.output_linear(output)

        return output


class MLP(nn.Module):
    """MLP with GELU activation"""

    def __init__(self, dim: int, mlp_ratio: int = 3):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, dim), nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropPath(nn.Module):
    """Drop paths per sample during training"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNorm2d(nn.Module):
    """Layer normalization in 2D for attention"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: NCHW tensor
        """
        x = x.permute(0, 2, 3, 1)  # NHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # NCHW
        return x


class BiFormer(nn.Module):
    """BiFormer block with pre-normalization"""

    def __init__(self, dim, num_heads=8, n_win=7, topk=4, mlp_ratio=4, side_dwconv=5, drop_path=0.0):
        super().__init__()
        # Normalization layers
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        # Bi-Level Routing Attention
        self.attn = BiLevelRoutingAttention(
            dim=dim, num_heads=num_heads, n_win=n_win, topk=topk, side_dwconv=side_dwconv
        )

        # MLP block
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(mlp_ratio * dim), dim, kernel_size=1),
        )

        # Drop path for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # Attention & MLP with pre-norm
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BottleneckBiFormer(nn.Module):
    """
    BiFormerBlock with channel reduction bottleneck to reduce computational cost.

    This module:
    1. Projects input channels to a lower dimension with Conv+BN+SiLU
    2. Applies BiFormerBlock attention on the reduced channels
    3. Projects back to the original dimension with Conv+BN
    4. Optionally adds a skip connection
    """

    def __init__(
        self,
        dim: int,
        reduction_factor: int = 4,
        use_skip: bool = False,
        num_heads: int = 8,
        n_win: int = 16,
        topk: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        """
        Args:
            dim: Input/output channel dimension
            reduction_factor: Factor by which to reduce channel dimension in bottleneck
            use_skip: Whether to use a skip connection
            norm_layer: Normalization layer type
            act_layer: Activation layer type
            num_heads: Number of attention heads for BiFormerBlock
            n_win: Number of windows for BiFormerBlock
            topk: Top-k connections for routing in BiFormerBlock
            drop_path: Drop path rate for BiFormerBlock
        """
        super().__init__()

        self.use_skip = use_skip
        hidden_dim = max(dim // reduction_factor, 32)  # Ensure minimum channels

        # Dimension reduction
        self.down_block = Conv(c1=dim, c2=hidden_dim, k=1, s=1)

        # BiFormer Block operating on reduced dimensions
        self.biformer = BiFormer(dim=hidden_dim, num_heads=num_heads, n_win=n_win, topk=topk, drop_path=drop_path)

        # Dimension expansion
        self.up_block = Conv(c1=hidden_dim, c2=dim, k=1, s=1)

        # Skip connection scaling if needed
        if use_skip:
            self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        # Save input for skip connection
        identity = x

        # Down-projection to lower dimension with BN and activation
        x = self.down_block(x)

        # Apply BiFormer block
        x = self.biformer(x)

        # Up-projection to original dimension with BN
        x = self.up_block(x)

        # Apply skip connection if enabled
        if self.use_skip:
            x = identity + self.gamma * x

        return x


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention module
    https://arxiv.org/abs/2103.02907
    Code adapted from: https://github.com/houqb/CoordAttention/blob/main/coordatt.py
    """

    def __init__(self, channels: int, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        c_ = max(8, channels // reduction)

        self.conv1 = Conv(channels, c_, k=1, s=1)

        self.conv_h = nn.Conv2d(c_, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(c_, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Coordinate Attention module
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
        Returns:
            out (torch.Tensor): Output tensor of shape [batch_size, channels, height, width]
        """
        identity = x  # (b, c, h, w)

        _, _, h, w = x.size()
        x_h = self.pool_h(x)  # (b, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b, c, w, 1)
        y = torch.cat([x_h, x_w], dim=2)  # (b, c, h + w, 1)
        y = self.conv1(y)  # (b, c_, h + w, 1)

        x_h, x_w = torch.split(y, [h, w], dim=2)  # (b, c_, h, 1), (b, c_, w, 1)
        x_w = x_w.permute(0, 1, 3, 2)  # (b, c_, 1, w)
        a_h = self.conv_h(x_h).sigmoid()  # (b, c2, h, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (b, c2, 1, w)
        out = identity * a_w * a_h  # (b, c, h, w)
        return out
