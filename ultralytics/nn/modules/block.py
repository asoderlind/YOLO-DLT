# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, RFAConv, autopad
from .transformer import TransformerBlock
from .utils import FAM_MODE, FSM_TYPE

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


class RepC3k2(C3k2):
    """RepC3k2 module with RepConv blocks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
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


class TemporalAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        qkv_bias=False,
        attn_drop: float = 0.0,
        scale: int = 25,
        mode: FAM_MODE = "both",
    ) -> None:
        # channels :input[batch_size,sequence_length, channels]-->output[batch_size, sequence_length, channels]
        # qkv_bias : Does it matter? (No)
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        # head_dim = channels // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5
        self.qkv_cls_linear = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.qkv_reg_linear = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mode = mode

    def forward(
        self,
        x_cls: torch.Tensor,
        x_reg: torch.Tensor,
        cls_score: torch.Tensor,
        ave: bool = True,
        sim_thresh: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TemporalAttention module.
        batch_size will be 1 since we are processing flattened sequences.

        Args:
            x_cls (torch.Tensor): Classification branch features [batch_size, sequence_length, channels]
            x_reg (torch.Tensor): Regression branch features [batch_size, sequence_length, channels]
            cls_score (torch.Tensor): Classification confidence scores [batch_size, sequence_length]
            ave: Whether to use average pooling over reference features

        Returns:
            x_cls (torch.Tensor): Classification branch features with attention applied &  [batch_size, sequence_length, 2 * channels]
            x_reg (None): Regression branch features (not used currently)
            sim_round2 (torch.Tensor): Similarity weights for average pooling [batch_size, sequence_length, sequence_length]
        """
        device = x_cls.device
        B, N, C = x_cls.shape  # batch size will be 1 since we are processing flattened sequences
        # PART 1: PREPARATION - Create Q, K, V matrices
        # Transform features to query, key, value representations
        qkv_cls = (
            self.qkv_cls_linear(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )  # [3, B=1, num_heads, N, C // num_heads]
        qkv_reg = (
            self.qkv_reg_linear(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )  # [3, 1, num_heads, N, C // num_heads]

        # Unpack Q, K, V from qkv tensors
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # [B=1, num_heads, N, C // num_heads]
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        # Normalize features for numerical stability (implementation detail)
        # In TemporalAttention.forward():
        # Safe normalization with built-in epsilon handling
        q_cls = F.normalize(q_cls, dim=-1, eps=1e-6)
        k_cls = F.normalize(k_cls, dim=-1, eps=1e-6)
        q_reg = F.normalize(q_reg, dim=-1, eps=1e-6)
        k_reg = F.normalize(k_reg, dim=-1, eps=1e-6)
        v_cls_normed = F.normalize(v_cls, dim=-1, eps=1e-6)
        v_reg_normed = F.normalize(v_reg, dim=-1, eps=1e-6)

        # Prepare confidence scores matrix for attention weighting
        # cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)  # [1, num_heads, N, N]
        # Assuming cls_score has shape [B, N]
        cls_score = cls_score.unsqueeze(1).unsqueeze(2)  # [B=1, 1, 1, N]
        cls_score = cls_score.repeat(1, self.num_heads, N, 1)  # [B, num_heads, N, N]

        # PART 2: AFFINITY MANNER (A.M.) - Section with confidence-weighted attention
        # Compute attention scores for classification branch
        cls_mult = self.scale * cls_score if self.mode in ["both", "cls"] else self.scale
        attn_cls: torch.Tensor = (q_cls @ k_cls.transpose(-2, -1)) * cls_mult  # [B = 1, num_heads, N, N]
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)  # [B = 1, num_heads, N, N]

        # Compute attention scores for regression branch
        attn_reg: torch.Tensor = (q_reg @ k_reg.transpose(-2, -1)) * self.scale
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)  # [B = 1, num_heads, N, N]

        # Combine attention from both branches (equivalent to SA_c(C) + SA_r(R) in paper)
        attn = (attn_reg + attn_cls) / 2  # [B, num_heads, N, N]

        x_cls_out = torch.zeros([B, N, 2 * C], device=device)  # [B=1, N, 2C]
        x_reg_out = torch.zeros([B, N, 2 * C], device=device)  # [B=1, N, 2C]

        match self.mode:
            case "both":
                # Apply combined attention to v_cls (the aggregation step)
                x_cls_inter = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
                x_reg_inter = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)

                # Original features for concatenation (this is the V in the paper)
                x_ori_cls = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
                x_ori_reg = v_reg.permute(0, 2, 1, 3).reshape(B, N, C)

                # This is the SA(F) = concat((SA_c(C) + SA_r(R)), V_c) operation from the paper
                x_cls_out = torch.cat([x_cls_inter, x_ori_cls], dim=-1)  # [B, N, 2 * C]
                x_reg_out = torch.cat([x_reg_inter, x_ori_reg], dim=-1)  # [B, N, 2 * C]
            case "cls":
                x_cls_inter = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
                x_ori_cls = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)  # [B, N, C]
                x_cls_out = torch.cat([x_cls_inter, x_ori_cls], dim=-1)  # [B, N, 2 * C]
            case "reg":
                x_reg_inter = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)
                x_ori_reg = v_reg.permute(0, 2, 1, 3).reshape(B, N, C)
                x_reg_out = torch.cat([x_reg_inter, x_ori_reg], dim=-1)  # [B, N, 2 * C]

        # Return early if we're not using average pooling
        if not ave:
            return x_cls_out, x_reg_out

        # PART 3: AVERAGE POOLING OVER REFERENCE FEATURES (A.P.)
        # This corresponds to N(V_c)N(V_c)^T in the paper
        # Result is one attn matrix per head that represents the average cosine similarity across heads
        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)  # [B, num_heads, N, N]

        # Average the similarity matrix across attention heads
        attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False) / self.num_heads  # [B, N, N]

        # Create binary mask for features with similarity > threshold τ (0.75)
        ones_matrix = torch.ones(attn.shape[2:], device=device)  # [N, N]
        zero_matrix = torch.zeros(attn.shape[2:], device=device)  # [N, N]

        # Average attention weights across heads
        sim_attn = torch.sum(attn, dim=1, keepdim=False) / self.num_heads  # [B, N, N]

        # Create similarity mask based on threshold
        sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)  # [B, N, N]

        # Apply mask and renormalize for average pooling
        # The positions with < thresh will not contribute to the final feature
        sim_round2 = torch.softmax(sim_attn, dim=-1)  # raw attention scores -> prob distributions, each row sums to 1
        sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))  # [B, N, N]

        # Calculate obj_mask only for "both" or "reg" modes
        if self.mode in ["both", "reg"]:
            conf_sim_thresh: float = 0.99  # this value is from the paper
            attn_reg_raw = v_reg_normed @ v_reg_normed.transpose(-2, -1)  # [B, num_heads, N, N]
            attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False) / self.num_heads  # [B, N, N]
            obj_mask = torch.where(attn_reg_raw > conf_sim_thresh, ones_matrix, zero_matrix)  # [B, N, N]
            obj_mask = obj_mask * sim_round2 / torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)  # [B, N, N]
        else:
            # For "cls" mode, just use identity or zero matrix as needed
            obj_mask = sim_round2  # Just use sim_round2 as a placeholder

        # Return concatenated features and similarity weights for further processing
        return x_cls_out, x_reg_out, sim_round2, obj_mask

    def __call__(
        self, x_cls: torch.Tensor, x_reg: torch.Tensor, cls_score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(x_cls, x_reg, cls_score)


class FeatureAggregationModule(nn.Module):
    """
    Feature Aggregation Module (FAM) for enhancing object detection in videos.

    Aggregates features across frames to improve detection quality by leveraging
    temporal information, particularly useful for challenging conditions like
    nighttime detection.
    """

    def __init__(
        self,
        cls_ch: int,
        reg_ch: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        scale: int = 25,
        mode: FAM_MODE = "both",
    ):
        super().__init__()
        self.common_ch = max(cls_ch, reg_ch)

        # Projection layers for cls and reg features
        self.cls_proj = nn.Conv2d(cls_ch, self.common_ch, kernel_size=1, bias=False)
        self.reg_proj = nn.Conv2d(reg_ch, self.common_ch, kernel_size=1, bias=False)

        self.temporal_attention = TemporalAttention(
            self.common_ch, num_heads, qkv_bias, attn_drop, scale=scale, mode=mode
        )

        self.mode = mode

        match mode:
            case "both":
                self.linear1_cls = nn.Linear(2 * self.common_ch, 2 * self.common_ch)
                self.linear1_reg = nn.Linear(2 * self.common_ch, 2 * self.common_ch)
                self.linear2_cls = nn.Linear(4 * self.common_ch, 4 * self.common_ch)
                self.linear2_reg = nn.Linear(4 * self.common_ch, 4 * self.common_ch)
            case "cls":
                self.linear1_cls = nn.Linear(2 * self.common_ch, 2 * self.common_ch)
                self.linear2_cls = nn.Linear(4 * self.common_ch, 4 * self.common_ch)
                self.linear1_reg = nn.Identity()  # type: ignore[assignment]
                self.linear2_reg = nn.Identity()  # type: ignore[assignment]
            case "reg":
                self.linear1_cls = nn.Identity()  # type: ignore[assignment]
                self.linear2_cls = nn.Identity()  # type: ignore[assignment]
                self.linear1_reg = nn.Linear(2 * self.common_ch, 2 * self.common_ch)
                self.linear2_reg = nn.Linear(4 * self.common_ch, 4 * self.common_ch)

    def weighted_feature_enrichment(self, features: torch.Tensor, similarity_weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted feature enrichment based on the similiarity weights from `TemporalAttention`.

        Args:
            features: Transformed features from temporal attention [B, N, 2*common_channel_dim]
            similarity_weights: Weights for averaging reference features [B, N, N]

        Returns:
            Combined features after weighted pooling [B, N, 4*common_channel_dim]
        """

        # Cast similarity weights to match feature dtype for mixed precision compatibility
        if not self.training:
            similarity_weights = similarity_weights.to(features.dtype)
        # Weighted aggregation of support features using similarity scores
        weighted_features = torch.bmm(similarity_weights, features)  # [B, N, 2*common_ch]

        # Concatenate weighted features with key features
        combined_features = torch.cat([weighted_features, features], dim=-1)  # [B, N, 4*common_ch]
        return combined_features

    def forward(
        self,
        cls_features: torch.Tensor,
        reg_features: torch.Tensor,
        cls_scores: torch.Tensor,
    ):
        """
        Forward pass of the Feature Aggregation Module.

        Args:
            cls_features (torch.tensor):  Classification branch features [1, N, C_cls]
            reg_features (torch.tensor):  Regression branch features [1, N, C_reg]
            cls_scores (torch.tensor): Confidence scores for classification [1, N]
            boxes (torch.tensor): Bounding boxes for the current frame [1, N, 4]
        """
        B, N, C_cls = cls_features.shape
        _, _, C_reg = reg_features.shape
        # Reshape for 1x1 convolutions (treating the N dimension as height * width)
        if C_cls != self.common_ch:
            cls_features_reshaped = cls_features.view(B, 1, N, C_cls).permute(0, 3, 1, 2)  # [1, C_cls, 1, N]
            cls_features = self.cls_proj(cls_features_reshaped)
            cls_features = cls_features.permute(0, 2, 3, 1).view(B, N, self.common_ch)  # [1, N, common_ch]
        if C_reg != self.common_ch:
            reg_features_reshaped = reg_features.view(B, 1, N, C_reg).permute(0, 3, 1, 2)  # [1, C_reg, 1, N]
            reg_features = self.reg_proj(reg_features_reshaped)
            reg_features = reg_features.permute(0, 2, 3, 1).view(B, N, self.common_ch)  # [1, N, common_ch]

        # [1, N, 2*common_ch], [1, N, N]
        enhanced_cls_features, enhanced_reg_features, similarity_weights, reg_similarity_weights = (
            self.temporal_attention(cls_features, reg_features, cls_scores)  # type: ignore[misc]
        )

        # First feature transformation
        transformed_cls_features = self.linear1_cls(enhanced_cls_features)  # [B, N, 2*common_ch]
        transformed_reg_features = self.linear1_reg(enhanced_reg_features)  # [B, N, 2*common_ch]

        match self.mode:
            case "both":
                pooled_cls_features = self.weighted_feature_enrichment(
                    transformed_cls_features, similarity_weights
                )  # [B, N, 4*common_ch]
                pooled_reg_features = self.weighted_feature_enrichment(
                    transformed_reg_features, reg_similarity_weights
                )  # [B, N, 4*common_ch]
            case "cls":
                pooled_cls_features = self.weighted_feature_enrichment(
                    transformed_cls_features, similarity_weights
                )  # [B, N, 4*common_ch]
                pooled_reg_features = transformed_reg_features  # [B, N, 2*common_ch]
            case "reg":
                pooled_cls_features = transformed_cls_features  # [B, N, 2*common_ch]
                pooled_reg_features = self.weighted_feature_enrichment(
                    transformed_reg_features, reg_similarity_weights
                )  # [B, N, 4*common_ch]

        # Second feature transformation
        final_cls_features: torch.Tensor = self.linear2_cls(pooled_cls_features)  # [B, N, 4*common_ch]
        final_reg_features: torch.Tensor = self.linear2_reg(pooled_reg_features)

        # TODO: maybe use local aggregation here
        return final_cls_features, final_reg_features

    def __call__(
        self,
        cls_features: torch.Tensor,
        reg_features: torch.Tensor,
        cls_scores: torch.Tensor,
    ):
        return self.forward(cls_features, reg_features, cls_scores)


class FeatureSelectionModule(nn.Module):
    """
    Feature Selection Module (FSM) for YOLOV++ video object detection.
    Selects high-quality candidate regions from raw predictions for temporal aggregation.
    """

    def __init__(
        self,
        fsm_type: FSM_TYPE = "nms",
        conf_thresh: float = 0.001,
        nms_thresh_train: float = 0.75,
        nms_thresh_val: float = 0.4,
        topk_pre: int = 750,
        topk_post: int = 30,
        max_thresh_proposals_per_frame: int = 85,
    ):
        """
        Initialize the Feature Selection Module.

        Args:
            method (str): Selection method, either "thresh" or "nms"

            thresh_count (int): Target number of proposals to select for threshold method
            conf_thresh (float): Confidence threshold for filtering predictions for threshold method

            nms_thresh (float): NMS threshold for removing redundant detections (used only with "nms" method)
            topk_pre (int): Number of top proposals to consider before NMS (used only with "nms" method)
            topk_post (int): Number of top proposals to keep after NMS (used only with "nms" method)
        """
        super().__init__()
        self.fsm_type = fsm_type
        self.conf_thresh = conf_thresh
        self.nms_thresh_train = nms_thresh_train
        self.nms_thresh_val = nms_thresh_val
        self.topk_pre = topk_pre
        self.topk_post = topk_post
        self.max_thresh_proposals_per_frame = max_thresh_proposals_per_frame

    def __call__(
        self, raw_preds: torch.Tensor, vid_features: torch.Tensor, reg_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select features based on the configured method.

        Returns:
            tuple: (selected_cls_features, selected_reg_features, selected_boxes,
                   selected_scores, selected_indices)
        """
        if self.fsm_type == "thresh":
            return self._threshold_selection(raw_preds, vid_features, reg_features)
        elif self.fsm_type == "nms":
            return self._nms_selection(raw_preds, vid_features, reg_features)
        else:
            raise ValueError(f"Unknown selection method: {self.fsm_type}")

    def _threshold_selection(
        self, raw_preds: torch.Tensor, vid_features: torch.Tensor, reg_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Select features using confidence threshold (YOLOV++ Thresh method).
        Maintains dense foreground predictions for better label assignment compatibility.

        Key differences from NMS method:
        - No NMS applied - keeps all detections above threshold
        - Variable output size per frame (typically <100 vs fixed 30 for NMS)
        - Better compatibility with dense label assignment strategies

        Args:
            raw_preds (torch.Tensor): Raw predictions from the model [batch_size, 4 + num_classes, num_anchors]
            vid_features (torch.Tensor): Video features [batch_size, num_anchors, cls_ch]
            reg_features (torch.Tensor): Regression features [batch_size, num_anchors, reg_ch]
        """
        predictions = raw_preds.permute(0, 2, 1)  # [batch, num_anchors, 4+nc]
        batch_size = raw_preds.shape[0]

        # Collect all selected features across the batch (variable length per frame)
        all_selected_cls_features: list[torch.Tensor] = []
        all_selected_reg_features: list[torch.Tensor] = []
        all_selected_boxes: list[torch.Tensor] = []
        all_selected_scores: list[torch.Tensor] = []
        batch_selected_predictions: list[torch.Tensor] = []
        batch_selected_indices: list[torch.Tensor] = []

        # Process each frame in the batch
        for b in range(batch_size):
            # Extract boxes and scores for this frame
            boxes = predictions[b, :, :4]  # [num_anchors, 4]
            scores = predictions[b, :, 4:]  # [num_anchors, num_classes]

            # Get max class confidence (equivalent to cls_score in YOLOV++)
            max_scores, _ = scores.max(dim=1)  # [num_anchors]

            # YOLOV++ uses obj_score * cls_score, but YOLO11 only has class scores
            # So we use max_class_score directly as confidence
            confidence_scores = max_scores

            # Apply confidence threshold (this is the core "Thresh" operation)
            thresh_mask = confidence_scores >= self.conf_thresh  # [num_anchors]
            thresh_indices = torch.where(thresh_mask)[0]  # [num_selected_anchors]

            # CRITICAL: Limit maximum selections per frame
            if len(thresh_indices) > self.max_thresh_proposals_per_frame:  # e.g., 100-150
                # Take top-k by confidence from thresh selections
                thresh_scores = max_scores[thresh_indices]
                _, topk_among_thresh = torch.topk(thresh_scores, self.max_thresh_proposals_per_frame)
                thresh_indices = thresh_indices[topk_among_thresh]

            # Key difference from our current implementation:
            # We don't force a fixed target_count - we keep all above threshold
            elif len(thresh_indices) == 0:
                # Handle edge case: no detections above threshold
                # Take the single highest scoring detection to ensure we have something
                _, highest_idx = confidence_scores.max(0)
                thresh_indices = highest_idx.unsqueeze(0)

            # Store selected features (variable length per frame, like YOLOV++)
            all_selected_cls_features.append(vid_features[b, thresh_indices])
            all_selected_reg_features.append(reg_features[b, thresh_indices])
            all_selected_boxes.append(boxes[thresh_indices])
            all_selected_scores.append(confidence_scores[thresh_indices])

            # Store for loss calculation
            batch_selected_predictions.append(predictions[b, thresh_indices])
            batch_selected_indices.append(thresh_indices)

        # Concatenate all features across frames (following YOLOV++ style)
        concat_cls_features = torch.cat(all_selected_cls_features, dim=0).unsqueeze(0)  # [1, total_detections, cls_ch]
        concat_reg_features = torch.cat(all_selected_reg_features, dim=0).unsqueeze(0)  # [1, total_detections, reg_ch]
        concat_boxes = torch.cat(all_selected_boxes, dim=0).unsqueeze(0)  # [1, total_detections, 4]
        concat_scores = torch.cat(all_selected_scores, dim=0).unsqueeze(0)  # [1, total_detections]

        return (
            concat_cls_features,  # [1, total_detections, cls_ch]
            concat_reg_features,  # [1, total_detections, reg_ch]
            concat_boxes,  # [1, total_detections, 4]
            concat_scores,  # [1, total_detections]
            batch_selected_predictions,  # len batch_size, [variable_count, 4+nc]
            batch_selected_indices,  # len batch_size, [variable_count]
        )

    def _nms_selection(
        self, raw_preds: torch.Tensor, vid_features: torch.Tensor, reg_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Select features using NMS, following YOLOV's concatenate-all approach.
        Returns variable length features that get concatenated across batch.

        Args:
            raw_preds (torch.Tensor): Raw predictions from the model [batch_size, 4 + num_classes, num_anchors]
            vid_features (torch.Tensor): Video features [batch_size, num_anchors, cls_ch]
            reg_features (torch.Tensor): Regression features [batch_size, num_anchors, reg_ch]
        """
        predictions = raw_preds.permute(0, 2, 1)  # [batch, num_anchors, 4+nc]
        batch_size = raw_preds.shape[0]
        device = raw_preds.device

        # Collect all selected features across the batch
        all_selected_cls_features = []
        all_selected_reg_features = []
        all_selected_boxes = []
        all_selected_scores = []
        batch_selected_predictions = []
        batch_selected_indices = []

        # extract boxes and scores for all frames
        boxes = predictions[:, :, :4]  # [batch_size, num_anchors, 4]
        scores = predictions[:, :, 4:]  # [batch_size, num_anchors, num_classes]
        max_scores, cls_ids = scores.max(dim=2)  # [batch_size, num_anchors]

        # Process each frame in the batch
        for b in range(batch_size):
            # Stage 1: Pre-selection by score
            if max_scores[b].shape[0] > self.topk_pre:
                topk_values, topk_indices = torch.topk(max_scores[b], self.topk_pre)
            else:
                topk_indices = torch.arange(max_scores[b].shape[0], device=device)
                topk_values = max_scores[b][topk_indices]

            # Stage 2: NMS
            nms_indices = batched_nms(
                boxes[b][topk_indices],
                topk_values,
                cls_ids[b][topk_indices],
                iou_threshold=self.nms_thresh_train if self.training else self.nms_thresh_val,
            )

            # Stage 3: Final selection (take up to topk_post)
            final_indices = topk_indices[nms_indices[: self.topk_post]]

            # Store selected features (variable length per frame)
            # From this batch, out of the 8400 anchor points, select up to topk_post (30) detections
            # Each anchor point makes a full prediction, so we select the best ones based on NMS
            all_selected_cls_features.append(vid_features[b, final_indices])
            all_selected_reg_features.append(reg_features[b, final_indices])
            all_selected_boxes.append(boxes[b, final_indices])
            all_selected_scores.append(max_scores[b, final_indices])

            batch_selected_predictions.append(predictions[b, final_indices])
            batch_selected_indices.append(final_indices)

        # Concatenate all features across frames (YOLOV style)
        concat_cls_features = torch.cat(all_selected_cls_features, dim=0).unsqueeze(0)
        concat_reg_features = torch.cat(all_selected_reg_features, dim=0).unsqueeze(0)
        concat_boxes = torch.cat(all_selected_boxes, dim=0).unsqueeze(0)
        concat_scores = torch.cat(all_selected_scores, dim=0).unsqueeze(0)

        return (
            concat_cls_features,  # [1, total_detections, cls_ch]
            concat_reg_features,  # [1, total_detections, reg_ch]
            concat_boxes,  # [1, total_detections, 4]
            concat_scores,  # [1, total_detections]
            batch_selected_predictions,  #  len batch_size, [up to topk_post, 4+nc]
            batch_selected_indices,  # len batch_size, [up to topk_post] (indices in the original raw_preds tensor)
        )
