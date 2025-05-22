# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import PixelShuffle

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class EfficientFocus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)
        self.channel_reducer = Conv(c1 * 4, c1, 1, 1)  # Add 1×1 conv to control channels

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        x = self.channel_reducer(
            torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
        )
        return self.conv(x)
        # return self.conv(self.contract(x))


class DWFocus(Focus):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__(c1=c1, c2=c2, k=k, s=s, act=act)
        self.conv = DWConv(c1 * 4, c2, k=k, s=s, act=act)


class GhostFocus(Focus):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__(c1=c1, c2=c2, k=k, s=s, act=act)
        self.conv = GhostConv(c1 * 4, c2, k=k, s=s, act=act)


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class ProgressiveFocus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.dwconv = DWConv(c1 * 4, c1 * 4, k=1, act=act)  # DWConv for spatial processing
        self.conv = Conv(c1 * 4, c2, k=1, s=1, act=act)  # Regular conv for channel

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(self.dwconv(x))


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]


class GC(nn.Module):
    def __init__(self, c1: int, ratio: float = 1.0 / 16.0):
        super().__init__()
        # context modeling
        self.channel_conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

        # transform
        self.transform_channels_ = c1 if not int(c1 * ratio) else int(c1 * ratio)
        self.transform = nn.Sequential(
            nn.Conv2d(c1, self.transform_channels_, kernel_size=1, bias=False),
            nn.LayerNorm([self.transform_channels_, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.transform_channels_, c1, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.shape

        # context modeling
        weights: torch.Tensor = self.channel_conv(x).view(batch, height * width, 1)  # B, 1, H, W => # B, H*W, 1
        weights = self.softmax(weights)  # B, H*W, 1
        context = torch.matmul(x.view(batch, channel, height * width), weights).view(batch, channel, 1, 1)  # B, C, 1, 1

        # transform and resiudal
        return x + self.transform(context)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


class NewConv(nn.Module):
    def __init__(self, c1, c2, g=1, d=1, act=True):
        super().__init__()
        self.c_ = c1 * 2  # hidden channels
        self.conv = Conv(c1, self.c_, k=3, s=2, p=1, g=g, d=d, act=act)
        self.channel_conv = nn.Conv2d(self.c_ + 4 * c1, c2, kernel_size=1, bias=False)

    def spd(self, x: torch.Tensor) -> torch.Tensor:  # (b,c,w,h) -> (b,4c,w/2,h/2)
        """space to depth operation according to https://arxiv.org/abs/2208.03641"""
        return torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_conv(torch.cat((self.conv(x), self.spd(x)), 1))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


class NewConv2(nn.Module):
    """Modified version of NewConv where we have a full SPD conv block instead of just the SPD operation."""

    def __init__(self, c1, c2, g=1, d=1, act=True):
        super().__init__()
        self.c_ = c1 * 2
        self.conv = Conv(c1, self.c_, k=3, s=2, p=1, g=g, d=d, act=act)
        self.spd_conv = Focus(c1, self.c_, k=3, s=1, p=1, g=g, act=act)
        self.channel_conv = Conv(2 * self.c_, c2, k=3, s=1, p=1, g=g, d=d, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spd = self.spd_conv(x)  # (b,2c,w/2,h/2)
        c = self.conv(x)  # (b,2c,w/2,h/2)
        res = torch.cat((c, spd), 1)  # (b,4c,w/2,h/2)
        return self.channel_conv(res)  # (b,c2,w/2,h/2)


class CBM(Conv):
    """
    Conv block with Mish activation.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, d: int = 1) -> None:
        super().__init__(c1, c2, k, s, p, g, d)
        self.act = nn.Mish()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class RFAConv(nn.Module):  # RFAConv implemented based on Group Conv
    """
    Receptive-Field Attention Convolution (RFA Conv) block.
    Adapted from: https://arxiv.org/abs/2304.03198
    Code from:  https://github.com/Liuchen1997/RFAConv/blob/main/model.py

    """

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size**2), kernel_size=1, groups=in_channel, bias=False),
        )

        self.generate_feature = Conv(
            in_channel,
            in_channel * (kernel_size**2),
            k=kernel_size,
            s=stride,
            g=in_channel,
            act=nn.ReLU(),
        )

        self.conv = Conv(
            in_channel,
            out_channel,
            k=kernel_size,
            s=kernel_size,
            act=nn.ReLU(),
        )

        self.pixel_shuffle = PixelShuffle(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RFAConv (Receptive-Field Attention Convolution).

        This method implements the core RFA algorithm which:
        1. Generates spatial attention weights for each position in the receptive field
        2. Extracts features from the input's receptive field
        3. Applies the attention weights to modulate these features
        4. Rearranges the weighted features into an expanded feature map
        5. Applies convolution to produce the final output

        The attention mechanism allows the network to focus on the most informative
        parts of each receptive field, enhancing feature representation quality.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

        Returns:
            out (torch.Tensor): Output tensor after applying receptive field attention
                          and convolution
        """
        b, c = x.shape[0:2]
        weight: torch.Tensor = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size**2, h, w).softmax(
            2
        )  # [b, c*kernel**2,h,w] ->  [b, c, k**2, h, w]
        feature: torch.Tensor = self.generate_feature(x)  # [b, c*kernel**2,h,w]
        feature = feature.view(
            b, c, self.kernel_size**2, h, w
        )  # [b, c*kernel**2,h,w] ->  [b, c, k**2, h, w]   obtain receptive field spatial features
        weighted_data = feature * weighted
        # conv_data = rearrange(
        #     weighted_data,
        #     "b c (n1 n2) h w -> b c (h n1) (w n2)",
        #     n1=self.kernel_size,  # [b, c, k**2, h, w] ->  [b, c, h*k, w*k]
        #     n2=self.kernel_size,
        # )  # [b, c, h*k, w*k]

        conv_data = self.pixel_shuffle(weighted_data.view(b, c * self.kernel_size**2, h, w))

        return self.conv(conv_data)  # [b, c, h*k, w*k] -> [b, c_out, h, w]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LDConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_param: int, stride: int = 1) -> None:
        super().__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = Conv(in_c, out_c, k=(num_param, 1), s=(num_param, 1), p=0)

        self.p_conv = nn.Conv2d(in_c, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        # Register buffers for both p_n and p_0 base coordinates
        self.register_buffer("p_n", None)
        self.register_buffer("p_0_base", None)
        self._initialized = False

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def _initialize_buffers(self, x: torch.Tensor, h: int, w: int):
        """Initialize buffers on first forward pass"""
        device = x.device
        dtype = x.dtype
        N = self.num_param

        # Initialize p_n
        self.p_n = self._get_p_n(N=N, device=device, dtype=dtype)

        # Pre-compute base coordinates that can be reused
        # Store normalized coordinates for grid_sample
        y_coords = torch.arange(0, h * self.stride, self.stride, device=device, dtype=dtype)
        x_coords = torch.arange(0, w * self.stride, self.stride, device=device, dtype=dtype)

        # Normalize to [-1, 1] for grid_sample
        y_coords = 2.0 * y_coords / (x.size(2) - 1) - 1.0
        x_coords = 2.0 * x_coords / (x.size(3) - 1) - 1.0

        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Shape: (1, h, w, 2)
        self.p_0_base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h, w = H // self.stride, W // self.stride

        # Initialize on first use
        if not self._initialized:
            self._initialize_buffers(x, h, w)

        # Get offsets
        offset = self.p_conv(x)  # (B, 2*num_param, h, w)

        # Use grid_sample for efficient bilinear interpolation
        # Reshape offset to (B, num_param, h, w, 2)
        offset = rearrange(offset, "b (n c) h w -> b n h w c", c=2)

        # Normalize offsets to [-1, 1] range
        offset_y = offset[..., 0:1] * 2.0 / (H - 1)
        offset_x = offset[..., 1:2] * 2.0 / (W - 1)
        offset_norm = torch.cat([offset_x, offset_y], dim=-1)

        # Expand p_0_base for batch size
        base_grid = self.p_0_base.expand(B, -1, -1, -1)  # (B, h, w, 2)

        # Reshape p_n for broadcasting
        p_n_reshaped = self.p_n.view(1, self.num_param, 1, 1, 2)
        p_n_norm = p_n_reshaped * 2.0 / torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype)

        # Create sampling grids for each offset
        # (B, num_param, h, w, 2)
        sampling_grids = base_grid.unsqueeze(1) + p_n_norm + offset_norm

        # Clamp to valid range
        sampling_grids = torch.clamp(sampling_grids, -1.0, 1.0)

        # Sample features using grid_sample
        # First, expand input for each sampling point
        x_expanded = x.unsqueeze(1).expand(-1, self.num_param, -1, -1, -1)

        # Reshape for grid_sample
        x_reshaped = rearrange(x_expanded, "b n c h w -> (b n) c h w")
        grid_reshaped = rearrange(sampling_grids, "b n h w c -> (b n) h w c")

        # Perform bilinear sampling
        sampled = F.grid_sample(x_reshaped, grid_reshaped, mode="bilinear", padding_mode="border", align_corners=True)

        # Reshape back
        sampled = rearrange(sampled, "(b n) c h w -> b c (n h) w", b=B, n=self.num_param)

        # Apply final convolution
        out = self.conv(sampled)

        return out

    def _get_p_n(self, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base_int = round(math.sqrt(N))
        row_number = N // base_int
        mod_number = N % base_int

        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number, device=device, dtype=dtype),
            torch.arange(0, base_int, device=device, dtype=dtype),
            indexing="ij",
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)

        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1, device=device, dtype=dtype),
                torch.arange(0, mod_number, device=device, dtype=dtype),
                indexing="ij",
            )
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))

        # Return as (N, 2) shape for easier manipulation
        p_n = torch.stack([p_n_x, p_n_y], dim=-1)
        return p_n


class HybridConv(nn.Module):
    """
    Hybric Conv (HConv) block implementation.
    Adapted from: https://www.mdpi.com/2072-4292/16/23/4493
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        assert c2 % 2 == 0, "c2 must be divisible by 2"

        half_c2 = c2 // 2

        self.conv = Conv(c1, half_c2, k=k, s=s, p=p, g=g, d=d, act=act)
        self.dwconv = DWConv(c1, half_c2, k=k, s=s, d=d, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies DwConv and Conv to input tensor and concatenates the output.
        DwConv and Conv outputs c2 / 2 channels each, which are concatenated to form the final output.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            out (torch.Tensor): Output tensor after applying Hybrid Convolution of shape [batch_size, out_channels, height2, width2]
        """
        return torch.cat([self.dwconv(x), self.conv(x)], dim=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, 2, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class BilateralSliceApply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grid, guide, input):
        """
        grid: (B, C_out, 2, D, H, W) - coefficients (weight and bias)
        guide: (B, C_in, H, W) - normalized feature values [0, 1]
        input: (B, C_in, H, W) - input feature map
        """
        B, C_out, _, D, Gh, Gw = grid.shape
        _, C_in, H, W = input.shape

        # Normalize guide to [-1, 1] for grid sampling
        guide = guide.unsqueeze(2)  # (B, C_in, 1, H, W)

        # Create normalized grid coordinates
        x_norm = torch.linspace(-1, 1, W, device=grid.device).view(1, 1, 1, W).expand(B, C_in, H, W)
        y_norm = torch.linspace(-1, 1, H, device=grid.device).view(1, 1, H, 1).expand(B, C_in, H, W)
        z_norm = (guide * 2 - 1).permute(0, 1, 3, 4, 2)  # (B, C_in, H, W, 1)

        # Combine coordinates (trilinear interpolation)
        sample_grid = torch.stack((x_norm, y_norm, z_norm.squeeze(-1)), dim=-1).unsqueeze(1)  # (B, 1, C_in, H, W, 3)

        # Sample grid coefficients for each input channel
        # We need to reshape grid to (B*C_out, 2, D, Gh, Gw) for batch processing
        grid_reshaped = grid.view(B * C_out, 2, D, Gh, Gw)
        sample_grid = sample_grid.expand(-1, C_out, -1, -1, -1, -1).reshape(B * C_out, C_in, H, W, 3)

        coefficients = F.grid_sample(
            grid_reshaped, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (B*C_out, 2, C_in, H, W)

        # Reshape back and split into weight and bias
        coefficients = coefficients.view(B, C_out, 2, C_in, H, W)
        weight = coefficients[:, :, 0]  # (B, C_out, C_in, H, W)
        bias = coefficients[:, :, 1]  # (B, C_out, C_in, H, W)

        # Apply: sum over input channels (input * weight) + bias
        input_expanded = input.unsqueeze(1)  # (B, 1, C_in, H, W)
        output = (input_expanded * weight).sum(dim=2) + bias.mean(dim=2)
        return output


class ResConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels * 5, 1),
            nn.BatchNorm2d(channels * 5),
            nn.PReLU(),
            nn.Conv2d(channels * 5, channels * 5, 3, padding=1, groups=channels * 5),
            nn.BatchNorm2d(channels * 5),
            nn.PReLU(),
            nn.Conv2d(channels * 5, channels, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return x + self.conv(x)


class MalleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample_factor=4, hidden_channels=64, num_blocks=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor

        # Downsample layer
        self.downsample = nn.AvgPool2d(downsample_factor)

        # Coefficient predictor network
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.PReLU(),
            *[ResConvBlock(hidden_channels) for _ in range(num_blocks)],
            nn.Conv2d(hidden_channels, 2 * out_channels * in_channels, 1),
        )

        # Coefficient applier
        self.apply_coeffs = BilateralSliceApply()

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        # 1. Downsample input
        x_down: torch.Tensor = self.downsample(x)  # (B, C_in, H//ds, W//ds)

        # 2. Predict coefficients grid (weight and bias per output channel)
        grid = self.predictor(x_down)  # (B, 2*C_out*C_in, H//ds, W//ds)
        grid = grid.view(
            B, self.out_channels, self.in_channels, 2, H // self.downsample_factor, W // self.downsample_factor
        )
        grid = grid.permute(0, 1, 3, 4, 5, 2)  # (B, C_out, 2, D, H, W, C_in)

        # 3. Apply coefficients to all channels simultaneously
        guide = torch.clamp(x, 0, 1)  # (B, C_in, H, W)
        output = self.apply_coeffs(grid, guide, x)  # (B, C_out, H, W)

        return self.act(self.bn(output))  # (B, C_out, H, W)


class CARAFEPlusPlusUpsample(nn.Module):
    def __init__(
        self,
        channels: int,
        compressed_channels: int = 64,
        kernel_size: int = 5,
        encoder_kernel_size: int = 3,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Channel compressor: 1x1 conv to reduce channels
        self.channel_compressor = nn.Conv2d(channels, compressed_channels, kernel_size=1, bias=False)

        # Content encoder: predicts kernel weights
        self.content_encoder = nn.Conv2d(
            compressed_channels,
            kernel_size**2 * scale_factor**2,  # Kernels for each upsampled position
            kernel_size=encoder_kernel_size,
            padding=encoder_kernel_size // 2,
            bias=False,
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.channel_compressor.weight)
        nn.init.normal_(self.content_encoder.weight, mean=0.0, std=0.001)

    def kernel_prediction(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_up = h * self.scale_factor
        w_up = w * self.scale_factor

        # 1. Predict reassembly kernels
        compressed = self.channel_compressor(x)  # [b, compressed_channels, h, w]
        kernels = self.content_encoder(compressed)  # [b, k²*scale_factor², h, w]

        # 2. Rearrange kernel weights to upsampled resolution
        kernels = F.pixel_shuffle(kernels, self.scale_factor)  # [b, k², h_up, w_up]

        # 3. Normalize kernels with softmax
        b, k2, h_up, w_up = kernels.shape
        kernels = kernels.view(b, k2, -1)  # [b, k², h_up*w_up]
        kernels = F.softmax(kernels, dim=1)
        kernels = kernels.view(b, k2, h_up, w_up)  # [b, k², h_up, w_up]

        return kernels

    def content_aware_reassembly(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_up = h * self.scale_factor
        w_up = w * self.scale_factor

        # Extract neighborhoods
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)  # [b, c*k*k, h*w]
        x_unfold = x_unfold.view(b, c, self.kernel_size**2, h * w)  # [b, c, k*k, h*w]

        # Prepare output container
        out = torch.zeros(b, c, h_up, w_up, device=x.device)

        # Apply reassembly - map each input position to scale_factor² output positions
        for i in range(h):
            for j in range(w):
                # Get input neighborhood
                idx_in = i * w + j
                neighborhood = x_unfold[:, :, :, idx_in]  # [b, c, k*k]

                # For each corresponding output position
                for di in range(self.scale_factor):
                    for dj in range(self.scale_factor):
                        i_out = i * self.scale_factor + di
                        j_out = j * self.scale_factor + dj

                        # Apply corresponding kernel
                        kernel = kernels[:, :, i_out, j_out].unsqueeze(1)  # [b, 1, k*k]
                        out[:, :, i_out, j_out] = (neighborhood * kernel).sum(dim=2)

        return out

    def efficient_content_aware_reassembly(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """
        Vectorized implementation without loops
        """
        b, c, h, w = x.shape
        h_up = h * self.scale_factor
        w_up = w * self.scale_factor

        # Extract neighborhoods
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)  # [b, c*k*k, h*w]
        x_unfold = x_unfold.view(b, c, self.kernel_size**2, h * w)  # [b, c, k*k, h*w]

        # Reshape for efficient computation
        # For each output position, we need the corresponding input neighborhood
        # and the corresponding kernel weights

        # Reshape x_unfold to prepare for broadcasting with kernels
        # We repeat each neighborhood scale_factor² times
        x_unfold = x_unfold.repeat_interleave(self.scale_factor**2, dim=3)  # [b, c, k*k, h*w*scale_factor²]

        # Reshape kernels to align with x_unfold
        kernels = kernels.view(b, self.kernel_size**2, -1)  # [b, k*k, h_up*w_up]

        # Apply kernels: multiply and sum over kernel dimension
        out = (x_unfold * kernels.unsqueeze(1)).sum(dim=2)  # [b, c, h_up*w_up]
        out = out.view(b, c, h_up, w_up)

        return out

    def forward(self, x):
        # 1. Predict reassembly kernels
        kernels = self.kernel_prediction(x)

        # 2. Apply content-aware reassembly
        # Using the vectorized implementation
        out = self.efficient_content_aware_reassembly(x, kernels)

        return out


class CARAFEPlusPlusDownsample(nn.Module):
    def __init__(
        self,
        channels: int,
        compressed_channels: int = 16,
        kernel_size: int = 5,
        encoder_kernel_size: int = 3,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Channel compressor: 1x1 conv to reduce channels
        self.channel_compressor = nn.Conv2d(channels, compressed_channels, kernel_size=1, bias=False)

        # Content encoder: 3x3 conv with stride to predict reassembly kernels at lower resolution
        self.content_encoder = nn.Conv2d(
            compressed_channels,
            kernel_size**2,
            kernel_size=encoder_kernel_size,
            stride=scale_factor,  # Key difference for downsampling
            padding=encoder_kernel_size // 2,
            bias=False,
        )

    def kernel_prediciton(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Predict reassembly kernels (at target resolution)
        compressed = self.channel_compressor(x)  # [b, compressed_channels, h, w]
        kernels: torch.Tensor = self.content_encoder(compressed)  # [b, k*k, h / scale, w / scale]

        b, _, h_down, w_down = kernels.shape
        # 2. Normalize kernels with softmax
        kernels = kernels.view(b, self.kernel_size * self.kernel_size, -1)  # [b, k*k, h_down*w_down]
        kernels = F.softmax(kernels, dim=1)
        kernels = kernels.view(b, self.kernel_size * self.kernel_size, h_down, w_down)  # [b, k*k, h_down, w_down]
        return kernels

    def content_aware_reassembly(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W]
            kernels (torch.Tensor): Reassembly kernels of shape [B, k*k, H_down, W_down]
        """
        b, c, h, w = x.shape
        h_down = h // self.scale_factor
        w_down = w // self.scale_factor
        c_r = self.kernel_size * self.kernel_size
        # 3. Extract neighborhoods and apply reassembly
        # Extract k×k neighborhoods around each pixel
        x_unfold = F.unfold(
            x, kernel_size=self.kernel_size, stride=self.scale_factor, padding=self.kernel_size // 2
        )  # [b, c*k*k, h_down*w_down]
        x_unfold = x_unfold.view(b, c, c_r, h_down * w_down)  # [b, c, k*k, h_down*w_down]

        # Reshape kernels for batch matrix multiplication
        kernels = kernels.view(b, c_r, h_down * w_down)  # [b, k*k, h_down*w_down]

        # Apply reassembly: weighted sum over the neighborhood
        # The dimensions work with broadcasting
        out = (x_unfold * kernels.unsqueeze(1)).sum(dim=2)  # [b, c, h_down*w_down]
        out = out.view(b, c, h_down, w_down)  # [b, c, h_down, w_down]
        return out

    def forward(self, x):
        # 1. Predict reassembly kernels
        kernels = self.kernel_prediciton(x)
        # Reassemble
        out = self.content_aware_reassembly(x, kernels)

        return out


class CARAFEConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, scale_factor: int = 2, compressed_channels: int = 16
    ) -> None:
        super().__init__()
        self.carafe = CARAFEPlusPlusDownsample(
            channels=in_channels,
            scale_factor=scale_factor,
        )
        self.conv = Conv(c1=in_channels, c2=out_channels, k=kernel_size, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CARAFEConv.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W]

        Returns:
            out (torch.Tensor): Output tensor of shape [B, C_out, H, W]
        """
        x = self.carafe(x)
        x = self.conv(x)
        return x
