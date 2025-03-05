import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules import ECA, FEM, GC, SE, Conv, NewConv, SimSPPF


class TestGC:
    @pytest.mark.parametrize("shape", [(2, 64, 32, 32), (1, 32, 16, 16), (4, 128, 64, 64)])
    def test_output_shape(self, shape):
        """Test that GC block preserves input tensor dimensions."""
        x = torch.randn(*shape)
        gc = GC(shape[1])
        output = gc(x)
        assert output.shape == x.shape

    @pytest.mark.parametrize("channels", [3, 2, 1])
    def test_low_channel_case(self, channels):
        """Test behavior when input channels are less than 1/ratio."""
        x = torch.randn(2, channels, 32, 32)
        gc = GC(channels)
        output = gc(x)
        assert output.shape == x.shape
        assert gc.transform_channels_ == channels

    @pytest.mark.parametrize("batch_size,channels,spatial", [(2, 16, 8), (1, 32, 16), (4, 8, 4)])
    def test_weights_sum_to_one(self, batch_size, channels, spatial):
        """Test that attention weights from softmax sum to 1."""
        x = torch.randn(batch_size, channels, spatial, spatial)
        gc = GC(channels)

        weights = gc.channel_conv(x).view(batch_size, spatial * spatial, 1)
        print(weights.shape)
        weights = gc.softmax(weights)

        sums = weights.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), f"Sum of weights: {sums}"

    @pytest.mark.parametrize("channels,ratio,expected", [(64, 1.0 / 8.0, 8), (32, 1.0 / 16.0, 2), (128, 1.0 / 4.0, 32)])
    def test_custom_ratio(self, channels, ratio, expected):
        """Test that transform_channels is correctly calculated from ratio."""
        gc = GC(channels, ratio=ratio)
        assert gc.transform_channels_ == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self):
        """Test that the block works correctly when moved to GPU."""
        x = torch.randn(2, 64, 32, 32).cuda()
        gc = GC(64).cuda()
        output = gc(x)
        assert output.device == x.device


class TestNewConv:
    @pytest.mark.parametrize("shape", [(2, 64, 32, 32), (1, 32, 16, 16), (4, 128, 64, 64)])
    def test_output_shape(self, shape):
        """Test that NewConv block produces the correct output shape."""
        x = torch.randn(*shape)
        conv = NewConv(c1=shape[1], c2=128)
        output = conv(x)
        expected_shape = (shape[0], 128, shape[2] // 2, shape[3] // 2)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("batch_size,channels,spatial", [(2, 16, 8), (1, 32, 16), (4, 8, 4)])
    def test_spd_output_shape(self, batch_size, channels, spatial):
        """Test that the space-to-depth operation produces the correct output shape."""
        x = torch.randn(batch_size, channels, spatial, spatial)
        conv = NewConv(c1=channels, c2=128)
        spd_output = conv.spd(x)
        expected_shape = (batch_size, channels * 4, spatial // 2, spatial // 2)
        assert spd_output.shape == expected_shape

    @pytest.mark.parametrize("batch_size,channels,spatial", [(2, 16, 8), (1, 32, 16), (4, 8, 4)])
    def test_forward_output_shape(self, batch_size, channels, spatial):
        """Test that the forward pass produces the correct output shape."""
        x = torch.randn(batch_size, channels, spatial, spatial)
        conv = NewConv(c1=channels, c2=128)
        output = conv(x)
        expected_shape = (batch_size, 128, spatial // 2, spatial // 2)
        assert output.shape == expected_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self):
        """Test that the block works correctly when moved to GPU."""
        x = torch.randn(2, 64, 32, 32).cuda()
        conv = NewConv(c1=64, c2=128).cuda()
        output = conv(x)
        assert output.device == x.device


class TestSimSPPF:
    @pytest.mark.parametrize("shape", [(2, 64, 32, 32), (1, 32, 16, 16), (4, 128, 64, 64)])
    def test_output_shape(self, shape):
        """Test that SimSPPF preserves spatial dimensions and produces expected output channels."""
        x = torch.randn(*shape)
        sppf = SimSPPF(shape[1], 64)
        output = sppf(x)
        expected_shape = (shape[0], 64, shape[2], shape[3])
        assert output.shape == expected_shape

    @pytest.mark.parametrize("batch_size,channels,spatial", [(2, 16, 8), (1, 32, 16), (4, 8, 4)])
    def test_intermediate_shapes(self, batch_size, channels, spatial):
        """Test shapes of intermediate feature maps after pooling."""
        x = torch.randn(batch_size, channels, spatial, spatial)
        sppf = SimSPPF(channels, 64)

        # Get intermediate feature after first conv
        inter = sppf.cv1(x)
        expected_inter_channels = channels // 2
        assert inter.shape == (batch_size, expected_inter_channels, spatial, spatial)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self):
        """Test that SimSPPF works correctly when moved to GPU."""
        x = torch.randn(2, 64, 32, 32).cuda()
        sppf = SimSPPF(64, 128).cuda()
        output = sppf(x)
        assert output.device == x.device

    def test_activation_presence(self):
        """Test that ReLU activations are present in the module."""
        sppf = SimSPPF(64, 128)
        assert isinstance(sppf.cv1.act, torch.nn.ReLU)
        assert isinstance(sppf.cv2.act, torch.nn.ReLU)


class TestFEM:
    @pytest.mark.parametrize(
        "in_channels, out_channels, kernel_size",
        [
            (64, 64, 3),
            (128, 128, 3),
            (256, 256, 3),
            (32, 64, 3),
        ],
    )
    def test_fem_output_shape(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Test that FEM produces correct output shape."""
        fem = FEM(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

        x = torch.randn(2, in_channels, 32, 32)
        output = fem(x)

        expected_shape = (2, out_channels, 32, 32)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    def test_fem_components(self) -> None:
        """Test that FEM components are initialized correctly."""
        in_channels, out_channels = 64, 64
        fem = FEM(in_channels=in_channels, out_channels=out_channels)

        # Test that convolutions are properly initialized
        assert isinstance(fem.conv1, Conv)
        assert isinstance(fem.conv2, Conv)
        assert isinstance(fem.conv3, Conv)

        # Test dilation rates
        assert fem.conv1.conv.dilation == (1, 1)
        assert fem.conv2.conv.dilation == (3, 3)
        assert fem.conv3.conv.dilation == (5, 5)

        # Test activation functions
        assert isinstance(fem.conv1.act, nn.ReLU)
        assert isinstance(fem.conv2.act, nn.ReLU)
        assert isinstance(fem.conv3.act, nn.ReLU)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_fem_batch_processing(self, batch_size: int) -> None:
        """Test FEM with different batch sizes."""
        in_channels, out_channels = 64, 64
        fem = FEM(in_channels=in_channels, out_channels=out_channels)

        x = torch.randn(batch_size, in_channels, 32, 32)
        output = fem(x)

        assert output.shape[0] == batch_size

        # Check that output has valid values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fem_device_consistency(self) -> None:
        """Test FEM works correctly when moved to GPU."""
        fem = FEM(64, 64).cuda()
        x = torch.randn(2, 64, 32, 32).cuda()
        output = fem(x)

        assert output.device == x.device
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_fem_gradient_flow(self) -> None:
        """Test that gradients flow properly through FEM."""
        fem = FEM(64, 64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)

        output = fem(x)
        loss = output.sum()
        loss.backward()  # type: ignore[no-untyped-call]

        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestSE:
    """Tests for SE (Squeeze-and-Excitation) module."""

    @pytest.fixture
    def se_module(self):
        """Create SE module fixture."""
        return SE(in_channels=64, reduction=16)

    def test_init(self, se_module):
        """Test SE initialization."""
        assert isinstance(se_module.avg_pool, nn.AdaptiveAvgPool2d)
        assert isinstance(se_module.excitation, nn.Sequential)
        assert len(se_module.excitation) == 4
        assert isinstance(se_module.excitation[0], nn.Linear)
        assert isinstance(se_module.excitation[1], nn.ReLU)
        assert isinstance(se_module.excitation[2], nn.Linear)
        assert isinstance(se_module.excitation[3], nn.Sigmoid)

        # Check dimensions
        assert se_module.excitation[0].in_features == 64
        assert se_module.excitation[0].out_features == 4
        assert se_module.excitation[2].in_features == 4
        assert se_module.excitation[2].out_features == 64

    def test_forward_shape(self, se_module):
        """Test if output shape matches input shape."""
        batch_size = 4
        channels = 64
        height = 32
        width = 32

        x = torch.randn(batch_size, channels, height, width)
        output = se_module(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.device == x.device

    def test_channel_attention(self, se_module):
        """Test if SE applies channel-wise attention correctly."""
        x = torch.ones(2, 64, 16, 16)
        output = se_module(x)

        # Check that attention modifies each channel
        assert not torch.allclose(output[:, 0], output[:, 1])

        # Check output range (should be between 0 and input due to sigmoid)
        assert torch.all(output >= 0)
        assert torch.all(output <= x)

    @pytest.mark.parametrize("reduction", [4, 8, 16, 32])
    def test_reduction_ratio(self, reduction):
        """Test SE with different reduction ratios."""
        channels = 64
        se = SE(in_channels=channels, reduction=reduction)

        hidden_dim = channels // reduction
        assert se.excitation[0].out_features == hidden_dim
        assert se.excitation[2].in_features == hidden_dim

    def test_invalid_reduction(self):
        """Test if SE raises error for invalid reduction ratio."""
        with pytest.raises(ValueError, match="The number of input channels must be greater than the reduction factor."):
            SE(in_channels=16, reduction=32)  # Reduction larger than channels


class TestECA:
    """
    Test suite for the Efficient Channel Attention (ECA) module.
    """

    @pytest.fixture
    def sample_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fixture to generate sample inputs for testing.

        Returns:
            Tuple of tensors with different shapes and channel counts
        """
        # Small input: [batch_size=2, channels=16, height=24, width=24]
        x_small = torch.randn(2, 16, 24, 24)

        # Medium input: [batch_size=1, channels=64, height=32, width=32]
        x_medium = torch.randn(1, 64, 32, 32)

        # Large input: [batch_size=4, channels=256, height=56, width=56]
        x_large = torch.randn(4, 256, 56, 56)

        return x_small, x_medium, x_large

    def test_output_shape(self, sample_inputs):
        """
        Test that the output shape matches the input shape for various input sizes.
        """
        x_small, x_medium, x_large = sample_inputs

        # Test small input
        eca_small = ECA(channel=16)
        output_small = eca_small(x_small)
        assert output_small.shape == x_small.shape, f"Expected shape {x_small.shape}, got {output_small.shape}"

        # Test medium input
        eca_medium = ECA(channel=64)
        output_medium = eca_medium(x_medium)
        assert output_medium.shape == x_medium.shape, f"Expected shape {x_medium.shape}, got {output_medium.shape}"

        # Test large input
        eca_large = ECA(channel=256)
        output_large = eca_large(x_large)
        assert output_large.shape == x_large.shape, f"Expected shape {x_large.shape}, got {output_large.shape}"

    @pytest.mark.parametrize(
        "channels,gamma,b,expected_k_size",
        [
            (16, 2, 1, 3),  # Small channel count
            (64, 2, 1, 3),  # Medium channel count
            (256, 2, 1, 5),  # Large channel count
            (1024, 2, 1, 5),  # Very large channel count
        ],
    )
    def test_kernel_size_calculation(self, channels, gamma, b, expected_k_size):
        """Test that the kernel size is calculated correctly based on channel count."""
        eca = ECA(channel=channels, gamma=gamma, b=b)
        # Access the kernel size from the Conv1d layer
        actual_k_size = eca.conv.kernel_size[0]
        assert actual_k_size == expected_k_size, (
            f"For {channels} channels (gamma={gamma}, b={b}), "
            f"expected kernel size {expected_k_size}, got {actual_k_size}"
        )

    def test_gradient_flow(self):
        """
        Test that gradients flow properly through the ECA module.
        """
        # Create a simple input with requires_grad=True
        x = torch.randn(2, 32, 20, 20, requires_grad=True)
        eca = ECA(channel=32)
        output = eca(x)

        # Compute loss (e.g., mean of output)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None, "Input gradients not computed"

        # Check that all parameters have gradients
        for name, param in eca.named_parameters():
            assert param.grad is not None, f"Gradient not computed for parameter: {name}"
            # Check that gradients are not all zeros
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), (
                f"Gradient is all zeros for parameter: {name}"
            )

    def test_attention_values(self):
        """
        Test that the attention weights are properly bounded between 0 and 1.
        """
        x = torch.randn(2, 32, 20, 20)
        eca = ECA(channel=32)

        # Get the attention weights
        with torch.no_grad():
            # Replicate part of the forward pass to get attention weights
            y = eca.avg_pool(x)
            y = y.squeeze(-1)
            y = y.transpose(-1, -2)
            y = eca.conv(y)
            y = y.transpose(-1, -2)
            y = y.unsqueeze(-1)
            attention_weights = eca.sigmoid(y)

        # Check bounds
        assert torch.all(attention_weights >= 0), "Attention weights contain negative values"
        assert torch.all(attention_weights <= 1), "Attention weights exceed 1"
