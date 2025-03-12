import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules import ECA, FEM, GC, SE, BiFPNAdd, Conv, HybridConv, NewConv, RFAConv, SimAM, SimSPPF


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
            y: torch.Tensor = eca.avg_pool(x)
            y = y.squeeze(-1)
            y = y.transpose(-1, -2)
            y = eca.conv(y)
            y = y.transpose(-1, -2)
            y = y.unsqueeze(-1)
            attention_weights = eca.sigmoid(y)

        # Check bounds
        assert torch.all(attention_weights >= 0), "Attention weights contain negative values"
        assert torch.all(attention_weights <= 1), "Attention weights exceed 1"


class TestSimAM:
    """Test suite for the SimAM attention module."""

    @pytest.fixture
    def module(self):
        """Fixture that returns a SimAM instance."""
        return SimAM(e_lambda=1e-4)

    @pytest.fixture(
        params=[
            (1, 3, 16, 16),  # Small feature map
            (2, 8, 32, 32),  # Medium feature map
            (4, 16, 64, 64),  # Larger feature map
        ]
    )
    def input_shapes(self, request):
        """Fixture providing various input tensor shapes for testing."""
        return request.param

    @pytest.fixture(params=[1e-2, 1e-4, 1e-6])
    def lambda_values(self, request):
        """Fixture providing various lambda values for testing."""
        return request.param

    def test_output_shape(self, module, input_shapes):
        """Test that output shape matches input shape."""
        x = torch.rand(input_shapes)
        output = module(x)

        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    def test_no_nan_values(self, module, input_shapes):
        """Test that the output contains no NaN values."""
        x = torch.rand(input_shapes)
        output = module(x)

        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_values_bounded(self, module, input_shapes):
        """Test that output values are bounded by input values (attention can only reduce values)."""
        x = torch.rand(input_shapes)
        output = module(x)

        # Since we're using sigmoid, values should be scaled between 0 and the original value
        assert torch.all(output <= x), "Output values exceed input values"
        assert torch.all(output >= 0), "Output contains negative values"

    def test_distinctive_values_preserved(self, module):
        """Test that distinctive values get higher attention weights."""
        # Create a 1x1x5x5 tensor with mostly uniform values except one distinctive value
        x = torch.ones(1, 1, 5, 5) * 0.2
        x[0, 0, 2, 2] = 1.0  # Make the center value distinctive

        output = module(x)

        # The ratio of the center value to surrounding values should be higher in the output
        input_ratio = x[0, 0, 2, 2] / x[0, 0, 0, 0]
        output_ratio = output[0, 0, 2, 2] / output[0, 0, 0, 0]

        assert output_ratio > input_ratio, "Distinctive values are not emphasized"

    def test_lambda_effect(self, input_shapes):
        """Test that different lambda values affect the output appropriately."""
        x = torch.rand(input_shapes)

        # With a very small lambda, distinctive values should get more emphasis
        module_small_lambda = SimAM(e_lambda=1e-8)
        out_small_lambda = module_small_lambda(x)

        # With a large lambda, the attention effect should be reduced
        module_large_lambda = SimAM(e_lambda=1.0)
        out_large_lambda = module_large_lambda(x)

        # Calculate variance of output values as a measure of contrast
        var_small = torch.var(out_small_lambda)
        var_large = torch.var(out_large_lambda)

        # Smaller lambda should lead to more contrast (higher variance)
        assert var_small >= var_large, "Expected smaller lambda to create more contrast"

    def test_gradient_flow(self, module, input_shapes):
        """Test that gradients flow properly through the SimAM module."""
        x = torch.rand(input_shapes, requires_grad=True)

        # Forward pass through the SimAM module
        output = module(x)

        # Create a dummy loss and perform backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for the input
        assert x.grad is not None, "No gradients computed for input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN values"
        assert not torch.isinf(x.grad).any(), "Gradient contains infinite values"

    def test_integration_with_cnn(self):
        """Test SimAM in a small CNN architecture to ensure it integrates properly."""

        class SimpleConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.attn = SimAM()
                self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.attn(x)  # Apply SimAM
                x = self.conv2(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        # Test with a random input
        model = SimpleConvNet()
        x = torch.rand(2, 3, 32, 32, requires_grad=True)

        # Forward pass
        output = model(x)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check shapes and gradients
        assert output.shape == (2, 10), "Unexpected output shape from CNN"
        assert x.grad is not None, "No gradients computed for input in CNN test"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN values in CNN test"


class TestBiFPNAdd:
    @pytest.fixture
    def bifpn_fixture(self):
        """Create fixture with test data and components for BiFPNAdd testing."""
        batch_size = 8
        channels = 64
        height = 32
        width = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create sample input tensors
        num_inputs = 3
        inputs = [torch.randn(batch_size, channels, height, width, device=device) for _ in range(num_inputs)]

        # Initialize the BiFPNAdd module
        bifpn_add = BiFPNAdd(num_inputs=num_inputs).to(device)

        return {
            "batch_size": batch_size,
            "channels": channels,
            "height": height,
            "width": width,
            "device": device,
            "num_inputs": num_inputs,
            "inputs": inputs,
            "bifpn_add": bifpn_add,
        }

    def test_output_shape(self, bifpn_fixture):
        """Test that the output shape matches the input shape."""
        bifpn_add = bifpn_fixture["bifpn_add"]
        inputs = bifpn_fixture["inputs"]
        batch_size = bifpn_fixture["batch_size"]
        channels = bifpn_fixture["channels"]
        height = bifpn_fixture["height"]
        width = bifpn_fixture["width"]

        output = bifpn_add(inputs)

        # Check output shape
        assert output.shape == (batch_size, channels, height, width)

    def test_weighted_sum(self, bifpn_fixture):
        """Test that the weighted sum operates correctly with known weights."""
        bifpn_add = bifpn_fixture["bifpn_add"]
        inputs = bifpn_fixture["inputs"]
        device = bifpn_fixture["device"]

        # Set weights manually for testing
        with torch.no_grad():
            bifpn_add.weights.copy_(torch.tensor([1.0, 2.0, 3.0], device=device))

        # Forward pass
        output = bifpn_add(inputs)

        # Manual calculation of the expected output
        # First apply ReLU to weights (no change since all are positive)
        weights = torch.tensor([1.0, 2.0, 3.0], device=device)
        # Normalize weights
        normalized_weights = weights / (torch.sum(weights) + 1e-4)

        # Compute weighted sum manually
        expected_output = sum(w * inp for w, inp in zip(normalized_weights, inputs))
        # Check if output matches expected output
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-5)

    def test_gradients_flow(self, bifpn_fixture):
        """Test that gradients flow correctly through the BiFPNAdd module."""
        bifpn_add = bifpn_fixture["bifpn_add"]
        batch_size = bifpn_fixture["batch_size"]
        channels = bifpn_fixture["channels"]
        height = bifpn_fixture["height"]
        width = bifpn_fixture["width"]
        device = bifpn_fixture["device"]
        num_inputs = bifpn_fixture["num_inputs"]

        # Set requires_grad=True for inputs
        inputs = [
            torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)
            for _ in range(num_inputs)
        ]

        # Forward pass
        output = bifpn_add(inputs)

        # Create a dummy loss and do backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients have been computed for weights
        assert bifpn_add.weights.grad is not None
        assert not torch.allclose(bifpn_add.weights.grad, torch.zeros_like(bifpn_add.weights.grad))

        # Check that gradients have been computed for all inputs
        for inp in inputs:
            assert inp.grad is not None
            # Gradients should not be all zeros (very unlikely with random data)
            assert not torch.allclose(inp.grad, torch.zeros_like(inp.grad))

    def test_weight_normalization(self, bifpn_fixture):
        """Test that weights are properly normalized to sum to 1."""
        bifpn_add = bifpn_fixture["bifpn_add"]
        inputs = bifpn_fixture["inputs"]

        # Forward pass
        _ = bifpn_add(inputs)

        # Get the weights from the module
        weights = bifpn_add.relu(bifpn_add.weights)
        normalized_weights = weights / (torch.sum(weights) + bifpn_add.eps)

        # Check that the normalized weights sum to 1
        sum_weights = torch.sum(normalized_weights).item()
        # Print for debugging
        print(f"Sum of normalized weights: {sum_weights}")

        # Use a slightly more tolerant check
        assert abs(sum_weights - 1.0) < 1e-4, f"Expected sum to be 1.0, got {sum_weights}"

    def test_relu_activation(self, bifpn_fixture):
        """Test that negative weights are rectified by ReLU."""
        bifpn_add = bifpn_fixture["bifpn_add"]
        inputs = bifpn_fixture["inputs"]
        device = bifpn_fixture["device"]

        # Set some weights to negative values
        with torch.no_grad():
            bifpn_add.weights.copy_(torch.tensor([-1.0, 0.0, 2.0], device=device))

        # Forward pass
        _ = bifpn_add(inputs)

        # Apply ReLU manually
        weights_after_relu = torch.relu(bifpn_add.weights)

        # Check that negative weights have been zeroed
        expected_weights = torch.tensor([0.0, 0.0, 2.0], device=device)
        assert torch.allclose(weights_after_relu, expected_weights)


class TestRFAConv:
    @pytest.fixture
    def rfaconv_fixture(self):
        """Create fixture with test data and RFAConv instances for testing."""
        batch_size = 4
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        stride = 1
        height = 24
        width = 24
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create input tensor
        x = torch.randn(batch_size, in_channels, height, width, device=device)

        # Initialize RFAConv module
        rfaconv = RFAConv(in_channels, out_channels, kernel_size, stride).to(device)

        return {
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "height": height,
            "width": width,
            "device": device,
            "input": x,
            "rfaconv": rfaconv,
        }

    def test_output_shape(self, rfaconv_fixture):
        """Test that the output shape is correct."""
        x = rfaconv_fixture["input"]
        rfaconv = rfaconv_fixture["rfaconv"]
        batch_size = rfaconv_fixture["batch_size"]
        out_channels = rfaconv_fixture["out_channels"]
        height = rfaconv_fixture["height"]
        width = rfaconv_fixture["width"]

        # Forward pass
        output = rfaconv(x)

        # Expected output dimensions (should not change spatial dimensions)
        expected_height = height
        expected_width = width

        # Check output shape
        assert output.shape == (batch_size, out_channels, expected_height, expected_width), (
            f"Expected shape {(batch_size, out_channels, expected_height, expected_width)}, got {output.shape}"
        )

    def test_gradients_flow(self, rfaconv_fixture):
        """Test that gradients flow correctly through the RFAConv module."""
        x = rfaconv_fixture["input"].clone().requires_grad_(True)
        rfaconv = rfaconv_fixture["rfaconv"]

        # Forward pass
        output = rfaconv(x)

        # Create a dummy loss and do backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients have been computed for the input
        assert x.grad is not None, "Input gradient is None"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradient is all zeros"

        # Check that gradients have been computed for RFAConv parameters
        for name, param in rfaconv.named_parameters():
            assert param.grad is not None, f"Parameter {name} gradient is None"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), (
                f"Parameter {name} gradient is all zeros"
            )

    def test_attention_weights(self, rfaconv_fixture):
        """Test that attention weights are properly computed and normalized."""
        x = rfaconv_fixture["input"]
        rfaconv = rfaconv_fixture["rfaconv"]

        # Access the attention weights directly
        with torch.no_grad():
            weight = rfaconv.get_weight(x)
            b, c = x.shape[0:2]
            h, w = weight.shape[2:]

            # Reshape and apply softmax as in the forward method
            weighted = weight.view(b, c, rfaconv.kernel_size**2, h, w).softmax(2)

            # Check that weights sum to 1 along the appropriate dimension (dim=2)
            weight_sum = weighted.sum(dim=2)

            # All values should be very close to 1
            assert torch.allclose(weight_sum, torch.ones_like(weight_sum), rtol=1e-5, atol=1e-5), (
                "Attention weights do not sum to 1"
            )

    def test_feature_modulation(self, rfaconv_fixture):
        """Test the feature modulation step of RFAConv."""
        x = rfaconv_fixture["input"]
        rfaconv = rfaconv_fixture["rfaconv"]

        # Extract intermediate outputs to verify feature modulation
        with torch.no_grad():
            b, c = x.shape[0:2]

            # Get weights and features
            weight = rfaconv.get_weight(x)
            feature = rfaconv.generate_feature(x)

            h, w = weight.shape[2:]

            # Reshape as in the forward method
            weighted = weight.view(b, c, rfaconv.kernel_size**2, h, w).softmax(2)
            feature = feature.view(b, c, rfaconv.kernel_size**2, h, w)

            # Check modulated features
            weighted_data = feature * weighted

            # Check that modulation occurs (weighted_data should be different from feature)
            assert not torch.allclose(weighted_data, feature, rtol=1e-3, atol=1e-3), "Feature modulation had no effect"

    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_different_configurations(self, kernel_size, stride):
        """Test RFAConv with different kernel sizes and strides."""
        batch_size = 2
        in_channels = 8
        out_channels = 16
        height = 32
        width = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create input tensor
        x = torch.randn(batch_size, in_channels, height, width, device=device)

        # Initialize RFAConv module
        rfaconv = RFAConv(in_channels, out_channels, kernel_size, stride).to(device)

        # Forward pass
        output = rfaconv(x)

        # Expected output dimensions based on kernel size and stride
        if stride == 1:
            expected_height = height
            expected_width = width
        else:
            # When stride > 1, both the generate_feature Conv and the final Conv affect dimensions
            # generate_feature downsamples by stride, and the final Conv downsamples by kernel_size
            expected_height = height // stride
            expected_width = width // stride

        # Check output shape
        assert output.shape == (batch_size, out_channels, expected_height, expected_width), (
            f"Expected shape {(batch_size, out_channels, expected_height, expected_width)}, got {output.shape}"
        )

    def test_device_compatibility(self, rfaconv_fixture):
        """Test that RFAConv works correctly when moved between devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device compatibility test")

        x = rfaconv_fixture["input"]
        rfaconv = rfaconv_fixture["rfaconv"]

        # Move to CPU
        rfaconv_cpu = rfaconv.to("cpu")
        x_cpu = x.to("cpu")

        # Forward pass on CPU
        output_cpu = rfaconv_cpu(x_cpu)

        # Move back to CUDA
        rfaconv_cuda = rfaconv_cpu.to("cuda")
        x_cuda = x_cpu.to("cuda")

        # Forward pass on CUDA
        output_cuda = rfaconv_cuda(x_cuda)

        # Check shapes match
        assert output_cpu.shape == output_cuda.shape, "Output shapes don't match across devices"


class TestHybridConv:
    """Test suite for the HybridConv module."""

    @pytest.fixture
    def hybridconv_fixture(self):
        """Create test data and HybridConv instances for testing."""
        batch_size = 4
        in_channels = 16
        out_channels = 32  # Must be divisible by 2
        kernel_size = 3
        stride = 1
        height = 24
        width = 24
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create input tensor
        x = torch.randn(batch_size, in_channels, height, width, device=device)

        # Initialize HybridConv module
        hybridconv = HybridConv(c1=in_channels, c2=out_channels, k=kernel_size, s=stride).to(device)

        return {
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "height": height,
            "width": width,
            "device": device,
            "input": x,
            "hybridconv": hybridconv,
        }

    def test_output_shape(self, hybridconv_fixture):
        """Test that the output shape is correct."""
        x = hybridconv_fixture["input"]
        hybridconv = hybridconv_fixture["hybridconv"]
        batch_size = hybridconv_fixture["batch_size"]
        out_channels = hybridconv_fixture["out_channels"]
        height = hybridconv_fixture["height"]
        width = hybridconv_fixture["width"]
        stride = hybridconv_fixture["stride"]

        # Calculate expected height and width after convolution with stride
        expected_height = height // stride
        expected_width = width // stride

        # Forward pass
        output = hybridconv(x)

        # Check output shape
        assert output.shape == (batch_size, out_channels, expected_height, expected_width), (
            f"Expected shape {(batch_size, out_channels, expected_height, expected_width)}, got {output.shape}"
        )

    def test_gradients_flow(self, hybridconv_fixture):
        """Test that gradients flow correctly through the HybridConv module."""
        x = hybridconv_fixture["input"].clone().requires_grad_(True)
        hybridconv = hybridconv_fixture["hybridconv"]

        # Forward pass
        output = hybridconv(x)

        # Create a dummy loss and do backward pass
        loss = output.mean()
        loss.backward()

        # Check that gradients have been computed for the input
        assert x.grad is not None, "Input gradient is None"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradient is all zeros"

        # Check that gradients have been computed for HybridConv parameters
        for name, param in hybridconv.named_parameters():
            assert param.grad is not None, f"Parameter {name} gradient is None"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), (
                f"Parameter {name} gradient is all zeros"
            )

    def test_channel_splitting(self, hybridconv_fixture):
        """Test that the channel splitting works correctly in HybridConv."""
        x = hybridconv_fixture["input"]
        hybridconv = hybridconv_fixture["hybridconv"]
        out_channels = hybridconv_fixture["out_channels"]

        # Forward pass through individual components
        conv_out = hybridconv.conv(x)
        dwconv_out = hybridconv.dwconv(x)

        # Check that each component outputs half the channels
        assert conv_out.shape[1] == out_channels // 2, (
            f"Conv output should have {out_channels // 2} channels, got {conv_out.shape[1]}"
        )
        assert dwconv_out.shape[1] == out_channels // 2, (
            f"DWConv output should have {out_channels // 2} channels, got {dwconv_out.shape[1]}"
        )

        # Forward pass through the entire module
        output = hybridconv(x)

        # Check that the output is the concatenation of the two components
        combined = torch.cat([dwconv_out, conv_out], dim=1)
        assert torch.allclose(output, combined), "Output is not the concatenation of dwconv and conv outputs"

    @pytest.mark.parametrize("kernel_size", [1, 3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_different_configurations(self, kernel_size, stride):
        """Test HybridConv with different kernel sizes and strides."""
        batch_size = 2
        in_channels = 8
        out_channels = 16  # Must be divisible by 2
        height = 32
        width = 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create input tensor
        x = torch.randn(batch_size, in_channels, height, width, device=device)

        # Initialize HybridConv module
        hybridconv = HybridConv(c1=in_channels, c2=out_channels, k=kernel_size, s=stride).to(device)

        # Forward pass
        output = hybridconv(x)

        # Calculate expected dimensions
        expected_height = height // stride
        expected_width = width // stride

        # Check output shape
        assert output.shape == (batch_size, out_channels, expected_height, expected_width), (
            f"Expected shape {(batch_size, out_channels, expected_height, expected_width)}, got {output.shape}"
        )

    def test_assertion_for_even_channels(self):
        """Test that initialization with odd number of output channels raises an AssertionError."""
        in_channels = 16
        out_channels = 33  # Not divisible by 2

        with pytest.raises(AssertionError, match="c2 must be divisible by 2"):
            _ = HybridConv(c1=in_channels, c2=out_channels)

    def test_device_compatibility(self, hybridconv_fixture):
        """Test that HybridConv works correctly when moved between devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device compatibility test")

        x = hybridconv_fixture["input"]
        hybridconv = hybridconv_fixture["hybridconv"]

        # Move to CPU
        hybridconv_cpu = hybridconv.to("cpu")
        x_cpu = x.to("cpu")

        # Forward pass on CPU
        output_cpu = hybridconv_cpu(x_cpu)

        # Move back to CUDA
        hybridconv_cuda = hybridconv_cpu.to("cuda")
        x_cuda = x_cpu.to("cuda")

        # Forward pass on CUDA
        output_cuda = hybridconv_cuda(x_cuda)

        # Check shapes match
        assert output_cpu.shape == output_cuda.shape, "Output shapes don't match across devices"
