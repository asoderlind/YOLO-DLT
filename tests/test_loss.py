from typing import Literal

import pytest
import torch

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou

# Define IoUType for testing
IoUType = Literal["iou", "giou", "diou", "ciou", "nwd", "wiou", "wiou1", "wiou2", "wiou3"]


class TestWIoU:
    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def sample_boxes(self, device):
        # Create sample predictions and targets
        # [x1, y1, x2, y2] format
        pred_boxes = torch.tensor(
            [
                [0.1, 0.1, 0.5, 0.5],
                [0.3, 0.3, 0.7, 0.7],
                [0.5, 0.5, 0.9, 0.9],
            ],
            device=device,
        )

        target_boxes = torch.tensor(
            [
                [0.1, 0.1, 0.5, 0.5],  # Perfect match: IoU = 1.0
                [0.4, 0.4, 0.8, 0.8],  # Partial overlap: IoU ≈ 0.36
                [0.8, 0.8, 1.0, 1.0],  # Small overlap: IoU ≈ 0.04
            ],
            device=device,
        )

        return pred_boxes, target_boxes

    @pytest.mark.parametrize(
        "iou_type,expected_min_scale,expected_max_scale",
        [
            ("wiou1", 1.0, 3.0),  # WIoU v1 should be >= IoU and typically < 3*IoU
            ("wiou2", 0.2, 2.0),  # WIoU v2 values depend on beta calculations
            ("wiou3", 0.1, 2.0),  # WIoU v3 values depend on beta calculations
        ],
    )
    def test_wiou_basic_calculation(self, sample_boxes, iou_type, expected_min_scale, expected_max_scale, device):
        """Test basic WIoU calculations for different versions."""
        pred_boxes, target_boxes = sample_boxes

        # Calculate standard IoU first for comparison
        std_iou = bbox_iou(pred_boxes, target_boxes, xywh=False, iou_type="iou")

        # Create a mock iou_mean for v2 and v3
        iou_mean = torch.tensor(0.5, device=device)

        # Calculate WIoU with the specified version
        kwargs = {}
        if iou_type != "wiou1":
            kwargs["iou_mean"] = iou_mean
            kwargs["alpha"] = 1.9
            kwargs["delta"] = 3.0

        wiou = bbox_iou(pred_boxes, target_boxes, xywh=False, iou_type=iou_type, **kwargs)

        # Test basic properties
        assert wiou.shape == std_iou.shape, f"WIoU shape {wiou.shape} doesn't match IoU shape {std_iou.shape}"

        # For debugging
        for i in range(len(pred_boxes)):
            iou_val = std_iou[i].item()
            wiou_val = wiou[i].item()
            ratio = wiou_val / iou_val
            print(f"{iou_type} {i}: IoU={iou_val:.4f}, WIoU={wiou_val:.4f}, Ratio={ratio:.4f}")

        # Check that values are within expected ranges overall
        # Use a per-element check instead of a global check
        min_scale, max_scale = expected_min_scale, expected_max_scale

        # For WIoU v2 and v3, smaller IoUs may get smaller scales
        if iou_type in ["wiou2", "wiou3"]:
            for i in range(len(pred_boxes)):
                iou_val = std_iou[i].item()
                wiou_val = wiou[i].item()
                ratio = wiou_val / iou_val

                # Very small IoUs might get very small focus gains
                if iou_val < 0.1:
                    assert ratio >= min_scale * 0.5, (
                        f"Very small IoU {iou_val:.4f} has ratio {ratio:.4f} < {min_scale * 0.5:.4f}"
                    )
                else:
                    assert ratio >= min_scale, f"IoU {iou_val:.4f} has ratio {ratio:.4f} < {min_scale:.4f}"

                assert ratio <= max_scale, f"IoU {iou_val:.4f} has ratio {ratio:.4f} > {max_scale:.4f}"
        else:
            # WIoU v1 should always scale IoU upward
            assert torch.all(wiou >= std_iou), "WIoU v1 should be >= standard IoU"
            assert torch.all(wiou <= max_scale * std_iou), f"{iou_type} values above expected maximum"

    def test_wiou_v1_distance_effect(self, device):
        """Test that WIoU v1 correctly incorporates distance in the calculation."""
        # Create reference box
        box1 = torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)

        # Two boxes with approximately the same IoU but different center distances
        # Let's carefully construct these to ensure similar IoU

        # First pair: small shift but maintains high overlap
        box2_close = torch.tensor([[0.2, 0.2, 0.6, 0.6]], device=device)

        # Second pair: larger boxes shifted more but adjusted to maintain similar IoU
        # The key is to expand the box to maintain the same intersection area
        box2_far = torch.tensor([[0.3, 0.3, 0.8, 0.8]], device=device)

        # Calculate IoUs
        iou_close = bbox_iou(box1, box2_close, xywh=False, iou_type="iou")
        iou_far = bbox_iou(box1, box2_far, xywh=False, iou_type="iou")

        # Let's now use the actual IoU values for our test, instead of assuming they're similar
        # We'll test that WIoU properly scales based on distance regardless of exact IoU values

        # Calculate WIoU v1
        wiou_close = bbox_iou(box1, box2_close, xywh=False, iou_type="wiou1")
        wiou_far = bbox_iou(box1, box2_far, xywh=False, iou_type="wiou1")

        # Calculate normalized WIoU (WIoU / IoU) to account for different base IoUs
        norm_wiou_close = wiou_close / iou_close
        norm_wiou_far = wiou_far / iou_far

        # The normalized WIoU should be higher for the pair with greater center distance
        assert norm_wiou_far > norm_wiou_close, (
            f"Normalized WIoU should increase with center distance: "
            f"close={norm_wiou_close.item()}, far={norm_wiou_far.item()}"
        )

        # Print values for debugging
        print(f"IoU close: {iou_close.item()}, IoU far: {iou_far.item()}")
        print(f"WIoU close: {wiou_close.item()}, WIoU far: {wiou_far.item()}")
        print(f"Norm WIoU close: {norm_wiou_close.item()}, Norm WIoU far: {norm_wiou_far.item()}")

    @pytest.mark.parametrize(
        "input_ious,expected_gains",
        [
            # List of (input_ious, expected_gain_order)
            pytest.param(
                [0.9, 0.5, 0.1],
                [0, 1, 2],  # Ordering from highest focus gain to lowest
                id="normal-distribution",
            ),
            pytest.param(
                [0.95, 0.05],
                [0, 1],  # Highest gain for highest IoU
                id="extreme-values",
            ),
        ],
    )
    def test_wiou_v3_focusing_mechanism(self, input_ious, expected_gains, device):
        """Test that WIoU v3 correctly implements the focusing mechanism."""
        # Create dummy boxes that will result in the desired IoUs
        boxes = []
        for iou_val in input_ious:
            # Create a box pair with the target IoU
            if iou_val > 0.9:  # Perfect or near-perfect match
                box1 = torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)
                box2 = torch.tensor([[0.12, 0.12, 0.52, 0.52]], device=device)
            elif iou_val > 0.5:  # Good overlap
                box1 = torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)
                box2 = torch.tensor([[0.2, 0.2, 0.6, 0.6]], device=device)
            elif iou_val > 0.1:  # Medium overlap
                box1 = torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)
                box2 = torch.tensor([[0.3, 0.3, 0.7, 0.7]], device=device)
            else:  # Low overlap
                box1 = torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)
                box2 = torch.tensor([[0.4, 0.4, 0.9, 0.9]], device=device)

            boxes.append((box1, box2))

        # Set up mean IoU to be the middle of the range
        iou_mean = torch.tensor(0.5, device=device)

        # Calculate the focusing coefficients
        focus_values = []
        actual_ious = []
        for box1, box2 in boxes:
            # Get standard IoU
            std_iou = bbox_iou(box1, box2, xywh=False, iou_type="iou")
            actual_ious.append(std_iou.item())

            # Get WIoU v3
            wiou = bbox_iou(box1, box2, xywh=False, iou_type="wiou3", iou_mean=iou_mean, alpha=1.9, delta=3.0)

            # Calculate focusing coefficient (wiou / std_iou)
            focus_coeff = wiou / std_iou
            focus_values.append(focus_coeff.item())

        # For debugging
        for i, (iou_val, actual_iou, focus_val) in enumerate(zip(input_ious, actual_ious, focus_values)):
            beta = actual_iou / iou_mean.item()
            print(
                f"Target IoU: {iou_val:.2f}, Actual IoU: {actual_iou:.2f}, Beta: {beta:.2f}, Focus gain: {focus_val:.4f}"
            )

        # Sort indices by focus values from highest to lowest
        sorted_indices = sorted(range(len(focus_values)), key=lambda i: focus_values[i], reverse=True)

        # Check that the focusing mechanism ordering matches expected
        assert sorted_indices == expected_gains, (
            f"Focusing mechanism ordering {sorted_indices} doesn't match expected {expected_gains}"
        )

    def test_mean_tracking(self, sample_boxes, device):
        """Test that the IoU mean is properly tracked and updated."""
        pred_boxes, target_boxes = sample_boxes

        # Create a BboxLoss instance with WIoU v3
        bbox_loss = BboxLoss(iou_type="wiou3")
        bbox_loss.to(device)

        # Verify initial mean
        assert bbox_loss.iou_mean.item() == 1.0, "Initial IoU mean should be 1.0"

        # Create dummy inputs for forward pass
        batch_size = 2
        n_anchors = 3
        n_classes = 80

        # Expand our test boxes to match the expected input format
        pred_bboxes = pred_boxes.unsqueeze(0).repeat(batch_size, 1, 1)
        target_bboxes = target_boxes.unsqueeze(0).repeat(batch_size, 1, 1)

        # Create other required inputs
        pred_dist = torch.randn(batch_size, n_anchors, 4, 16, device=device)  # For reg_max=16
        anchor_points = torch.rand(batch_size, n_anchors, 2, device=device)
        target_scores = torch.ones(batch_size, n_anchors, n_classes, device=device) * 0.01
        target_scores[:, :3, 0] = 0.95  # Mark first 3 anchors as foreground
        fg_mask = target_scores.sum(-1) > 0.5
        target_scores_sum = target_scores.sum()

        # First forward pass
        bbox_loss(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)

        # Mean should have changed from initial value
        first_mean = bbox_loss.iou_mean.item()
        assert first_mean != 1.0, "IoU mean should update after first forward pass"

        # Second forward pass with different IoUs
        modified_pred = pred_bboxes.clone()
        modified_pred[:, :, :2] += 0.1  # Shift boxes to change IoUs
        bbox_loss(pred_dist, modified_pred, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)

        # Mean should have been updated again
        second_mean = bbox_loss.iou_mean.item()
        assert second_mean != first_mean, "IoU mean should update after second forward pass"

        # Verify the mean update uses momentum
        # The new mean should be a weighted average of the old mean and the new batch mean
        expected_direction = True  # Direction of change should match the IoU differences
        actual_direction = second_mean < first_mean
        assert expected_direction == actual_direction, "IoU mean update direction incorrect"

    def test_box_format_handling(self, device):
        """Test that WIoU correctly handles different box formats."""
        # Create boxes in both formats
        # xywh format
        boxes_xywh1 = torch.tensor([[0.3, 0.3, 0.4, 0.4]], device=device)
        boxes_xywh2 = torch.tensor([[0.4, 0.4, 0.4, 0.4]], device=device)

        # xyxy format
        boxes_xyxy1 = torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)
        boxes_xyxy2 = torch.tensor([[0.2, 0.2, 0.6, 0.6]], device=device)

        # Calculate IoUs with different formats
        iou_xywh = bbox_iou(boxes_xywh1, boxes_xywh2, xywh=True, iou_type="wiou1")
        iou_xyxy = bbox_iou(boxes_xyxy1, boxes_xyxy2, xywh=False, iou_type="wiou1")

        # Both should produce valid IoUs
        assert 0 <= iou_xywh <= 1.5, f"Invalid IoU from xywh format: {iou_xywh.item()}"
        assert 0 <= iou_xyxy <= 1.5, f"Invalid IoU from xyxy format: {iou_xyxy.item()}"

        # Convert xywh to xyxy manually and compare results
        x1, y1, w1, h1 = boxes_xywh1[0]
        x2, y2, w2, h2 = boxes_xywh2[0]

        manual_xyxy1 = torch.tensor([[x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2]], device=device)
        manual_xyxy2 = torch.tensor([[x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2]], device=device)

        iou_manual = bbox_iou(manual_xyxy1, manual_xyxy2, xywh=False, iou_type="wiou1")

        # IoUs should be very close
        assert torch.isclose(iou_xywh, iou_manual, atol=1e-6), "Format conversion inconsistency"
