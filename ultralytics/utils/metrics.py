# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


IoUType = Literal[
    "iou",
    "giou",
    "diou",
    "ciou",
    "nwd",
    "wiou",
    "wiou1",
    "wiou2",
    "wiou3",
    "siou",
    "ciou+nwd",
    "eiou",
    "focal-eiou",
    "isiou",
    "thiou",
]


def bbox_iou(
    box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, iou_type: IoUType = "iou", eps: float = 1e-7, **kwargs
) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                              (x1, y1, x2, y2) format. Defaults to True.
        iou_type (IoUType, optional): The type of IoU to calculate. Defaults to "iou".
                                     Can be one of "iou", "giou", "diou", "ciou", "siou", "eiou", "isiou", "thiou" "wiou","wiouv1", "wiouv2", "wiouv3" or "nwd".
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        **kwargs: Additional parameters for specific IoU types, e.g., momentum, alpha, delta for WIoU, theta for SIoU, ratio for IS-IoU, threshold for ThIoU.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, CIoU, SIoU, WIoU, or NWD values depending on the specified type.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Special handling for NWD which operates directly on xyxy format
    iou_type = iou_type.lower()  # type: ignore[assignment]
    if iou_type == "nwd":
        # Reassemble boxes in xyxy format for NWD calculation
        boxes1 = torch.cat([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
        boxes2 = torch.cat([b2_x1, b2_y1, b2_x2, b2_y2], dim=-1)
        return calculate_nwd(boxes1, boxes2, eps)

    # For other metrics, calculate intersection and union as before
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # Standard IoU
    iou = inter / union

    # Handle WIoU versions
    if iou_type in ["wiou", "wiou1", "wiou2", "wiou3"]:
        version = 3  # Default to v3
        if iou_type == "wiou1":
            version = 1
        elif iou_type == "wiou2":
            version = 2
        elif iou_type == "wiou3":
            version = 3

        return calculate_wiou(
            iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps, version=version, **kwargs
        )

    # Return the appropriate IoU calculation based on the specified type
    elif iou_type == "iou":
        return iou
    elif iou_type == "giou":
        return calculate_giou(iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, union, eps)
    elif iou_type == "diou":
        return calculate_diou(iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps)
    elif iou_type == "ciou":
        return calculate_ciou(iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, w1, h1, w2, h2, eps)
    elif iou_type == "siou":
        theta = kwargs.get("theta", 4.0)  # Default value from the original implementation
        return calculate_siou(iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps, theta)
    # combination test from YOLO-EPF
    elif iou_type == "ciou+nwd":
        return 0.8 * calculate_ciou(
            iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, w1, h1, w2, h2, eps
        ) + 0.2 * calculate_nwd(
            torch.cat([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1), torch.cat([b2_x1, b2_y1, b2_x2, b2_y2], dim=-1), eps
        )
    elif iou_type == "eiou" or iou_type == "focal-eiou":
        return calculate_eiou(iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps)
    elif iou_type == "isiou":
        return calculate_isiou(b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, **kwargs)
    elif iou_type == "thiou":
        return calculate_thiou(iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, **kwargs)
    elif iou_type == "mpdiou":
        return calculate_mpdiou(iou, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2)
    else:
        valid_types = "iou, giou, diou, ciou, wiou, wiou1, wiou2, wiou3, nwd"
        raise ValueError(f"Invalid IoU type: {iou_type}. Must be one of {valid_types}.")


def calculate_mpdiou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the Minimum Penalty Distance IoU (MPDIoU)

    Args:
        iou: IoU values between boxes
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of first bounding box (predicted)
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of second bounding box (ground truth)
        eps: Small value to prevent division by zero

    Returns:
        MPDIoU values between boxes
    """
    # Calculate the squared distances between corners
    d_1_2 = torch.pow(b2_x1 - b1_x1, 2) + torch.pow(b2_y1 - b1_y1, 2)
    d_2_2 = torch.pow(b2_x2 - b1_x2, 2) + torch.pow(b2_y2 - b1_y2, 2)

    # Calculate squared height and width of ground truth box
    h_2_gt = torch.pow(b2_y1 - b2_y2, 2)
    w_2_gt = torch.pow(b2_x1 - b2_x2, 2)

    # Calculate the penalty terms
    penalty_term = d_1_2 / (h_2_gt + w_2_gt + eps) + d_2_2 / (h_2_gt + w_2_gt + eps)

    # Calculate MPDIoU
    mpdiou = iou - penalty_term

    return mpdiou


def calculate_thiou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    thresh: float = 0.01,
    eps: float = 1e-7,
) -> torch.Tensor:
    # Calculate the squared distances between corners
    d_1_2 = torch.pow(b2_x1 - b1_x1, 2) + torch.pow(b2_y1 - b1_y1, 2)
    d_2_2 = torch.pow(b2_x2 - b1_x2, 2) + torch.pow(b2_y2 - b1_y2, 2)

    # Calculate squared height and width of ground truth box
    h_2_gt = torch.pow(b2_y1 - b2_y2, 2)
    w_2_gt = torch.pow(b2_x1 - b2_x2, 2)

    # Maximum squared distance between corners
    D = torch.max(d_1_2, d_2_2)
    # Minimum squared dimension of ground truth box
    L = torch.min(h_2_gt, w_2_gt)

    # Calculate the penalty terms
    penalty_term = d_1_2 / (h_2_gt + w_2_gt + eps) + d_2_2 / (h_2_gt + w_2_gt + eps)

    # Calculate ThresIoU
    th_iou = iou - penalty_term

    # Create a mask for the threshold condition
    condition = (D / (L + eps)) > thresh

    # Apply the threshold condition element-wise
    result = torch.where(condition, th_iou, torch.ones_like(iou))

    return result


def calculate_isiou(
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    ratio: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate Inner-Shape IoU (ISIoU) metric.

    Args:
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of the first bounding box
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of the second bounding box
        ratio (float | optional): Ratio of the inner shape to the outer shape (default: 1.0)
        eps (float): Small value to avoid division by zero

    Returns:
        isiou (torch.Tensor): ISIoU value (higher is better, like IoU)
    """

    # Calc widths and heights
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1

    # Calc center points
    center1_x = (b1_x1 + b1_x2) / 2
    center1_y = (b1_y1 + b1_y2) / 2
    center2_x = (b2_x1 + b2_x2) / 2
    center2_y = (b2_y1 + b2_y2) / 2

    # boundaries for ground truth
    boundary_left_gt = center2_x - (w2 * ratio) / 2
    boundary_right_gt = center2_x + (w2 * ratio) / 2
    boundary_top_gt = center2_y - (h2 * ratio) / 2
    boundary_bottom_gt = center2_y + (h2 * ratio) / 2

    # boundaries for predicted
    boundary_left_pred = center1_x - (w1 * ratio) / 2
    boundary_right_pred = center1_x + (w1 * ratio) / 2
    boundary_top_pred = center1_y - (h1 * ratio) / 2
    boundary_bottom_pred = center1_y + (h1 * ratio) / 2

    # intersection area
    inter_width = torch.min(boundary_right_gt, boundary_right_pred) - torch.max(boundary_left_gt, boundary_left_pred)
    inter_height = torch.min(boundary_bottom_gt, boundary_bottom_pred) - torch.max(boundary_top_gt, boundary_top_pred)
    inter_area = inter_width * inter_height

    # union area (with epsilon to avoid division by zero)
    shape = (w2 * h2) * ratio**2 + (w1 * h1) * ratio**2 - inter_area + eps

    isiou = inter_area / shape
    return isiou


def calculate_eiou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate Enhanced IoU (EIoU) metric.

    Args:
        iou: Standard IoU value
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of the first bounding box
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of the second bounding box
        eps: Small value to avoid division by zero

    Returns:
        torch.Tensor: EIoU value (higher is better, like IoU)
    """
    # Calculate widths and heights
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1

    # Calculate center points
    center1_x = (b1_x1 + b1_x2) / 2
    center1_y = (b1_y1 + b1_y2) / 2
    center2_x = (b2_x1 + b2_x2) / 2
    center2_y = (b2_y1 + b2_y2) / 2

    # Center distance
    d_center_x = center1_x - center2_x
    d_center_y = center1_y - center2_y

    # Square center distance
    d_center_square = torch.square(d_center_x) + torch.square(d_center_y)

    # Calculate the smallest enclosing box
    wh_box_x = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    wh_box_y = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)

    # Square of enclosing box diagonal
    l2_box = torch.square(wh_box_x) + torch.square(wh_box_y)

    # Distance penalty: center distance divided by enclosing box diagonal
    dist_penalty = d_center_square / (l2_box + eps)

    # Width and height penalties
    w_penalty = torch.square(w1 - w2) / (torch.square(wh_box_x) + eps)
    h_penalty = torch.square(h1 - h2) / (torch.square(wh_box_y) + eps)

    # Final EIoU (as a similarity metric like IoU, not a loss)
    eiou = iou - dist_penalty - w_penalty - h_penalty

    return eiou


def calculate_siou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    eps: float = 1e-7,
    theta: float = 4.0,
) -> torch.Tensor:
    """
    Calculate Scylla IoU (SIoU) metric.

    Args:
        iou: Standard IoU value
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of the first bounding box
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of the second bounding box
        eps: Small value to avoid division by zero
        theta: Parameter for shape cost (default: 4.0)

    Returns:
        torch.Tensor: SIoU value
    """
    # Calculate center points and their distance
    center1_x = (b1_x1 + b1_x2) / 2
    center1_y = (b1_y1 + b1_y2) / 2
    center2_x = (b2_x1 + b2_x2) / 2
    center2_y = (b2_y1 + b2_y2) / 2

    # Calculate widths and heights
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1

    # Center distance
    d_center_x = center1_x - center2_x
    d_center_y = center1_y - center2_y

    # Square distance vector
    d_center_square = torch.square(d_center_x) + torch.square(d_center_y)

    # Calculate the smallest enclosing box
    wh_box_x = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    wh_box_y = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)

    # Angle Cost
    # Calculate the minimum absolute coordinate difference
    min_d_abs = torch.minimum(torch.abs(d_center_x), torch.abs(d_center_y))

    # Calculate arcsin(min_d_abs / sqrt(d_center_square))
    angle = torch.arcsin(min_d_abs / (torch.sqrt(d_center_square) + eps))
    angle = torch.sin(2 * angle) - 2

    # Distance Cost
    # Calculate normalized distances
    normalized_dx = torch.square(d_center_x / wh_box_x)
    normalized_dy = torch.square(d_center_y / wh_box_y)

    # Apply angle to distances
    dist_cost_x = torch.exp(angle * normalized_dx)
    dist_cost_y = torch.exp(angle * normalized_dy)
    dist_cost = 2 - dist_cost_x - dist_cost_y

    # Shape Cost
    d_w = torch.abs(w1 - w2)
    d_h = torch.abs(h1 - h2)
    big_w = torch.maximum(w1, w2)
    big_h = torch.maximum(h1, h2)

    w_shape = 1 - torch.exp(-d_w / big_w)
    h_shape = 1 - torch.exp(-d_h / big_h)

    shape_cost = torch.pow(w_shape, theta) + torch.pow(h_shape, theta)

    # Final SIoU
    siou = iou - (dist_cost + shape_cost) / 2

    return siou


def calculate_wiou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    eps: float = 1e-7,
    version: int = 3,
    iou_mean: torch.Tensor | None = None,
    alpha: float = 1.9,
    delta: float = 3.0,
) -> torch.Tensor:
    """
    Calculate Wise IoU (WIoU) metric.

    Args:
        iou: Standard IoU value
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of the first bounding box
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of the second bounding box
        eps: Small value to avoid division by zero
        version: WIoU version (1, 2, or 3)
        iou_mean: IoU mean value tensor (required for v2 and v3)
        alpha: Alpha parameter for v3 (controls the peak location)
        delta: Delta parameter for v3 (controls the width of the curve)

    Returns:
        torch.Tensor: WIoU value based on specified version
    """
    # Calculate center points and their distance
    center1_x = (b1_x1 + b1_x2) / 2
    center1_y = (b1_y1 + b1_y2) / 2
    center2_x = (b2_x1 + b2_x2) / 2
    center2_y = (b2_y1 + b2_y2) / 2

    center_distance_squared = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2

    # Calculate the smallest enclosing box dimensions
    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    c_area = cw * ch + eps

    # WIoU v1: Basic attention-based IoU
    # Uses distance between centers with normalization by enclosing box
    wiou = torch.exp(center_distance_squared / c_area.detach()) * iou

    # Handle different versions
    if version == 1:
        return wiou

    # For v2 and v3, we need the running average of IoU
    if iou_mean is None:
        raise ValueError("iou_mean is required for WIoU v2 and v3")

    # Calculate outlier degree (beta)
    beta = iou.detach() / iou_mean

    if version == 2:
        # WIoU v2: Monotonic focusing mechanism
        return (beta**0.5) * wiou

    elif version == 3:
        # WIoU v3: Dynamic non-monotonic focusing mechanism
        # Using the formula from the paper: β/(δ+α^(β-δ))
        r = beta / (delta * alpha ** (beta - delta))
        return r * wiou

    else:
        raise ValueError(f"Invalid WIoU version: {version}. Must be 1, 2, or 3.")


def calculate_nwd(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    constant: float = 12.0,
) -> torch.Tensor:
    """
    Calculate the Normalized Wasserstein Distance between sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): First set of boxes in xyxy format
        boxes2 (torch.Tensor): Second set of boxes in xyxy format
        eps (float): Small constant for numerical stability
        constant (float): Normalization constant (default: 12.0)

    Returns:
        torch.Tensor: Normalized Wasserstein Distance (similarity score between 0-1)
    """
    # Ensure we're working with the same number of boxes for 1-to-1 comparison
    if boxes1.shape != boxes2.shape:
        raise ValueError(
            f"boxes1 and boxes2 must have the same shape for 1-to-1 NWD calculation. "
            f"Got {boxes1.shape} and {boxes2.shape}"
        )

    # Calculate centers of bounding boxes (for 1-to-1 comparison)
    center1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    center2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    whs = center1 - center2

    # Calculate center distance term
    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

    # Calculate width and height of boxes
    w1 = boxes1[..., 2] - boxes1[..., 0] + eps
    h1 = boxes1[..., 3] - boxes1[..., 1] + eps
    w2 = boxes2[..., 2] - boxes2[..., 0] + eps
    h2 = boxes2[..., 3] - boxes2[..., 1] + eps

    # Calculate shape distance term (1-to-1)
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    # Calculate Wasserstein distance
    wasserstein = torch.sqrt(center_distance + wh_distance)

    # Normalize the distance
    nwd = torch.exp(-wasserstein / constant)

    # Return in the same format as other IoU functions (ensuring last dim is 1)
    if nwd.dim() > 0 and nwd.shape[-1] != 1:
        nwd = nwd.unsqueeze(-1)

    return nwd


def calculate_giou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    union: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Calculate Generalized IoU (GIoU) from the standard IoU.

    GIoU improves IoU by accounting for the area of the smallest enclosing box.
    Reference: https://arxiv.org/pdf/1902.09630.pdf

    Args:
        iou: Standard IoU value
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of the first bounding box
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of the second bounding box
        union: Union area of the two boxes
        eps: Small value to avoid division by zero

    Returns:
        torch.Tensor: GIoU value
    """
    # Calculate the convex hull (smallest enclosing box)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c_area = cw * ch + eps  # convex area

    # Calculate GIoU: IoU - (convex_area - union) / convex_area
    return iou - (c_area - union) / c_area


def calculate_diou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Calculate Distance IoU (DIoU) from the standard IoU.

    DIoU improves IoU by considering the distance between the centers of the boxes.
    Reference: https://arxiv.org/abs/1911.08287v1

    Args:
        iou: Standard IoU value
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of the first bounding box
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of the second bounding box
        eps: Small value to avoid division by zero

    Returns:
        torch.Tensor: DIoU value
    """
    # Calculate the convex hull (smallest enclosing box)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared

    # Calculate the squared distance between the centers of the boxes
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4  # center distance squared

    # Calculate DIoU: IoU - ρ²/c²
    return iou - rho2 / c2


def calculate_ciou(
    iou: torch.Tensor,
    b1_x1: torch.Tensor,
    b1_y1: torch.Tensor,
    b1_x2: torch.Tensor,
    b1_y2: torch.Tensor,
    b2_x1: torch.Tensor,
    b2_y1: torch.Tensor,
    b2_x2: torch.Tensor,
    b2_y2: torch.Tensor,
    w1: torch.Tensor,
    h1: torch.Tensor,
    w2: torch.Tensor,
    h2: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Calculate Complete IoU (CIoU) from the standard IoU.

    CIoU improves DIoU by also considering aspect ratio consistency.
    Reference: https://arxiv.org/abs/1911.08287v1

    Args:
        iou: Standard IoU value
        b1_x1, b1_y1, b1_x2, b1_y2: Coordinates of the first bounding box
        b2_x1, b2_y1, b2_x2, b2_y2: Coordinates of the second bounding box
        w1, h1: Width and height of the first box
        w2, h2: Width and height of the second box
        eps: Small value to avoid division by zero

    Returns:
        torch.Tensor: CIoU value
    """
    # Calculate the convex hull (smallest enclosing box)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared

    # Calculate the squared distance between the centers of the boxes
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4  # center distance squared

    # Calculate the consistency of aspect ratio
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)

    # Calculate the trade-off parameter alpha
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    # Calculate CIoU: IoU - (ρ²/c² + α·v)
    return iou - (rho2 / c2 + v * alpha)


def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Note:
        OBB format: [center_x, center_y, width, height, rotation_angle].
        If CIoU is True, returns CIoU instead of IoU.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps=0.1):
    """
    Computes smoothed positive and negative Binary Cross-Entropy targets.

    This function calculates positive and negative label smoothing BCE targets based on a given epsilon value.
    For implementation details, refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.

    Args:
        eps (float, optional): The epsilon value for label smoothing. Defaults to 0.1.

    Returns:
        (tuple): A tuple containing the positive and negative label smoothing BCE targets.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (Array[M]): The class labels.
        """
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # remove background class if task=detect

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn  # scope for faster 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names={}, on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names={}, xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
        ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
        unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
        p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
        r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
        f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
        x (np.ndarray): X-axis values for the curves. Shape: (1000,).
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.2, 1.0, 1.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DistMetrics(SimpleClass):
    """
    Utility class for computing distance metrics for YOLO.

    Attributes:
        e_A (list): Absolute distance error values.
        e_R (list): Relative distance error values.

    Methods:
        mean_results(): Returns [mean_absolute_error, mean_relative_error].
        update(results): Updates the metrics with a new results tuple.
        fitness(): Returns a fitness score based on the current error values.
    """

    def __init__(self) -> None:
        self.e_A: list[np.ndarray] = []  # list storing absolute distance errors (nc,)
        self.e_R: list[np.ndarray] = []  # list storing relative distance errors (nc,)
        self.e_min: list[np.ndarray] = []  # list storing min distance errors (nc,)
        self.e_mean: list[np.ndarray] = []  # list storing mean distance errors (nc,)
        self.e_max: list[np.ndarray] = []  # list storing max distance errors (nc,)
        self.e_std: list[np.ndarray] = []  # list storing std distance errors (nc,)

    @property
    def mean_absolute_error(self) -> float:
        """Mean absolute distance error."""
        return sum(self.e_A) / len(self.e_A) if len(self.e_A) else 0.0

    @property
    def mean_relative_error(self) -> float:
        """Mean relative distance error."""
        return sum(self.e_R) / len(self.e_R) if len(self.e_R) else 0.0

    @property
    def mean_min_error(self) -> float:
        """Mean min error."""
        return sum(self.e_min) / len(self.e_min) if len(self.e_min) else 0.0

    @property
    def mean_mean_error(self) -> float:
        """Mean mean error."""
        return sum(self.e_mean) / len(self.e_mean) if len(self.e_mean) else 0.0

    @property
    def mean_max_error(self) -> float:
        """Mean max error."""
        return sum(self.e_max) / len(self.e_max) if len(self.e_max) else 0.0

    @property
    def mean_std_error(self) -> float:
        """Mean std error."""
        return sum(self.e_std) / len(self.e_std) if len(self.e_std) else 0.0

    def mean_results(self):
        """Returns [mean_absolute_error, mean_relative_error]."""
        return [
            self.mean_absolute_error,
            self.mean_relative_error,
            self.mean_min_error,
            self.mean_mean_error,
            self.mean_max_error,
            self.mean_std_error,
        ]

    def class_result(self, i):
        """Returns [absolute_error, relative_error, min_error, mean_error, max_error, std_error]."""
        return self.e_A[i], self.e_R[i], self.e_min[i], self.e_mean[i], self.e_max[i], self.e_std[i]

    def update(self, results):
        """
        Updates the distance metrics.

        Args:
            results (tuple): A tuple containing:
                - e_A (list): List of absolute distance error values.
                - e_R (list): List of relative distance error values.
        """
        self.e_A, self.e_R, self.e_min, self.e_mean, self.e_max, self.e_std = results

    def fitness(self):
        """
        Computes a fitness score based on the distance errors.
        For example, lower errors yield higher fitness. Adjust the combination as needed.

        Returns:
            float: A fitness score.
        """
        # Here we define fitness such that lower errors yield higher fitness.
        total_error = self.mean_absolute_error + self.mean_relative_error
        return 1.0 / (1.0 + total_error)


def plot_distance_dependency(gt: np.ndarray, pred: np.ndarray, class_name: str, save_dir: Path):
    """
    Plot the distance errors against the distances and save the figure.

    Args:
        gt (list): Array of ground truth distances.
        pred (list): Array of predicted distances.
        class_name (str): Name of the class.
        save_dir (Path): Directory to save the plot.

    Returns:
        None
    """
    bins = np.arange(0, gt.max() + 5, 5)  # bins: [0, 5, 10, ..., max+5]
    bin_indices = np.digitize(gt, bins, right=False)

    mean_absolute_errors = []
    mean_relative_errors = []
    bin_centers = []

    for i in range(1, len(bins)):
        in_bin = bin_indices == i
        if np.any(in_bin):
            absolute_errors = np.abs(pred[in_bin] - gt[in_bin])
            relative_errors = absolute_errors / np.maximum(gt[in_bin], 1)

            mean_absolute_errors.append(absolute_errors.mean())
            mean_relative_errors.append(relative_errors.mean())

            bin_centers.append(bins[i])

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(bin_centers, mean_absolute_errors, width=4, edgecolor="#6C8EBF", color="#DAE8FC", linewidth=1)
    ax.set_xlabel("Groundtruth distance in meters")
    ax.set_ylabel(r"$\varepsilon_A$")
    ax.set_xticks(bin_centers)
    ax.tick_params(axis="x", labelrotation=90, length=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / f"distance_dependency_eA_{class_name}.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(bin_centers, mean_relative_errors, width=4, edgecolor="#B85450", color="#F8CECC", linewidth=1)
    ax.set_xlabel("Groundtruth distance in meters")
    ax.set_ylabel(r"$\varepsilon_R$")
    ax.set_xticks(bin_centers)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / f"distance_dependency_eR_{class_name}.png", dpi=300)
    plt.close()


def plot_distance_error_distribution(
    errors: np.ndarray, mean: np.floating, std: np.floating, class_name: str, save_dir: Path
):
    """
    Plot the distribution of distance errors along with a curve
    based on the mean and standard deviation and save the figure.

    Args:
        errors (list): List of distance errors.
        mean (float): Mean of the distance errors.
        std (float): Standard deviation of the distance errors.
        save_path (str): Path to save the plot.

    Returns:
        None
    """
    fig, ax1 = plt.subplots()

    save_path = save_dir / f"distance_error_distribution_{class_name}.png"

    # Plot histogram on left y-axis
    counts, bins, _ = ax1.hist(errors, bins=50, label=class_name, color="#6C8EBF")
    ax1.set_xlabel("Distance Error (m)")
    ax1.set_ylabel("Frequency", color="#6C8EBF")
    ax1.tick_params(axis="y", labelcolor="#6C8EBF")

    # Plot normal distribution on right y-axis
    ax2 = ax1.twinx()
    x = np.linspace(min(errors), max(errors), 1000)
    p = norm.pdf(x, mean, std)
    ax2.plot(x, p, "-", label="\u03bc=%.2f, \u03c3=%.2f" % (mean, std), color="#B85450")
    ax2.set_ylabel("Probability Density", color="#B85450")
    ax2.tick_params(axis="y", labelcolor="#B85450")
    ax2.set_ylim(bottom=0)

    # Add legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Distance Error Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def get_distance_errors_per_class(
    pred2gt_dist: np.ndarray,
    pred2gt_cls: np.ndarray,
    pred_dist: np.ndarray,
    target_cls: np.ndarray,
    nc: int,
    max_dist: int = 150,
    iou_level: int = 0,
    plot: bool = False,
    save_dir: Path = Path(),
    names={},
) -> tuple[list, list, list, list, list, list]:
    """
    Compute the mean absolute and relative distance errors for a set of predictions.

    Args:
        pred_dist: Predicted distance values.
        target_dist: Ground truth distance values.
        target_cls: Ground truth class labels.
        pred2gt: Mapping of predictions to ground truth indices for a specific IoU level.
        nc: Number of classes.
        max_dist: Maximum distance value for normalization.
        iou_level: IoU level for filtering predictions.
        plot: Whether to plot the distance error distribution.
        save_dir: Directory to save the plot.

    Returns:
        e_A: Mean absolute distance error for each class. (nc,)
        e_R: Mean relative distance error for each class. (nc,)
        e_min: Minimum distance error for each class. (nc,)
        e_mean: Mean distance error for each class. (nc,)
        e_max: Maximum distance error for each class. (nc,)
        e_std: Standard deviation of distance errors for each class. (nc,)
    """
    unique_classes, _ = np.unique(target_cls, return_counts=True)

    # map predictions to ground truth for a specific IoU level
    pred2gt_dist_iou = pred2gt_dist[:, iou_level]
    pred2gt_cls_iou = pred2gt_cls[:, iou_level]

    num_preds = len(pred_dist)
    dist_gt_cls = np.full((num_preds, 3), -1, dtype=np.float32)  # holding pred_dist, target_dist, target_cls
    for pred_id, (gt_dist, gt_cls) in enumerate(zip(pred2gt_dist_iou, pred2gt_cls_iou)):
        if gt_dist > 0:
            dist_gt_cls[pred_id] = (pred_dist[pred_id], gt_dist, gt_cls)

    e_A = [np.zeros((1, 1)) for _ in range(nc)]
    e_R = [np.zeros((1, 1)) for _ in range(nc)]
    e_min = [np.zeros((1, 1)) for _ in range(nc)]
    e_mean = [np.zeros((1, 1)) for _ in range(nc)]
    e_max = [np.zeros((1, 1)) for _ in range(nc)]
    e_std = [np.zeros((1, 1)) for _ in range(nc)]
    errors_per_class = [np.zeros((1, 1)) for _ in range(nc)]
    pred_per_class = [np.zeros((1, 1)) for _ in range(nc)]
    gt_per_class = [np.zeros((1, 1)) for _ in range(nc)]

    # de-normalize the distances
    dist_gt_cls[:, 0] = dist_gt_cls[:, 0] * max_dist
    dist_gt_cls[:, 1] = dist_gt_cls[:, 1] * max_dist

    for i in unique_classes:
        idx = int(i)
        # Filter out pairs of predictions and ground truth with a certain class
        pred_gt = dist_gt_cls[dist_gt_cls[:, 2] == idx]
        if len(pred_gt) > 0:
            pred_per_class[idx] = pred_gt[:, 0]
            gt_per_class[idx] = pred_gt[:, 1]
            errors_per_class[idx] = pred_gt[:, 0] - pred_gt[:, 1]
            _errors = pred_gt[:, 0] - pred_gt[:, 1]
            # Calculate min, mean, and max errors
            e_min[idx] = np.min(_errors)
            e_mean[idx] = np.mean(_errors)
            e_max[idx] = np.max(_errors)
            e_std[idx] = np.std(_errors)
            # Calculate mean absolute and relative errors
            absolute_errors = np.abs(_errors)
            relative_errors = absolute_errors / np.maximum(pred_gt[:, 1], 1)
            e_A[idx] = np.mean(absolute_errors)
            e_R[idx] = np.mean(relative_errors)

    # Class names
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict

    # Distance error distribution
    if plot:
        flat_errors = [e.ravel() for e in errors_per_class]
        errors_all_classes = np.concatenate(flat_errors)
        mean_all_classes = np.mean(errors_all_classes)
        std_all_classes = np.std(errors_all_classes)
        class_name = "All"
        plot_distance_error_distribution(errors_all_classes, mean_all_classes, std_all_classes, class_name, save_dir)
        for i, e in enumerate(errors_per_class):
            if len(e) > 0 and i in unique_classes:  # plot only for classes with data
                mean = np.mean(e)
                std = np.std(e)
                class_name = names.get(i, f"Class {i}")
                plot_distance_error_distribution(e, mean, std, class_name, save_dir)

    # e_A, e_R gt distance dependency plot
    if plot:
        for i, (pred, gt) in enumerate(zip(pred_per_class, gt_per_class)):
            if len(pred) > 0 and len(gt) > 0 and i in unique_classes:
                class_name = names.get(i, f"Class {i}")
                plot_distance_dependency(pred, gt, class_name, save_dir)

    return e_A, e_R, e_min, e_mean, e_max, e_std


class DetMetrics(SimpleClass):
    """
    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP) of an
    object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (dict of str): A dict of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (dict of str): A dict of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
        curves: TODO
        curves_results: TODO
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names={}, max_dist=150) -> None:
        """Initialize a DetMetrics instance with a save directory, plot flag, callback function, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.max_dist = max_dist
        self.box = Metric()
        self.dist = DistMetrics()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"

    def process(self, tp, conf, pred_cls, target_cls, pred_dist, pred2gt_dist, pred2gt_cls):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        nc = len(self.names)
        self.box.nc = nc
        self.box.update(results)
        results = get_distance_errors_per_class(
            pred2gt_dist,
            pred2gt_cls,
            pred_dist,
            target_cls,
            nc,
            max_dist=self.max_dist,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
        )
        self.dist.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/e_A(D)",
            "metrics/e_R(D)",
            "metrics/e_min(D)",
            "metrics/e_mean(D)",
            "metrics/e_max(D)",
            "metrics/e_std(D)",
        ]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95, mean absolute and relative distance errors."""
        return self.box.mean_results() + self.dist.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i) + self.dist.class_result(i)

    def distance_results(self):
        """Returns the mean absolute and relative distance errors."""
        return self.dist.mean_results()

    def distance_class_result(self, i):
        """Returns the mean absolute and relative distance errors for a specific class."""
        return self.dist.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results


class SegmentMetrics(SimpleClass):
    """
    Calculates and aggregates detection and segmentation metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize a SegmentMetrics instance with a save directory, plot flag, callback function, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.seg = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "segment"

    def process(self, tp, tp_m, conf, pred_cls, target_cls):
        """
        Processes the detection and segmentation metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp_m (list): List of True Positive masks.
            conf (list): List of confidence scores.
            pred_cls (list): List of predicted classes.
            target_cls (list): List of target classes.
        """
        results_mask = ap_per_class(
            tp_m,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns a list of keys for accessing metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self):
        """Return the mean metrics for bounding box and segmentation results."""
        return self.box.mean_results() + self.seg.mean_results()

    def class_result(self, i):
        """Returns classification results for a specified class index."""
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        """Returns mAP scores for object detection and semantic segmentation models."""
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        """Get the fitness score for both segmentation and bounding box models."""
        return self.seg.fitness() + self.box.fitness()

    @property
    def ap_class_index(self):
        """Boxes and masks have the same ap_class_index."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns results of object detection model for evaluation."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.seg.curves_results


class PoseMetrics(SegmentMetrics):
    """
    Calculates and aggregates detection and pose metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        pose (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pred_cls, target_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize the PoseMetrics class with directory path, class names, and plotting options."""
        super().__init__(save_dir, plot, names)
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.pose = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "pose"

    def process(self, tp, tp_p, conf, pred_cls, target_cls):
        """
        Processes the detection and pose metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp_p (list): List of True Positive keypoints.
            conf (list): List of confidence scores.
            pred_cls (list): List of predicted classes.
            target_cls (list): List of target classes.
        """
        results_pose = ap_per_class(
            tp_p,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        results_box = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns list of evaluation metric keys."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self):
        """Return the mean results of box and pose."""
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self, i):
        """Return the class-wise detection results for a specific class i."""
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        """Returns the mean average precision (mAP) per class for both box and pose detections."""
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        """Computes classification metrics and speed using the `targets` and `pred` inputs."""
        return self.pose.fitness() + self.box.fitness()

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.pose.curves_results


class ClassifyMetrics(SimpleClass):
    """
    Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (Dict[str, float]): A dictionary containing the time taken for each step in the pipeline.
        fitness (float): The fitness of the model, which is equal to top-5 accuracy.
        results_dict (Dict[str, Union[float, str]]): A dictionary containing the classification metrics and fitness.
        keys (List[str]): A list of keys for the results_dict.

    Methods:
        process(targets, pred): Processes the targets and predictions to compute classification metrics.
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "classify"

    def process(self, targets, pred):
        """Target classes and predicted classes."""
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self):
        """Returns mean of top-1 and top-5 accuracies as fitness score."""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property."""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []


class OBBMetrics(SimpleClass):
    """Metrics for evaluating oriented bounding box (OBB) detection, see https://arxiv.org/pdf/2106.06072.pdf."""

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize an OBBMetrics instance with directory, plotting, callback, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []
