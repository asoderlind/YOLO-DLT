"""
Default values across all training configurations.
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
OPTIMIZER = "SGD"
MOMENTUM = 0.9
BATCH = 16
IOU_TYPE = "ciou"
LR0 = 0.01
WARMUP_BIAS_LR = 0.0
