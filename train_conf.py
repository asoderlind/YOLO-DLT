"""
Default values across all training configurations.
"""

import torch

MODEL = "dlt-models/yolo11n.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
OPTIMIZER = "SGD"
MOMENTUM = 0.9
BATCH = 16
IOU_TYPE = "ciou"
LR0 = 0.01
WARMUP_BIAS_LR = 0.0
PRETRAINED = False
EPOCHS = 200
CLUSTER_OUTPUT_PATH = "/mnt/machine-learning-storage/ML1/ClusterOutput/MLC-499/Runs"
CLUSTER_BDDK_WEIGHT_PATH = "/mnt/machine-learning-storage/ML1/ClusterOutput/MLC-499/Runs/bdd100k_night_cluster-yolo11n-bic-repc3k2-seed-0/weights/last.pt"
SEED = 0
MAX_DIST_WAYMO = 85
