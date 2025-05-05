from ultralytics import YOLO, settings
import torch
import os

# Update a setting
settings.update({"runs_dir": "/mnt/machine-learning-storage/ML1/ClusterOutput/MLC-499"})

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

"""
Regular set with distance loss disabled
"""
model1 = YOLO("dlt-models/yolo11n.yaml")
model1.train(
    data="ultralytics/cfg/datasets/kitti-cluster.yaml",
    epochs=1,
    pretrained=False,
    workers=1,
    device=device,
    use_fe=False,
    use_dist=False,
)
