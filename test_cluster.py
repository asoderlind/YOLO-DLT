from ultralytics import YOLO, settings
import torch
import os

# Update a setting
settings.update({"runs_dir": os.getenv("RUNS_DIR", "runs")})

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

"""
Regular set with distance loss disabled
"""
model1 = YOLO("dlt-models/yolo11n.yaml")
model1.train(
    data="ultralytics/cfg/datasets/bdd100k-mini-cluster.yaml", epochs=1, device=device, use_fe=False, use_dist=False
)
