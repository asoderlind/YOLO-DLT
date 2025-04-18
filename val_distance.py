from ultralytics import YOLO
import torch

# PROD
model_path = "weights/kitti.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-noDontCare-d0.05_best.pt"

# LOCAL
# model_path = "./weights/kitti.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-noDontCare-d0.05_best.pt"

data_path = "kitti-mini.yaml"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO(model_path)
name = f"{data_path.replace('/', '-')}-{model_path.replace('/', '-')}_"
model.val(
    use_dist=True,
    data=data_path,
    device=device,
    name=name,
)
