from ultralytics import YOLO
import torch


# Defaults
model_path = "yolo11n.pt"
data_path = "kitti.yaml"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
use_fe = False
epochs = 100
optimizer = "SGD"
scale = 0.0
mosaic = 1.0
use_dist = True
ds = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

model_path = "./weights/kitti-dist.pt"
# model_path=f"{name}/weights/last.py"
model = YOLO(model_path)
for d in ds:
    model = YOLO(model_path)
    name = f"{data_path}-{model_path}-{epochs}e-{optimizer}-scale{scale}-mosaic{mosaic}-d{d}_"
    model.train(
        data=data_path,
        epochs=epochs,
        device=device,
        optimizer=optimizer,
        momentum=0.9,
        lr0=0.01,
        warmup_bias_lr=0.0,
        use_fe=use_fe,
        name=name,
        iou_type="ciou",
        mosaic=mosaic,
        scale=scale,
        use_dist=use_dist,
        dist=d,
        resume=False,
    )
