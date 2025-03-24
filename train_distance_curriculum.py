from ultralytics import YOLO
import torch
from typing import Any

# Defaults
model_path = "yolo11n.pt"
data_path = "kitti-mini.yaml"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
optimizer = "SGD"
scale = 0.0
mosaic = 1.0
# model_path=f"{name}/weights/last.py"
curriculum: list[dict[str, Any]] = [
    {
        "epochs": 150,
        "f": 0,
        "use_dist": False,
        "box": 7.5,
        "dfl": 0.5,
        "cls": 1.5,
        "d": 0,
    },
    {"epochs": 50, "f": 23, "use_dist": True, "box": 0, "dfl": 0, "cls": 0, "d": 0.35},
]

model = YOLO(model_path)
for cur in curriculum:
    epochs = cur["epochs"]
    f = cur["f"]
    use_dist = cur["use_dist"]
    box = cur["box"]
    dfl = cur["dfl"]
    cls = cur["cls"]
    d = cur["d"]
    name = f"{data_path}-{model_path}-curriculum-{epochs}e-{optimizer}-scale{scale}-mosaic{mosaic}-f{f}-d{d}_"
    model.train(
        data=data_path,
        epochs=epochs,
        device=device,
        batch=16,
        optimizer=optimizer,
        momentum=0.9,
        lr0=0.01,
        warmup_bias_lr=0.0,
        name=name,
        iou_type="ciou",
        mosaic=mosaic,
        scale=scale,
        use_dist=use_dist,
        exist_ok=True,
        dist=d,
        box=box,
        dfl=dfl,
        cls=cls,
        freeze=f,
    )
