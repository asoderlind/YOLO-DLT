from ultralytics import YOLO


# Defaults
model_path = "yolo11n.pt"
data_path = "kitti.yaml"
device = "cuda"
use_fe = False
epochs = 200
optimizer = "SGD"
scale = 0.0
mosaic = 1.0
use_dist = True
d=0.35

# model_path=f"{name}/weights/last.py"

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
    # resume = True
)
