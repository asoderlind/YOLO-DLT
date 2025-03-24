from ultralytics import YOLO


# Defaults
model_path = "yolo11n.pt"
data_path = "kitti.yaml"
device = "cuda"
use_fe = False
epochs = 100
optimizer = "SGD"
scale = 0.0
mosaic = 1.0
use_dist = True
ds=[0.40, 0.50, 1.0, 1.5, 2.0, 5.0, 10.0] # up next

# model_path = "/home/phoawb/repos/YOLO-DLT/runs/detect/kitti.yaml-yolo11l.pt-200e-SGD-scale0.0-mosaic1.0-d0.35_14/weights/last.pt"
# model_path=f"{name}/weights/last.py"
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
        resume = False
    )
