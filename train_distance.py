from ultralytics import YOLO


# Defaults
model_path = "yolo11n.pt"
data_path = "kitti.yaml"
device = "cuda"
use_fe = False
epochs = 100
optimizer = "SGD"
scales = [0.0, 0.25, 0.5]
mosaics = [0.0, 0.5, 1.0]
use_dist = True

# model_path=f"{name}/weights/last.py"

for scale in scales:
    for mosaic in mosaics:
        model = YOLO(model_path)
        name = f"{data_path}-{model_path}-{epochs}e-{optimizer}-scale{scale}-mosaic{mosaic}-d0.1_"
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
            dist=0.1,
            # resume = True
        )
