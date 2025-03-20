from ultralytics import YOLO


# Defaults
model_path = "yolo11n.pt"
data_path = "kitti.yaml"
device = "cuda"
use_fe = False
epochs = 10

use_dists = [True]
dists = [0.1]

for d in dists:
    for use_dist in use_dists:
        model = YOLO(model_path)
        model.train(
            data=data_path,
            epochs=epochs,
            device=device,
            optimizer="SGD",
            momentum=0.9,
            lr0=0.01,
            warmup_bias_lr=0.0,
            use_fe=use_fe,
            name=f"{data_path}-{model_path}-{epochs}e-{'dist' if use_dist else 'noDist'}-d{d}_",
            iou_type="ciou",
            use_dist=use_dist,
            dist=d,
        )
