from ultralytics import YOLO


# Defaults
model_path = "yolo11n.pt"
model = YOLO(model_path)
data_path = "waymo128-noConfidence.yaml"
# data_path = "coco8-dist.yaml"
device = "mps"
use_fe = False
epochs = 1
opt = "auto"


use_dists = [True]
augs = [True]

for use_dist in use_dists:
    for aug in augs:
        model.train(
            data=data_path,
            batch=2,
            epochs=epochs,
            device=device,
            optimizer=opt,
            use_fe=use_fe,
            mosaic=1.0 if aug else 0.0,
            translate=0.1 if aug else 0.0,
            scale=0.5 if aug else 0.0,
            name=f"{data_path}-{model_path}-{epochs}e-{'dist' if use_dist else 'noDist'}-{'fe' if use_fe else 'noFe'}-{'aug' if aug else 'noAug'}-{opt}",
            iou_type="ciou",
            use_dist=use_dist,
        )
