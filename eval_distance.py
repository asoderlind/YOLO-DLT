from ultralytics import YOLO


# Dist

model_path = "runs/detect/waymo128-noConfidence.yaml-yolo11n.pt-1e-dist-noFe-aug-auto-d0.1/weights/best.pt"
model = YOLO(model_path)
data_path = "waymo-noConf.yaml"
# data_path = "coco8-dist.yaml"
device = "mps"
use_fe = False
epochs = 30
opt = "auto"


use_dists = [True]
augs = [True]

for use_dist in use_dists:
    for aug in augs:
        model.val(
            data=data_path,
            epochs=epochs,
            device=device,
            name=f"eval-{data_path}-{model_path}-{epochs}e-{'dist' if use_dist else 'noDist'}-{'fe' if use_fe else 'noFe'}-{'aug' if aug else 'noAug'}-{opt}",
            iou_type="ciou",
            use_fe=use_fe,
            use_dist=use_dist,
        )
