from ultralytics import YOLO


# Defaults
model_path = "yolo11n.pt"
model = YOLO(model_path)
data_path = "waymo-noConf.yaml"
device = "cuda"
use_fe = False
epochs = 200
opt = "SGD"


use_dists = [True, False]
augs = [True, False]
dists = [0.1, 0.5, 1.0, 2.0]

for d in dists:
    for use_dist in use_dists:
        for aug in augs:
            model.train(
                data=data_path,
                epochs=epochs,
                device=device,
                optimizer=opt,
                use_fe=use_fe,
                mosaic=1.0 if aug else 0.0,
                translate=0.1 if aug else 0.0,
                scale=0.5 if aug else 0.0,
                name=f"{data_path}-{model_path}-{epochs}e-{'dist' if use_dist else 'noDist'}-{'fe' if use_fe else 'noFe'}-{'aug' if aug else 'noAug'}-{opt}-d{d}",
                iou_type="ciou",
                use_dist=use_dist,
            )
