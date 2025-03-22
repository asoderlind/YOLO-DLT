from ultralytics import YOLO


# Defaults
model_path = "yolo11n.pt"
data_path = "kitti.yaml"
device = "cuda"
use_fe = False
epochs = 100
optimizer="SGD"
noAug=False
use_dist = True
dists=[0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

#model_path=f"{name}/weights/last.py"

for d in dists:
    model = YOLO(model_path)
    name=f"{data_path}-{model_path}-{epochs}e-{optimizer}-{'noAug' if noAug else 'aug'}-{'dist' if use_dist else 'noDist'}-noAug-d{d}_"
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
        mosaic=0.0 if noAug else 1.0,
        scale=0.0 if noAug else 0.5,
        use_dist=use_dist,
        dist=d,
        # resume = True
    )
