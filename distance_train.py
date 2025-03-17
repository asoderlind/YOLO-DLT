from ultralytics import YOLO


aug = True
model_path = "yolo11n.pt"
data_path = "coco8-dist.yaml"
device = "mps"
use_fe = False
use_dist = True
epochs = 10

# Load the model
model = YOLO(model_path)
results = model.train(
    data=data_path,
    batch=2,
    epochs=epochs,
    device=device,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.9,
    use_fe=use_fe,
    mosaic=1.0 if aug else 0.0,
    translate=0.1 if aug else 0.0,
    scale=0.5 if aug else 0.0,
    name=f"{data_path}-{model_path}-{epochs}e-{'dist' if use_dist else 'noDist'}-{'fe' if use_fe else 'noFe'}-{'aug' if aug else 'noAug'}",
    iou_type="ciou",
    use_dist=True,
)
