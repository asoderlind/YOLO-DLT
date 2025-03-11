from ultralytics import YOLO

model = YOLO("yolo11n.yaml")

fe = True
epochs = 20
lc = 10.0

model.train(
    data="../ultralytics/cfg/datasets/exDark128-yolo.yaml",
    epochs=epochs,
    batch=2,
    pretrained=True,
    optimizer="auto",
    device="mps",
    use_fe=fe,
    augment=False,
    mosaic=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    fliplr=0.0,
    val=False,
    lambda_c=lc,
    name=f"exDark128-{'fe-' if fe else ''}{epochs}-allLoss-lc{lc}-Auto-noAug-preLoad",
)
