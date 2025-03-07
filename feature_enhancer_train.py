from ultralytics import YOLO

model = YOLO("yolo11n.yaml")


model.train(
    data="../ultralytics/cfg/datasets/exDark128-yolo.yaml",
    epochs=1,
    batch=1,
    lr0=0.001,
    pretrained=True,
    optimizer="AdamW",
    device="mps",
    use_fe=True,
    augment=False,
    val=False,
    cls=0,
    box=0,
    dfl=0,
    lambda_c=0.5,
)
