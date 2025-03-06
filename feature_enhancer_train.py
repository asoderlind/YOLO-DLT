from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.train(
    data="../ultralytics/cfg/datasets/exDark-yolo.yaml",
    epochs=30,
    batch=4,
    pretrained=True,
    optimizer="Adam",
    device="mps",
    use_fe=False,
    augment=False,
)
