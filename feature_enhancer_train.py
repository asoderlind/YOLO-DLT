from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.train(
    data="../ultralytics/cfg/datasets/coco8-dark.yaml",
    epochs=1,
    batch=1,
    pretrained=False,
    device="mps",
    use_fe=True,
)
