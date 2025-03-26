from ultralytics import YOLO

model = YOLO("runs/detect/bdd100k-yolo11l/weights/last.pt")

model.train(
    data="../ultralytics/cfg/datasets/bdd100k.yaml",
    epochs=200,
    resume=True,
    batch=16,
    device="cuda",
    use_fe=False,
)