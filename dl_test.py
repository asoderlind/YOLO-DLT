from ultralytics import YOLO

model = YOLO("dlt-models/yolo11n-temporal.yaml").load(
    "runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt"
)

model.train(data="waymo16.yaml", device="cuda", batch=4, temporal_freeze=True, epochs=1)
