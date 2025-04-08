from ultralytics import YOLO

model = YOLO("dlt-models/yolo11n-EOVOD.yaml").load("yolo11n.pt")
# model.info(detailed=True)
model.train(data="waymo16.yaml", epochs=1, batch=4, device="mps", temporal_window=2)
