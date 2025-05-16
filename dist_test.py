from ultralytics import YOLO

model = YOLO("yolov11n.yaml")


model.train(
    data="waymo_night.yaml",
    epochs=1,
    imgsz=640,
    device="cuda",
    use_dist=True,
)

model.train(
    data="waymo_night.yaml",
    epochs=1,
    imgsz=640,
    device="cuda",
    use_dist=False,
)
