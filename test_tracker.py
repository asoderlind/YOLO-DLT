from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.track(
    source="https://www.youtube.com/watch?v=Gr0HpDM8Ki8", save=True, show=True, device="mps", tracker="bytetrack.yaml"
)

# model.val_track(data="coco8.yaml", batch=1, device="mps")
