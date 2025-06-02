from ultralytics import YOLO

model_bic_rfaconv = YOLO("runs/detect/bdd100k_night-yolo11n-bic-reduced-channel-rfac3k2/weights/last.pt")


model_bic_rfaconv.val(
    data="bdd100k_night.yaml",
    device="cuda",
    batch=1,
    conf=0.25,
)
