from ultralytics import YOLO

model = YOLO("runs/detect/waymo_dark-yolo11n-40_e-16_lframe-mosaic-fixed/weights/last.pt")

model.val(
    name="waymo_dark-yolo11n-40_e-16_lframe-mosaic-fixed-val",
    data="waymo_dark.yaml",
    batch=1,
    lframe=16,
    dataset_type="temporal",
    device="cuda",
    temporal_stride=2,
)
