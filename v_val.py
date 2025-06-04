from ultralytics import YOLO

model = YOLO(
    "runs/detect/waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-0.25_attn_scale3/weights/last.pt"
)

model.val(
    name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-0.25_attn_scale3",
    data="waymo_dark.yaml",
    batch=1,
    gframe=16,
    dataset_type="temporal",
    device="cuda",
)
