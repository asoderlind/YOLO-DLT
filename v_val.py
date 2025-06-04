from ultralytics import YOLO

model = YOLO(
    "runs/detect/waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-12.5_attn_scale-fixed/weights/last.pt"
)

model.val(
    name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-12.5_attn_scale-fixed-val",
    data="waymo_dark.yaml",
    batch=1,
    gframe=16,
    dataset_type="temporal",
    device="cuda",
    temporal_stride=2,
)
