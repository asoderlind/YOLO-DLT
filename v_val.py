from ultralytics import YOLO

model = YOLO(
    "runs/detect/waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-fixed-preload_vid_cls-full_freeze/weights/last.pt"
)

# WAYMO_TEMPORAL_BASE = "runs/detect/waymo_dark-yolo11n-temporal-base/weights/last.pt"
# model = YOLO(WAYMO_TEMPORAL_BASE)

model.val(
    # name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-0.25_attn_scale3",
    name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-fixed-preload_vid_cls-full_freeze-val-seed",
    data="waymo_dark.yaml",
    batch=1,
    gframe=16,
    dataset_type="temporal",
    device="cuda",
    seed=1,
)
