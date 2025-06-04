from ultralytics import YOLO

model = YOLO("runs/detect/waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-mosaic-fixed/weights/last.pt")

model.val(
    name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-mosaic-fixed-val",
    data="waymo_dark.yaml",
    batch=1,
    gframe=8,
    lframe=8,
    dataset_type="temporal",
    device="cuda",
    temporal_stride=2,
)
