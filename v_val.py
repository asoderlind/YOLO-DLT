from ultralytics import YOLO

model_name = "waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-fixed"

model = YOLO(f"runs/detect/{model_name}/weights/last.pt")

# WAYMO_TEMPORAL_BASE = "runs/detect/waymo_dark-yolo11n-temporal-base/weights/last.pt"
# model = YOLO(WAYMO_TEMPORAL_BASE)

model.val(
    # name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-0.25_attn_scale3",
    name=f"{model_name}-val-seed-1-log-run",
    data="waymo_dark.yaml",
    batch=1,
    gframe=16,
    # lframe=8,
    dataset_type="temporal",
    device="cuda",
    seed=1,
    # temporal_stride=2,
)


# for i in range(0, 3):
#     model.val(
#         # name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-0.25_attn_scale3",
#         name=f"{model_name}-val-seed-{i}",
#         data="waymo_dark.yaml",
#         batch=1,
#         gframe=16,
#         # lframe=8,
#         dataset_type="temporal",
#         device="cuda",
#         seed=i,
#         # temporal_stride=2,
#     )
