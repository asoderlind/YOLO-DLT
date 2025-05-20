from ultralytics import YOLO

# model = YOLO("dlt-models/yolo11n-temporal.yaml")  # .load(
#     "runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt"
# )
model = YOLO("dlt-models/yolo11n.yaml")
model.train(
    data="waymo_night.yaml",
    epochs=1,
    batch=1,
    device="mps",
    gframe=4,
    # temporal_freeze=True,
)

# model = YOLO("runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt")
# model2 = YOLO("yolo11n.pt")

# d1 = model.state_dict()
# d2 = model2.state_dict()

# model.info(verbose=True)
# model2.info(verbose=True)

# # compare the two state dicts
# # Define color codes
# RED = "\033[91m"
# YELLOW = "\033[93m"
# GREEN = "\033[92m"
# RESET = "\033[0m"

# for k in d1.keys():
#     if k not in d2:
#         print(f"{RED}[MISSING] {k} not in d2{RESET}")
#     else:
#         if d1[k].shape != d2[k].shape:
#             print(f"{YELLOW}[MISMATCH] {k} shape: {d1[k].shape} vs {d2[k].shape}{RESET}")
#         else:
#             print(f"{GREEN}[MATCH] {k} shape: {d1[k].shape}{RESET}")

# model.train(data="waymo16.yaml", device="cuda", batch=1, temporal_freeze=True, epochs=1)
