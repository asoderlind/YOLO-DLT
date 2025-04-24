from ultralytics import YOLO

model = YOLO("runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt")
model2 = YOLO("yolo11n.pt")

d1 = model.state_dict()
d2 = model2.state_dict()

model.info(verbose=True)
model2.info(verbose=True)

# compare the two state dicts
for k in d1.keys():
    if k not in d2:
        print(f"{k} not in d2")
    else:
        if d1[k].shape != d2[k].shape:
            print(f"{k} shape mismatch: {d1[k].shape} vs {d2[k].shape}")
        else:
            print(f"{k} shape match: {d1[k].shape}")

# model.train(data="waymo16.yaml", device="cuda", batch=1, temporal_freeze=True, epochs=1)
