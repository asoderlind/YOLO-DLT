from ultralytics import YOLO


# Defaults
model_path = "./runs/detect/kitti.yaml-yolo11n.pt-100e-SGD-noAug-dist-noAug-d0.01_/weights/best.pt"
data_path = "kitti.yaml"
device = "cuda"

#model_path=f"{name}/weights/last.py"

model = YOLO(model_path)
name=f"{data_path}-{model_path}_"
model.val(
    data=data_path,
    device=device,
    name=name,
)
