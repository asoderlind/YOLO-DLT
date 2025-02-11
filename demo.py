from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco128.yaml", epochs=100, imgsz=640)