from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="../ultralytics/cfg/datasets/bdd1k.yaml", epochs=1, device="cpu")
