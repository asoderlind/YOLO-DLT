from ultralytics import YOLO
import torch


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

"""
Regular set with distance loss disabled
"""
model1 = YOLO("yolo11n.pt")
model1.train(data="../ultralytics/cfg/datasets/bdd100k.yaml", epochs=1, device=device, use_fe=False, use_dist=False)

"""
Distance set with distance loss enabled
"""
# model2 = YOLO("yolo11n.pt")
# model2.train(data="../ultralytics/cfg/datasets/kitti-mini.yaml", epochs=1, device=device, use_fe=False, use_dist=True)

"""
Distance set with distance loss disabled
"""
# model3 = YOLO("yolo11n.pt")
# model3.train(data="../ultralytics/cfg/datasets/kitti-mini.yaml", epochs=1, device=device, use_fe=False, use_dist=False)
