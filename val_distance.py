from ultralytics import YOLO
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ENV = os.getenv("ENV", "LOCAL")
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6]
WAYMO_CLASSES = [0, 1, 2, 3, 4]

configs = {
    "LOCAL": {
        """
        "KITTI": {
            "use_dist": True,
            "model_path": "runs/detect/kitti.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-noDontCare-d0.05_/weights/best.pt",
            "data_path": "kitti-mini.yaml",
            "classes": KITTI_CLASSES,
            "max_dist": 150,
        },
        """
        "WAYMO": {
            "use_dist": True,
            "model_path": "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-noDontCare-d0.05_2/weights/best.pt",
            "data_path": "waymo128-noConf.yaml",
            "classes": WAYMO_CLASSES,
            "max_dist": 85,
        },
    }
}


for conf in configs[ENV].values():
    model_path = str(conf["model_path"])
    data_path = str(conf["data_path"])
    classes = conf["classes"]
    use_dist = conf["use_dist"]

    name = f"val_distance/{data_path}-{model_path.replace('/', '-')}_"

    model = YOLO(model_path)

    model.val(
        device=DEVICE,
        data=data_path,
        classes=classes,
        use_dist=use_dist,
        name=name,
    )
