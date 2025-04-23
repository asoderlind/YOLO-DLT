from ultralytics import YOLO
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ENV = os.getenv("ENV", "LOCAL")
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6]


def eval_distance(model_path, data_path, use_dist, **kwargs):
    model_name = f"val_distance/{model_path.split('/')[-3]}"
    model = YOLO(model_path)
    model.val(
        device=DEVICE,
        data=data_path,
        use_dist=use_dist,
        name=model_name,
        **kwargs,
    )


if __name__ == "__main__":
    eval_distance(
        "runs/detect/kitti.yaml-yolo11n.pt-100e-SGD-dist-scale0.5-mosaic1.0-noDontCare-d0_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    """
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-noDontCare-d0.05_2/weights/best.pt",
        "waymo128-noConf.yaml",
        True,
        max_dist=85,
    )
    """
