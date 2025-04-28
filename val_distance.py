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
        "runs/detect/carla-town06-night.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt",
        "carla-town06-night.yaml",
        True,
        max_dist=100,
    )
    eval_distance(
        "runs/detect/carla-town06-night.yaml-yolo11n.pt-100e-SGD-noDist-scale0.0-mosaic1.0-c-d0.0_/weights/best.pt",
        "carla-town06-night.yaml",
        False,
        max_dist=100,
    )
    """
    eval_distance(
        "runs/detect/kitti.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-noDontCare-d0.01_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.1_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.5_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d1.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d2.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.01_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    """
