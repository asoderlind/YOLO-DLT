from ultralytics import YOLO
import torch


# Defaults
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
EPOCHS = 100
OPTIMIZER = "SGD"


def train_with_distance(
    model_path: str = "yolo11n.pt",
    data_path: str = "kitti.yaml",
    d: float = 0.05,
    use_dist: bool = True,
    classes=KITTI_CLASSES,
    scale: float = 0.0,
    mosaic: float = 1.0,
    device: str = DEVICE,
    **kwargs,
):
    model = YOLO(model_path)
    name = f"{data_path}-{model_path}-{EPOCHS}e-{OPTIMIZER}-{'dist' if use_dist else 'noDist'}-scale{scale}-mosaic{mosaic}-noDontCare-d{d}_"
    name = name.replace("/", "-")

    model.train(
        pretrained=True,
        data=data_path,
        epochs=EPOCHS,
        device=device,
        optimizer=OPTIMIZER,
        batch=16,
        momentum=0.9,
        lr0=0.01,
        warmup_bias_lr=0.0,
        name=name,
        iou_type="ciou",
        mosaic=mosaic,
        scale=scale,
        use_dist=use_dist,
        dist=d,
        classes=classes,
        **kwargs,
    )


# Augmentation ablations
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, d=0.05, max_dist=85, classes=[0,1,2])
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, d=0.00, max_dist=85, classes=[0,1,2])
train_with_distance(data_path="waymo-noConf.yaml", use_dist=False, d=0.00, max_dist=85, classes=[0,1,2])

"""
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.0, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.01, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.04, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.06, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.10, classes=[0, 1, 2, 3, 4, 5])

train_with_distance(
    data_path="kitti.yaml",
    model_path="runs/detect/kitti.yaml-yolo11n.pt-100e-SGD-noDist-scale0.0-mosaic1.0-noDontCare-d0_/weights/best.pt",
    use_dist=True,
    d=0.05,
    classes=KITTI_CLASSES,
    freeze=23,
)
train_with_distance(
    data_path="kitti.yaml",
    model_path="runs/detect/kitti.yaml-yolo11n.pt-100e-SGD-noDist-scale0.0-mosaic1.0-noDontCare-d0_/weights/best.pt",
    use_dist=True,
    d=0.5,
    classes=KITTI_CLASSES,
    freeze=23,
)
train_with_distance(
    data_path="kitti.yaml",
    model_path="./runs/detect/kitti.yaml-yolo11n.pt-100e-SGD-noDist-scale0.0-mosaic1.0-noDontCare-d0_/weights/best.pt",
    use_dist=True,
    d=1.0,
    classes=KITTI_CLASSES,
    freeze=23,
)
"""
