from ultralytics import YOLO
import torch


# Defaults
<<<<<<< HEAD
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
use_fe = False
epochs = 100
optimizer = "SGD"
scale = 0.0
mosaic = 1.0
confs = [
    {
        'd': 0,
        'useDist': False,
        'dataPath': 'waymo-noConf.yaml',
        'model_path': "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-noDist-scale0.0-mosaic1.0-noDontCare-d0_2/weights/last.pt",
        'classes': [1, 2, 3, 4]
    },
]
=======
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
EPOCHS = 100
OPTIMIZER = "SGD"
>>>>>>> b59a904967bdcac4c2a2444715962bc03e916184


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
    resume = model_path != "yolo11n.pt"
    model = YOLO(model_path)
    name = f"{data_path}-{model_path}-{EPOCHS}e-{OPTIMIZER}-{'dist' if use_dist else 'noDist'}-scale{scale}-mosaic{mosaic}-noDontCare-d{d}_"

    model.train(
        data=data_path,
        epochs=EPOCHS,
        device=device,
        optimizer=OPTIMIZER,
        batch_size=16,
        momentum=0.9,
        lr0=0.01,
        warmup_bias_lr=0.0,
        name=name,
        iou_type="ciou",
        mosaic=mosaic,
        scale=scale,
        use_dist=use_dist,
        dist=d,
        resume=resume,
        classes=classes,
        **kwargs,
    )


# Augmentation ablations
train_with_distance(scale=0.0, mosaic=0.0)
train_with_distance(scale=0.0, mosaic=1.0)
train_with_distance(scale=0.5, mosaic=0.0)
train_with_distance(scale=0.5, mosaic=1.0)
