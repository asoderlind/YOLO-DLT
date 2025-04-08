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
    resume = model_path != "yolo11n.pt"
    model = YOLO(model_path)
    name = f"{data_path}-{model_path}-{EPOCHS}e-{OPTIMIZER}-{'dist' if use_dist else 'noDist'}-scale{scale}-mosaic{mosaic}-noDontCare-d{d}_"

    model.train(
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
        resume=resume,
        classes=classes,
        cache=False
    )


# Augmentation ablations
# train_with_distance(data_path="carla.yaml", use_dist=False, d=0.0, classes=[0, 1, 2, 3, 4, 5])

#train_with_distance(data_path="kitti.yaml", use_dist=True, d=0.05, classes=KITTI_CLASSES)
train_with_distance(data_path="carla.yaml", use_dist=True, d=0.05, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", use_dist=True, d=0.0, classes=[0, 1, 2, 3, 4, 5])

