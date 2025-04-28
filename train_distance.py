from ultralytics import YOLO
from train_conf import DEVICE, KITTI_CLASSES, OPTIMIZER, MOMENTUM, BATCH, IOU_TYPE, LR0, WARMUP_BIAS_LR


def train_with_distance(
    model_path: str = "yolo11n.pt",
    data_path: str = "kitti.yaml",
    dist: float = 0.05,
    use_dist: bool = True,
    scale: float = 0.0,
    mosaic: float = 1.0,
    **kwargs,
):
    classes = kwargs.get("classes", [])
    class_string = "".join([str(c) for c in classes])

    epochs = 100

    model = YOLO(model_path)
    name = f"{data_path}-{model_path}-{epochs}e-{OPTIMIZER}-{'dist' if use_dist else 'noDist'}-scale{scale}-mosaic{mosaic}-c{class_string}-d{dist}_"

    model.train(
        name=name,
        data=data_path,
        pretrained=True,
        use_dist=use_dist,
        dist=dist,
        mosaic=mosaic,
        scale=scale,
        epochs=epochs,
        device=DEVICE,
        batch=BATCH,
        momentum=MOMENTUM,
        lr0=LR0,
        iou_type=IOU_TYPE,
        warmup_bias_lr=WARMUP_BIAS_LR,
        optimizer=OPTIMIZER,
        **kwargs,
    )


#########
# KITTI #
#########

"""
train_with_distance(
    data_path="kitti.yaml",
    use_dist=True,
    dist=0.01,
    classes=KITTI_CLASSES,
)
train_with_distance(
    data_path="kitti.yaml",
    use_dist=True,
    dist=0.10,
    classes=KITTI_CLASSES,
)

# Test with a slightly larger model
train_with_distance(
    model_path="yolo11s.pt",
    data_path="kitti.yaml",
    use_dist=True,
    dist=0.05,
    classes=KITTI_CLASSES,
)
"""

# finetune only the distance head on pretrained models
"""
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

#########
# WAYMO #
#########

"""
#train_with_distance(data_path="waymo-noConf.yaml", use_dist=False, dist=0.00, scale=0.5, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, dist=0.01, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, dist=0.05, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, dist=0.10, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, dist=0.50, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, dist=1.00, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, dist=2.00, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=True, d=0.00, max_dist=85)
train_with_distance(data_path="waymo-noConf.yaml", use_dist=False, d=0.00, max_dist=85)
# Curriculum learning
train_with_distance(
        data_path="waymo-noConf.yaml",
        model_path="./runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_/weights/best.pt",
        use_dist=True,
        d=2.00,
        max_dist=85,
        freeze=23)

"""
#########
# CARLA #
#########

train_with_distance(data_path="carla-town06-night.yaml", max_dist=100, use_dist=True, dist=0.05)
train_with_distance(data_path="carla-town06-night.yaml", max_dist=100, use_dist=False, dist=0.00)

train_with_distance(data_path="carla-town06-sunset.yaml", max_dist=100, use_dist=True, dist=0.05)
train_with_distance(data_path="carla-town06-sunset.yaml", max_dist=100, use_dist=False, dist=0.00)

train_with_distance(data_path="carla-town06-day.yaml", max_dist=100, use_dist=True, dist=0.05)
train_with_distance(data_path="carla-town06-day.yaml", max_dist=100, use_dist=False, dist=0.00)
