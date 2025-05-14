from ultralytics import YOLO
from train_conf import (
    DEVICE,
    KITTI_CLASSES,
    OPTIMIZER,
    MOMENTUM,
    BATCH,
    IOU_TYPE,
    LR0,
    WARMUP_BIAS_LR,
    PRETRAINED,
    MODEL,
    EPOCHS,
    SEED
)


def train_with_distance(
    model_path: str = MODEL,
    data_path: str = "kitti.yaml",
    dist: float = 0.05,
    use_dist: bool = True,
    scale: float = 0.0,
    mosaic: float = 1.0,
    seed: int = SEED,
    epochs: int = EPOCHS,
    name=None,
    **kwargs,
) -> str:
    classes = kwargs.get("classes", [])
    class_string = "".join([str(c) for c in classes])

    model = YOLO(model_path)

    if name is None:
        name = f"{data_path}-{model_path}-{epochs}e-{seed}s-{OPTIMIZER}-{'dist' if use_dist else 'noDist'}-scale{scale}-mosaic{mosaic}-c{class_string}-d{dist}_"
        name = name.replace("/", "-")

    model.train(
        name=name,
        data=data_path,
        pretrained=PRETRAINED,
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
        seed=seed,
        **kwargs,
    )
    return name


#########
# WAYMO #
#########

# Default
"""

train_with_distance(
    data_path="waymo-noConf.yaml",
    use_dist=False,
    dist=0.00,
    scale=0.5,
    max_dist=85,
)
train_with_distance(
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.01,
    max_dist=85,
)
train_with_distance(
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.05,
    max_dist=85,
)
train_with_distance(
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.1,
    max_dist=85,
)
train_with_distance(
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.5,
    max_dist=85,
)
train_with_distance(
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=1.0,
    max_dist=85,
)

# SPDConv
train_with_distance(
    model_path="dlt-models/yolo11n-SPDConv-3.yaml",
    data_path="waymo-noConf.yaml",
    use_dist=False,
    dist=0.00,
    scale=0.5,
    max_dist=85,
)
train_with_distance(
    model_path="dlt-models/yolo11n-SPDConv-3.yaml",
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.01,
    max_dist=85,
)
train_with_distance(
    model_path="dlt-models/yolo11n-SPDConv-3.yaml",
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.05,
    max_dist=85,
)
train_with_distance(
    model_path="dlt-models/yolo11n-SPDConv-3.yaml",
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.1,
    max_dist=85,
)
train_with_distance(
    model_path="dlt-models/yolo11n-SPDConv-3.yaml",
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=0.5,
    max_dist=85,
)
train_with_distance(
    model_path="dlt-models/yolo11n-SPDConv-3.yaml",
    data_path="waymo-noConf.yaml",
    use_dist=True,
    dist=1.0,
    max_dist=85,
)
"""

#########
# KITTI #
#########

# Test without dist
'''
 train_with_distance(data_path="kitti.yaml", use_dist=True, dist=0.5, classes=KITTI_CLASSES, epochs=200)
train_with_distance(
    data_path="kitti.yaml", model_path="runs/detect/kitti.yaml-dlt-models-yolo11n-SPDConv-3.yaml-200e-SGD-dist-scale0.0-mosaic1.0-c01234567-d0.5_5/weights/last.pt", use_dist=True, dist=0.5, classes=KITTI_CLASSES, epochs=200, resume=True
)
train_with_distance(
    data_path="kitti.yaml", model_path="dlt-models/yolo11n-SPDConv-3.yaml", use_dist=True, dist=1.0, classes=KITTI_CLASSES, epochs=200)
'''

# last ablations
# train_with_distance(data_path="kitti.yaml", use_dist=True, dist=0.5, seed=2, classes=KITTI_CLASSES, epochs=200, scale=0.5, mosaic=0.0)
# train_with_distance(data_path="kitti.yaml", use_dist=True, dist=0.5, seed=2, classes=KITTI_CLASSES, epochs=200, scale=0.5, mosaic=1.0)
# no dist
for s in [1]:
    for d in [1.0]:
        train_with_distance(data_path="kitti.yaml", use_dist=True, dist=d, seed=s, classes=KITTI_CLASSES, epochs=200, scale=0.0, mosaic=1.0)

for s in [2]:
    train_with_distance(data_path="kitti.yaml", use_dist=False, dist=d, seed=s, classes=KITTI_CLASSES, epochs=200, scale=0.5, mosaic=1.0)
    for d in [0.01, 0.05, 0.1, 1.0]:
        train_with_distance(data_path="kitti.yaml", use_dist=True, dist=d, seed=s, classes=KITTI_CLASSES, epochs=200, scale=0.0, mosaic=1.0)

# train_with_distance(data_path="kitti.yaml", use_dist=True, dist=0.5, seed=seed, classes=KITTI_CLASSES, epochs=200, scale=0.0, mosaic=0.0)
# train_with_distance(data_path="kitti.yaml", use_dist=True, dist=0.5, seed=seed, classes=KITTI_CLASSES, epochs=200, scale=0.0, mosaic=1.0)
