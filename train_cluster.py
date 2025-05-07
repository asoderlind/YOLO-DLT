from ultralytics import YOLO
from train_conf import (
    DEVICE,
    OPTIMIZER,
    MOMENTUM,
    BATCH,
    IOU_TYPE,
    LR0,
    WARMUP_BIAS_LR,
    PRETRAINED,
    EPOCHS,
)


model = YOLO("dlt-models/yolo11n.yaml")
model.train(
    project="/mnt/machine-learning-storage/ML1/ClusterOutput/MLC-499/Runs",
    data="ultralytics/cfg/datasets/kitti-cluster.yaml",
    use_dist=True,
    mosaic=1.0,
    scale=0.0,
    epochs=EPOCHS,
    pretrained=PRETRAINED,
    device=DEVICE,
    batch=BATCH,
    momentum=MOMENTUM,
    lr0=LR0,
    iou_type=IOU_TYPE,
    warmup_bias_lr=WARMUP_BIAS_LR,
    optimizer=OPTIMIZER,
)
