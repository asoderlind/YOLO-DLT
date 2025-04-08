from ultralytics import YOLO
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e6e8fb69 (remove ndo attribute)
import torch


# Defaults
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 327a1be6 (prepare for ablations for augs)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
EPOCHS = 100
OPTIMIZER = "SGD"
<<<<<<< HEAD


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
    name = f"world06{data_path}-{model_path}-{EPOCHS}e-{OPTIMIZER}-{'dist' if use_dist else 'noDist'}-scale{scale}-mosaic{mosaic}-noDontCare-d{d}_"
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
        cache=False,
        **kwargs,
    )


# Augmentation ablations
# train_with_distance(data_path="carla.yaml", use_dist=True, d=0.05, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.0, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.01, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.04, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.06, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", max_dist=100, use_dist=True, d=0.10, classes=[0, 1, 2, 3, 4, 5])

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
=======


# Defaults
model_path = "yolo11n.pt"
<<<<<<< HEAD
model = YOLO(model_path)
data_path = "waymo128-noConfidence.yaml"
# data_path = "coco8-dist.yaml"
device = "mps"
use_fe = False
epochs = 30
opt = "auto"

=======
data_path = "kitti.yaml"
device = "cuda"
use_fe = False
epochs = 100

use_dists = [True, False]
dists = [0.01]
>>>>>>> 4497c1dd (add kitti support)

use_dists = [True]
augs = [True]

for use_dist in use_dists:
    for aug in augs:
        model.train(
            data=data_path,
            batch=2,
            epochs=epochs,
            device=device,
            optimizer=opt,
            use_fe=use_fe,
            mosaic=1.0 if aug else 0.0,
            translate=0.1 if aug else 0.0,
            scale=0.5 if aug else 0.0,
            name=f"{data_path}-{model_path}-{epochs}e-{'dist' if use_dist else 'noDist'}-{'fe' if use_fe else 'noFe'}-{'aug' if aug else 'noAug'}-{opt}",
            iou_type="ciou",
            use_dist=use_dist,
        )
>>>>>>> 110dccde (rename files)
=======
model_path = "yolo11n.pt"
data_path = "kitti.yaml"
=======
>>>>>>> 6a83822a (mod8)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
use_fe = False
epochs = 1
optimizer = "SGD"
scale = 0.0
mosaic = 1.0
confs = [
    {
        "d": 0.025,
        "useDist": True,
        "dataPath": "waymo-noConf.yaml",
        "model_path": "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-noDontCare-d0.025_/weights/last.pt",
    },
    {"d": 0, "useDist": False, "dataPath": "waymo-noConf.yaml", "model_path": "yolo11n.pt"},
]

# model_path=f"{name}/weights/last.py"
<<<<<<< HEAD
model = YOLO(model_path)
name = f"{data_path}-{model_path}-{epochs}e-{optimizer}-scale{scale}-mosaic{mosaic}-d{d}_"
<<<<<<< HEAD
model.train(
    data=data_path,
    epochs=epochs,
    device=device,
    optimizer=optimizer,
    momentum=0.9,
    lr0=0.01,
    warmup_bias_lr=0.0,
    use_fe=use_fe,
    name=name,
    iou_type="ciou",
    mosaic=mosaic,
    scale=scale,
    use_dist=use_dist,
    dist=d,
)
>>>>>>> e6e8fb69 (remove ndo attribute)
=======
for d in ds:
=======
for conf in confs:
    d = conf["d"]
    use_dist = conf["useDist"]
    data_path = conf["dataPath"]
    model_path = conf["model_path"]

<<<<<<< HEAD
=======
        'd': 0.05, 'useDist': True,
        'dataPath': 'waymo-noConf.yaml',
        'model_path': "yolo11n.pt",
        'classes': [1,2,3,4]
    },
    {
        'd': 0,
        'useDist': False,
        'dataPath': 'waymo-noConf.yaml',
        'model_path': "yolo11n.pt",
        'classes': [1,2,3,4]
    }
    ]

# model_path=f"{name}/weights/last.py"
for conf in confs:
    d = conf['d']
    use_dist = conf['useDist']
    data_path = conf['dataPath']
    model_path = conf['model_path']
    classes = conf['classes']
>>>>>>> 1ea6a77a (updates)
=======
>>>>>>> ecd10376 (train dist)
    resume = model_path != "yolo11n.pt"

>>>>>>> 6a83822a (mod8)
=======


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
>>>>>>> 327a1be6 (prepare for ablations for augs)
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
        **kwargs,
    )
<<<<<<< HEAD
>>>>>>> 5969bc06 (merge)
=======


# Augmentation ablations
<<<<<<< HEAD
train_with_distance(scale=0.0, mosaic=0.0)
train_with_distance(scale=0.0, mosaic=1.0)
train_with_distance(scale=0.5, mosaic=0.0)
train_with_distance(scale=0.5, mosaic=1.0)
>>>>>>> 327a1be6 (prepare for ablations for augs)
=======
# train_with_distance(data_path="carla.yaml", use_dist=False, d=0.0, classes=[0, 1, 2, 3, 4, 5])

# train_with_distance(data_path="kitti.yaml", use_dist=True, d=0.05, classes=KITTI_CLASSES)
# train_with_distance(data_path="carla.yaml", use_dist=True, d=0.05, classes=[0, 1, 2, 3, 4, 5])
train_with_distance(data_path="carla.yaml", use_dist=True, d=0.0, classes=[0, 1, 2, 3, 4, 5])
>>>>>>> cbbd73c4 (update)
