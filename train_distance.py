'''
{
    'd': 0.05, 'useDist': True,
    'dataPath': 'carla.yaml',
    'model_path': "yolo11n.pt",
    'classes': [0, 1, 2, 3, 4, 5]
},
{
    'd': 0,
    'useDist': False,
    'dataPath': 'carla.yaml',
    'model_path': "yolo11n.pt",
    'classes': [0, 1, 2, 3, 4, 5]
},
'''
from ultralytics import YOLO
import torch


# Defaults
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
use_fe = False
epochs = 100
optimizer = "SGD"
scale = 0.0
mosaic = 1.0
confs = [
    {
        'd': 0.05, 'useDist': True,
        'dataPath': 'waymo-noConf.yaml',
        'model_path': "yolo11n.pt",
        'classes': [1, 2, 3, 4]
    },
    {
        'd': 0,
        'useDist': False,
        'dataPath': 'waymo-noConf.yaml',
        'model_path': "yolo11n.pt",
        'classes': [1, 2, 3, 4]
    },
]

# model_path=f"{name}/weights/last.py"
for conf in confs:
    d = conf['d']
    use_dist = conf['useDist']
    data_path = conf['dataPath']
    model_path = conf['model_path']
    classes = conf['classes']
    resume = model_path != "yolo11n.pt"

    model = YOLO(model_path)

    name = f"{data_path}-{model_path}-{epochs}e-{optimizer}-{'dist' if use_dist else 'noDist'}-scale{scale}-mosaic{mosaic}-noDontCare-d{d}_"

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
        resume=resume,
        classes=classes
        # classes=[0,1,2,3,4,5,6,7] KITTI classes
    )
