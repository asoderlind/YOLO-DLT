from ultralytics import YOLO
from train_conf import MODEL, DEVICE, OPTIMIZER, MOMENTUM, BATCH, IOU_TYPE, LR0, WARMUP_BIAS_LR, PRETRAINED, EPOCHS


def train_fe(model_name=MODEL, epochs=EPOCHS, data_path="exDark-yolo.yaml", lc=0.5, use_fe=True, **kwargs):
    name = f"{model_name}-{epochs}e-{data_path}{'-fe-' if use_fe else '-noFe-'}lc{lc}_"
    name = name.replace("/", "-")
    model = YOLO(MODEL)
    model.train(
        name=name,
        data=data_path,
        lambda_c=lc,
        use_fe=use_fe,
        device=DEVICE,
        batch=BATCH,
        momentum=MOMENTUM,
        lr0=LR0,
        iou_type=IOU_TYPE,
        warmup_bias_lr=WARMUP_BIAS_LR,
        optimizer=OPTIMIZER,
        pretrained=PRETRAINED,
        epochs=epochs,
        **kwargs,
    )
    return name


# Curriculum on bdd100k with mirnet
"""
first_layer_trained = train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=True,
    epochs=10,
    lc=0.5,
    box=0.0,
    cls=0.0,
    dfl=0.0,
    val=False
)
train_fe(
    model_name=f"runs/detect/dlt-models/yolo11n.yaml-10e-bdd100k_night_mirnet.yaml-fe-lc0.5_2/weights/last.pt",
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=False,
    lc=0.0,
    val=False,
    freeze=1
)
"""

# Curriculum on bdd100k with DLN
first_layer_trained = train_fe(
    data_path="bdd100k_night.yaml", use_fe=True, epochs=10, lc=0.5, box=0.0, cls=0.0, dfl=0.0, val=False
)
train_fe(
    model_name=f"runs/detect/{first_layer_trained}/weights/last.pt",
    data_path="bdd100k_night.yaml",
    use_fe=False,
    lc=0.0,
    val=False,
    freeze=1,
)

"""
# All loss on bdd100k with mirnet
train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=True,
    lc=0.5,
)

# All loss on bdd100k with DLN
train_fe(
    data_path="bdd100k_night.yaml",
    use_fe=True,
    lc=0.5,
)

# All loss on exDark with DLN
for lc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    train_fe(
        data_path="exDark-yolo.yaml",
        use_fe=True,
        lc=lc,
        epochs=100,
        val=False,
    )

# Curriculum on exDark with DLN
for lc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    first_layer_trained = train_fe(
        data_path="exDark-yolo.yaml",
        use_fe=True,
        lc=lc,
        epochs=10,
        box=0.0,
        cls=0.0,
        dfl=0.0,
        val=False,
    )
    train_fe(
        model_name=f"runs/detect/{first_layer_trained}/weights/last.pt",
        data_path="exDark-yolo.yaml",
        use_fe=False,
        lc=0.0,
        val=False,
        freeze=1,
    )
"""
