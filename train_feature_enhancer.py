from ultralytics import YOLO
from train_conf import MODEL, DEVICE, OPTIMIZER, MOMENTUM, BATCH, IOU_TYPE, LR0, WARMUP_BIAS_LR, PRETRAINED, EPOCHS, SEED


def train_fe(model_name=MODEL, epochs=EPOCHS, seed=SEED, data_path="exDark-yolo.yaml", **kwargs):
    use_fe = kwargs.get('use_fe', False)  # default
    lambda_c = kwargs.get('lambda_c', 0.5)  # default

    name = f"{model_name}-{seed}s-{epochs}e-{data_path}{'-fe-' if use_fe else '-noFe-'}lambda_c{lambda_c}_"
    name = name.replace("/", "-")
    model = YOLO(MODEL)
    model.train(
        name=name,
        data=data_path,
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


##########
# EXDARK #
##########

# Baseline exDark-yolo
train_fe(
    data_path="exDark-yolo.yaml",
    use_fe=False,
    lambda_c=0.0,
    val=True,
)

# All loss on exDark with DLN
for lambda_c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    train_fe(
        data_path="exDark-yolo.yaml",
        use_fe=True,
        lambda_c=lambda_c,
    )

# Curriculum on exDark with DLN
for lambda_c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    first_layer_trained = train_fe(
        data_path="exDark-yolo.yaml",
        use_fe=True,
        lambda_c=lambda_c,
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
        lambda_c=0.0,
        freeze=1,
    )

# Curriculum on bdd100k with mirnet
"""
first_layer_trained = train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=True,
    epochs=10,
    lambda_c=0.5,
    box=0.0,
    cls=0.0,
    dfl=0.0,
    val=False
)
train_fe(
    model_name=f"runs/detect/dlt-models/yolo11n.yaml-10e-bdd100k_night_mirnet.yaml-fe-lambda_c0.5_2/weights/last.pt",
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=False,
    lambda_c=0.0,
    val=False,
    freeze=1
)

# Curriculum on bdd100k with DLN
first_layer_trained = train_fe(
    data_path="bdd100k_night.yaml", use_fe=True, epochs=10, lambda_c=0.5, box=0.0, cls=0.0, dfl=0.0, val=False
)
train_fe(
    model_name=f"runs/detect/{first_layer_trained}/weights/last.pt",
    data_path="bdd100k_night.yaml",
    use_fe=False,
    lambda_c=0.0,
    val=False,
    freeze=1,
)

# All loss on bdd100k with mirnet
train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=True,
    lambda_c=0.5,
)

# All loss on bdd100k with DLN
train_fe(
    data_path="bdd100k_night.yaml",
    use_fe=True,
    lambda_c=0.5,
)
"""
