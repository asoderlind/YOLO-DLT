from ultralytics import YOLO
from train_conf import MODEL, DEVICE, OPTIMIZER, MOMENTUM, BATCH, IOU_TYPE, LR0, WARMUP_BIAS_LR, PRETRAINED, EPOCHS


def train_fe(model_name=MODEL, data_path="exDark-yolo.yaml", lc=0.5, use_fe=True, **kwargs):
    name = f"{model_name}-{data_path}{'-fe-' if use_fe else '-noFe-'}-lc{lc}_"
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
        epochs=EPOCHS,
        **kwargs,
    )


train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=False,
    lc=0.0,
)

train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=True,
    lc=0.5,
)

train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=True,
    lc=0.1,
)

train_fe(
    data_path="bdd100k_night_mirnet.yaml",
    use_fe=True,
    lc=1.0,
)


"""
# With training only first layer then freezing
epochs = 10
fe = True
lc = 0.5

model_name = "dlt-models/yolo11n-SPDConv-3.yaml"
data = "exDark-yolo.yaml"

pathConv1 = f"{model_name}-{data}-conv1-e{epochs}-lc{lc}"
model = YOLO(model_name)
model.train(
    data=data,
    epochs=10,
    batch=16,
    pretrained=True,
    optimizer="auto",
    device="cuda",
    use_fe=fe,
    val=False,
    lambda_c=lc,
    augment=True,
    box=0.0,
    cls=0.0,
    dfl=0.0,
    name=pathConv1,
)
"""

"""
freezeNum = 1
epochs = 200
model = YOLO(f"runs/detect/{pathConv1}/weights/last.pt")
model.train(
    data=data,
    epochs=epochs,
    pretrained=False,
    optimizer="auto",
    device="cuda",
    use_fe=False,
    val=True,
    lambda_c=lc,
    augment=True,
    name=f"{pathConv1}-freeze{freezeNum}-e{epochs}_",
    save_json=True,
    freeze=freezeNum,
)
"""
