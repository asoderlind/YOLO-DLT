from ultralytics import YOLO
from train_conf import (
        MODEL,DEVICE, OPTIMIZER, MOMENTUM, BATCH, IOU_TYPE, LR0, WARMUP_BIAS_LR, PRETRAINED, EPOCHS
        )


def train_fe(model_name=MODEL, data="exDark-yolo.yaml", lc=0.5, fe=True, **kwargs):
    name = f"{model_name}-{data}{'-fe-' if fe else '-noFe-'}e{epochs}-allLoss-lc{lc}-Auto-aug-preLoad_"
    model = YOLO(MODEL)
    model.train(
        data=data,
        use_fe=fe,
        epochs=EPOCHS,
        device=DEVICE,
        lambda_c=lc,
        optimizer=OPTIMIZER,
        name=name,
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
