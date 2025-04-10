from ultralytics import YOLO

""" might resume this one later
# Without fe
fe = False
epochs = 200
lc = 0.0

data = "exDark-yolo.yaml"
model_name = "dlt-models/yolo11n-SPDConv-3.yaml"
model = YOLO(model_name)

model.train(
    data=data,
    use_fe=fe,
    epochs=epochs,
    augment=True,
    device="cuda",
    lambda_c=lc,
    optimizer="auto",
    name=f"{model_name}-{data}-{'fe-' if fe else ''}-e{epochs}-allLoss-lc{lc}-Auto-aug-preLoad",
)

# With training first layer at same time as rest of net
fe = True
epochs = 200
lc = 0.5

data = "exDark-yolo.yaml"
model_name = "dlt-models/yolo11n-SPDConv-3.yaml"
model = YOLO(model_name)

model.train(
    data=data,
    use_fe=fe,
    epochs=epochs,
    augment=True,
    device="cuda",
    lambda_c=lc,
    optimizer="auto",
    name=f"{model_name}-{data}-{'fe-' if fe else ''}-e{epochs}-allLoss-lc{lc}-Auto-aug-preLoad",
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

freezeNum = 1
epochs = 200
model = YOLO(f"runs/detect/{pathConv1}/weights/last.pt")
model.train(
    data="bdd100k_night.yaml",
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
