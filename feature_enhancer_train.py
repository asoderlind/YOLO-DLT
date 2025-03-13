from ultralytics import YOLO


# With training first layer at same time as rest of net
"""
fe = True
epochs = 100
lcs = [0.2, 0.3, 0.4]

for lc in lcs:
    model = YOLO("yolo11n.yaml")
    model.train(
        data="../ultralytics/cfg/datasets/exDark-yolo.yaml",
        use_fe=fe,
        epochs=epochs,
        batch=16,
        augment=True,
        device="cuda",
        lambda_c=lc,
        optimizer="auto",
        name=f"exDark-{'fe-' if fe else ''}{epochs}-allLoss-lc{lc}-Auto-aug-preLoad"
    )


"""
# With training only first layer then freezing
fe = True
epochs_list = [10]
lcs = [0.5]

for epochs in epochs_list:
    for lc in lcs:
        pathConv1 = f"exDark-conv1-e{epochs}-lc{lc}"
        model = YOLO("yolo11n.yaml")
        model.train(
            data="../ultralytics/cfg/datasets/exDark-yolo.yaml",
            epochs=epochs,
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
            save_json=True,
            name=pathConv1,
        )

        freezeNum = 1
        model = YOLO(f"runs/detect/{pathConv1}/weights/last.pt")
        model.train(
            data="../ultralytics/cfg/datasets/exDark-yolo.yaml",
            epochs=100,
            batch=16,
            pretrained=False,
            optimizer="auto",
            device="cuda",
            use_fe=False,
            val=True,
            lambda_c=lc,
            augment=True,
            name=f"{pathConv1}-freeze{freezeNum}-e100_",
            save_json=True,
            freeze=freezeNum,
        )
