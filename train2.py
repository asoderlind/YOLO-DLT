import os

import torch.nn as nn

from train_conf import (
    BATCH,
    CLUSTER_OUTPUT_PATH,
    DEVICE,
    EPOCHS,
    IOU_TYPE,
    LR0,
    MODEL,
    MOMENTUM,
    OPTIMIZER,
    PRETRAINED,
    WARMUP_BIAS_LR,
)
from ultralytics import YOLO

activations: dict[str, nn.Module] = {
    "hardswish": nn.Hardswish(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "gelu-tanh": nn.GELU(approximate="tanh"),
}


def train_model(
    name="{data}-{model}-{fe}-{augment}-{epochs}e",
    model=MODEL,
    data="bdd100k_night.yaml",
    batch=BATCH,
    epochs=EPOCHS,
    device=DEVICE,
    use_fe=False,
    augment=True,
    pretrained=PRETRAINED,
    iou_type=IOU_TYPE,
    warmup_bias_lr=WARMUP_BIAS_LR,
    lr0=LR0,
    optimizer=OPTIMIZER,
    momentum=MOMENTUM,
    project=CLUSTER_OUTPUT_PATH,
    classes=None,
    resume=None,
    **kwargs,
):
    """
    Train a YOLO model with timing measurement

    Args:
        name (str): Template for the run name
        model (str): Path to the model YAML
        data (str): Path to the dataset YAML
        batch (int): Batch size
        epochs (int): Number of training epochs
        device (str): Training device (cuda, cpu)
        use_fe (bool): Whether to use feature engineering
        augment (bool): Whether to use augmentation
        pretrained (bool): Whether to use pretrained weights
        iou_type (str): IOU type for loss calculation
        warmup_bias_lr (float): Warmup bias learning rate
        classes (list): List of class indices to train on
        **kwargs: Additional arguments passed to YOLO train

    Returns:
        The results from model training
    """
    # Load the model
    model_obj = YOLO(model)

    # Build name
    final_name = name.format(
        data=os.path.basename(data).split(".")[0],
        epochs=epochs,
        augment=f"{'aug' if augment else 'noAug'}",
        fe=f"{'fe' if use_fe else 'noFe'}",
        warmup_bias_lr=warmup_bias_lr,
        model=os.path.basename(model).split(".")[0],
    )

    # Set default classes if not provided
    # classes_to_use = classes if classes is not None else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Train the model
    results = model_obj.train(
        data=data,
        batch=batch,
        epochs=epochs,
        device=device,
        pretrained=pretrained,
        optimizer=optimizer,
        lr0=lr0,
        momentum=momentum,
        use_fe=use_fe,
        augment=augment,
        name=final_name,
        iou_type=iou_type,
        warmup_bias_lr=warmup_bias_lr,
        project=project,
        classes=classes,
        resume=resume,
        **kwargs,  # Pass any additional kwargs to train
    )

    return results


def run_seeded_train(
    start_itr=0,
    iterations: int = 3,
    name="{data}-{model}-{fe}-{augment}-{epochs}e",
    model=MODEL,
    data="bdd100k_night.yaml",
    batch=BATCH,
    epochs=EPOCHS,
    device=DEVICE,
    use_fe=False,
    augment=True,
    pretrained=PRETRAINED,
    iou_type=IOU_TYPE,
    warmup_bias_lr=WARMUP_BIAS_LR,
    lr0=LR0,
    optimizer=OPTIMIZER,
    momentum=MOMENTUM,
    classes=None,
    **kwargs,
):
    if start_itr == 0:
        for i in range(iterations):
            train_model(
                name=f"{name}-seed-{i}",
                model=model,
                data=data,
                batch=batch,
                epochs=epochs,
                device=device,
                use_fe=use_fe,
                augment=augment,
                pretrained=pretrained,
                iou_type=iou_type,
                warmup_bias_lr=warmup_bias_lr,
                lr0=lr0,
                optimizer=optimizer,
                momentum=momentum,
                classes=classes,
                seed=i,
                **kwargs,
            )
    else:
        for i in range(start_itr, start_itr + iterations):
            train_model(
                name=f"{name}-seed-{i}",
                model=model,
                data=data,
                batch=batch,
                epochs=epochs,
                device=device,
                use_fe=use_fe,
                augment=augment,
                pretrained=pretrained,
                iou_type=iou_type,
                warmup_bias_lr=warmup_bias_lr,
                lr0=lr0,
                optimizer=optimizer,
                momentum=momentum,
                classes=classes,
                seed=i,
                **kwargs,
            )


def run_seeded_val(
    iterations: int = 3,
    name="{data}-{model}-{fe}-{augment}-{epochs}e",
    data="bdd100k_night.yaml",
    project=CLUSTER_OUTPUT_PATH,
    batch=BATCH,
    device=DEVICE,
    classes=None,
    **kwargs,
):
    for i in range(iterations):
        model_path = f"{project}/{name}-seed-{i}/weights/last.pt"
        model = YOLO(model_path)
        model.val(
            data=data,
            project=project,
            name=f"{name}-seed-{i}",
            batch=batch,
            device=device,
            classes=classes,
            kwargs=kwargs,
        )


grid_sgd = {
    "epochs": [50, 100, 150, 200],
    "lr0": [0.001, 0.005, 0.01],
    "lrf": [0.01, 0.005, 0.001],
    "freeze": [0, 10, 22],
    "optinmizer": "SGD",
}

grid_adamw = {
    "epochs": [50, 100, 150, 200],
    "lr0": [0.001, 0.0005, 0.0001],
    "lrf": [0.01, 0.005, 0.001],
    "freeze": [0, 10, 22],
    "optimizer": "AdamW",
}
# Example usage


def run_grid_search():
    # Phase 1: Quick Exploration
    configurations = [
        # # SGD configurations
        # {"optimizer": "SGD", "freeze": 10, "epochs": 50, "lr0": 0.005, "lrf": 0.01},
        # {"optimizer": "SGD", "freeze": 22, "epochs": 50, "lr0": 0.01, "lrf": 0.01},
        # {"optimizer": "SGD", "freeze": 0, "epochs": 50, "lr0": 0.001, "lrf": 0.01},
        # AdamW configurations
        # {"optimizer": "AdamW", "freeze": 10, "epochs": 50, "lr0": 0.0005, "lrf": 0.01},
        # {"optimizer": "AdamW", "freeze": 22, "epochs": 50, "lr0": 0.001, "lrf": 0.01},
        # {"optimizer": "AdamW", "freeze": 0, "epochs": 50, "lr0": 0.0001, "lrf": 0.01},
    ]

    results = []
    for i, config in enumerate(configurations):
        print(f"Running configuration {i + 1}/{len(configurations)}: {config}")

        name = f"waymo-yolo11n-bdd100k_night-{config['epochs']}e-lr{config['lr0']}-lrf{config['lrf']}-freeze{config['freeze']}-{config['optimizer']}"

        result = train_model(
            name=name,
            model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
            data="waymo-noConf-noDist-vid.yaml",
            epochs=config["epochs"],
            lr0=config["lr0"],
            lrf=config["lrf"],
            freeze=config["freeze"],
            optimizer=config["optimizer"],
        )

        results.append((config, result))

        # Save intermediate results to analyze
        with open("grid_search_results_phase1.txt", "a") as f:
            f.write(f"Config: {config}\n")
            f.write(f"Results: {result.results_dict}\n\n")

    # Analyze phase 1 results and continue with phase 2
    # This would typically involve checking metrics and selecting best configurations

    # Phase 2 and 3 would follow similar patterns


if __name__ == "__main__":
    data = "bdd100k_night.yaml"
    # waymo_night = "waymo_cluster_night.yaml"
    # waymo = "waymo_cluster.yaml"
    # bdd100k = "bdd100k_cluster.yaml"

    # train_model(
    #     name="bdd100k_night-yolo9t-seed-0",
    #     model="runs/detect/bdd100k_night-yolo9t-seed-0/weights/last.pt",
    #     data=data,
    #     project="",
    #     resume=True,
    # )

    # train_model(
    #     name="bdd100k_night-yolo8n-seed-1",
    #     model="yolov8n.yaml",
    #     data=data,
    #     project="",
    #     seed=1,
    # )

    # train_model(
    #     name="bdd100k_night-yolo8n-seed-2",
    #     model="runs/detect/bdd100k_night-yolo8n-seed-2/weights/last.pt",
    #     data=data,
    #     project="",
    #     seed=2,
    #     resume=True,
    # )

    # train_model(
    #     name="bdd100k_night-yolo9t-seed-1",
    #     model="runs/detect/bdd100k_night-yolo9t-seed-12/weights/last.pt",
    #     data=data,
    #     project="",
    #     seed=1,
    #     resume=True,
    # )

    train_model(
        name="bdd100k_night-yolo11n-bic-repc3k2-wiou2-seed-2",
        model="runs/detect/bdd100k_night-yolo11n-bic-repc3k2-wiou2-seed-2/weights/last.pt",
        data=data,
        project="",
        seed=2,
        iou_type="wiou2",
        resume=True,
    )

    train_model(
        name="bdd100k_night-yolo11n-bic-repc3k2-nwd-seed-2",
        model="dlt-models/yolo11n-bic-repc3k2.yaml",
        data=data,
        project="",
        seed=2,
        iou_type="nwd",
    )

    train_model(
        name="bdd100k_night-yolo11n-bic-repc3k2-mpdiou-seed-2",
        model="dlt-models/yolo11n-bic-repc3k2.yaml",
        data=data,
        project="",
        seed=2,
        iou_type="mpdiou",
    )

    train_model(
        name="bdd100k_night-yolo11n-bic-repc3k2-simsppf-seed-2",
        model="dlt-models/yolo11n-bic-repc3k2-simsppf.yaml",
        data=data,
        project="",
        seed=2,
    )

    train_model(
        name="bdd100k_night-yolo9t-seed-2",
        model="runs/detect/bdd100k_night-yolo9t-seed-2/weights/last.pt",
        data=data,
        project="",
        seed=2,
        resume=True,
    )

    train_model(
        name="bdd100k_night-yolo11s-seed-1",
        model="yolo11s.yaml",
        data=data,
        project="",
        seed=1,
    )

    train_model(
        name="bdd100k_night-yolo11s-seed-2",
        model="yolo11s.yaml",
        data=data,
        project="",
        seed=2,
    )

    # run_seeded_train(
    #     iterations=3,
    #     name="bdd100k_night_cluster-yolo9t",
    #     model="yolov9t.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     iterations=3,
    #     name="bdd100k_night_cluster-yolo8n",
    #     model="yolov8n.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11s",
    #     model="yolo11s.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic",
    #     model="dlt-models/yolo11n-bic.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-repc3k2",
    #     model="dlt-models/yolo11n-repc3k2.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-simsppf",
    #     model="dlt-models/yolo11n-bic-repc3k2-simsppf.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-sppfcsp",
    #     model="dlt-models/yolo11n-bic-repc3k2-sppfcsp.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-simsppfcsp",
    #     model="dlt-models/yolo11n-bic-repc3k2-simsppfcsp.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     iterations=3,
    #     name="bdd100k_night_cluster-yolo11n-bic-rfac3k2",
    #     model="dlt-models/yolo11n-bic-rfac3k2.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-se",
    #     model="dlt-models/yolo11n-bic-repc3k2-se.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-eca",
    #     model="dlt-models/yolo11n-bic-repc3k2-eca.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-simam",
    #     model="dlt-models/yolo11n-bic-repc3k2-simam.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-ca",
    #     model="dlt-models/yolo11n-bic-repc3k2-ca.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     iterations=3,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-thiou-ciou_ass",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="thiou",
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-wiou1",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="wiou1",
    # )
    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-wiou2",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="wiou2",
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-siou",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="siou",
    # )
    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-ciou+nwd",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="ciou+nwd",
    # )
    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-eiou",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="eiou",
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-isiou",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="isiou",
    # )
    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-mpdiou",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="mpdiou",
    # )
    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-nwd",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="nwd",
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-bic",
    #     model="dlt-models/yolo11n-bic.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     start_itr=1,
    #     iterations=2,
    #     name="bdd100k_night_cluster-yolo11n-repc3k2",
    #     model="dlt-models/yolo11n-repc3k2.yaml",
    #     data=data,
    # )

    # run_seeded_train(
    #     iterations=3,
    #     name="bdd100k_night_cluster-yolo11n-bic-repc3k2-focal-eiou",
    #     model="dlt-models/yolo11n-bic-repc3k2.yaml",
    #     data=data,
    #     iou_type="focal-eiou",
    # )
