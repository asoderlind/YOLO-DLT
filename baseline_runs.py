import os
import time
from datetime import datetime

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

    # Record start time
    start_time = time.time()

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
        **kwargs,  # Pass any additional kwargs to train
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Format time as hours, minutes, seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    project = kwargs.get("project", "")

    output_path = project if project else f"runs/detect/{final_name}/"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = os.path.join(output_path, "timing.txt")

    # Save timing information to file
    with open(output_file, "w") as f:
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {data}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total training time: {formatted_time}\n")
        f.write(f"Total seconds: {elapsed_time:.2f}\n")

    print(f"\nTraining completed in {formatted_time}")
    print(f"Timing information saved to: {output_path}")

    return results


def run_seeded_train(
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


if __name__ == "__main__":
    classes = 0
    run_seeded_train(
        name="waymo_dark-yolo11n",
        model="yolo11n.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        name="waymo_dark-yolo11n-bic-repc3k2",
        model="dlt-models/yolo11n-bic-repc3k2.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        name="waymo_dark-yolo11s",
        model="yolo11s.yaml",
        dataset="waymo_dark.yaml",
    )

    run_seeded_val(
        name="waymo_dark-yolo11n",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        name="waymo_dark-yolo11n-bic-repc3k2",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        name="waymo_dark-yolo11s",
        data="waymo_dark.yaml",
        classes=classes,
    )
