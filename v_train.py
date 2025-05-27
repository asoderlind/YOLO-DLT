import os
import time
from datetime import datetime

import torch.nn as nn

from ultralytics import YOLO

activations: dict[str, nn.Module] = {
    "hardswish": nn.Hardswish(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "gelu-tanh": nn.GELU(approximate="tanh"),
}


def train_model_v(
    name="{data}-{model}-{fe}-{augment}-{epochs}e",
    model="yolo11n.yaml",
    model_load_path: str = "",  # Path to the model pt file
    data="bdd100k_night.yaml",
    batch=16,
    epochs=200,
    device="cuda",
    use_fe=False,
    augment=True,
    pretrained=False,
    iou_type="ciou",
    warmup_bias_lr=0.0,
    lr0=0.01,
    optimizer="SGD",
    momentum=0.9,
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

    if model_load_path:
        # Load the model weights if provided
        model_obj.load(model_load_path)

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
        classes=classes,
        **kwargs,  # Pass any additional kwargs to train
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Format time as hours, minutes, seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    output_path = f"runs/detect/{final_name}/"

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


# Example usage
if __name__ == "__main__":
    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.5_nms-temporal_dl",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     gframe=16,
    #     batch=1,
    #     # batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.01_thresh-default_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.01.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     # gframe=16,
    #     batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.4_nms-default_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.4.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     # gframe=16,
    #     batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.7_nms-default_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.7.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     # gframe=16,
    #     batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-temporal_dl",
    #     model="yolo11n.yaml",
    #     batch=1,
    #     gframe=16,
    #     epochs=200,
    #     data="waymo_dark.yaml",
    # )

    train_model_v(
        name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.4_nms-0.4_nms_val-default_dl",
        model="dlt-models/yolo11n-temporal-nms-0.4.yaml",
        model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
        # gframe=16,
        batch=16,
        epochs=80,
        warmup_epochs=8,
        cos_lr=True,
        data="waymo_dark.yaml",
        lr0=0.005,
        nbs=16,
        temporal_freeze=True,
    )

    train_model_v(
        name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.01_thresh-default_dl",
        model="dlt-models/yolo11n-temporal-thresh-0.01.yaml",
        model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
        # gframe=16,
        batch=16,
        epochs=80,
        warmup_epochs=8,
        cos_lr=True,
        data="waymo_dark.yaml",
        lr0=0.005,
        nbs=16,
        temporal_freeze=True,
    )

    train_model_v(
        name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.02_thresh-default_dl",
        model="dlt-models/yolo11n-temporal-thresh-0.02.yaml",
        model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
        # gframe=16,
        batch=16,
        epochs=80,
        warmup_epochs=8,
        cos_lr=True,
        data="waymo_dark.yaml",
        lr0=0.005,
        nbs=16,
        temporal_freeze=True,
    )

    train_model_v(
        name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.05_thresh-default_dl",
        model="dlt-models/yolo11n-temporal-thresh-0.05.yaml",
        model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
        # gframe=16,
        batch=16,
        epochs=80,
        warmup_epochs=8,
        cos_lr=True,
        data="waymo_dark.yaml",
        lr0=0.005,
        nbs=16,
        temporal_freeze=True,
    )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.001_thresh-default_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.001.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     # gframe=16,
    #     batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.005_thresh-default_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.005.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     # gframe=16,
    #     batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    # )
