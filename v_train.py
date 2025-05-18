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
    #     name="waymo_night-yolo11nv-4_gframe-0.005_lr0-5_temporal_cls",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
    #     gframe=4,
    #     epochs=10,
    #     cos_lr=True,
    #     data="waymo_night.yaml",
    #     lr0=0.005,
    #     temporal_cls=0.5,
    #     batch=1,
    #     temporal_freeze=True,
    # )
    # train_model_v(
    #     name="waymo_night-yolo11nv-4_gframe-0.01_lr0-5_temporal_cls",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
    #     gframe=4,
    #     epochs=10,
    #     cos_lr=True,
    #     data="waymo_night.yaml",
    #     lr0=0.01,
    #     temporal_cls=0.5,
    #     batch=1,
    #     temporal_freeze=True,
    # )
    # train_model_v(
    #     name="waymo_night-yolo11nv-4_gframe-0.005_lr0-8_temporal_cls",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
    #     gframe=4,
    #     epochs=10,
    #     cos_lr=True,
    #     data="waymo_night.yaml",
    #     lr0=0.005,
    #     temporal_cls=8.0,
    #     batch=1,
    #     temporal_freeze=True,
    # )
    train_model_v(
        name="waymo_night-yolo11nv-4_gframe-0.01_lr0-10_temporal_cls",
        model="dlt-models/yolo11n-temporal.yaml",
        model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
        gframe=4,
        epochs=10,
        cos_lr=True,
        data="waymo_night.yaml",
        lr0=0.01,
        temporal_cls=10.0,
        batch=1,
        temporal_freeze=True,
    )
    # train_model_v(
    #     name="waymo-yolo11nv-4tw-1s-0.0005lr0",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     temporal_window=4,
    #     temporal_stride=1,
    #     model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
    #     temporal_freeze=True,
    #     epochs=20,
    #     cos_lr=True,
    #     data="waymo-noConf-noDist-vid.yaml",
    #     mosaic=0.0,
    #     lr0=0.0005,
    #     batch=1,
    #     workers=8,
    # )

    # train_model_v(
    #     name="waymo-yolo11nv-4tw-1s-0.0005lr0-0.5temporal_cls",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     temporal_window=4,
    #     temporal_stride=1,
    #     model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
    #     temporal_freeze=True,
    #     epochs=10,
    #     cos_lr=True,
    #     data="waymo-noConf-noDist-vid.yaml",
    #     mosaic=0.0,
    #     lr0=0.0005,
    #     workers=8,
    #     temporal_cls=0.5,
    #     batch=1,
    # )

    # train_model_v(
    #     name="waymo-yolo11nv-4tw-1s-0.0005lr0-0.5temporal_cls-0cls",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     temporal_window=4,
    #     temporal_stride=1,
    #     model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
    #     temporal_freeze=True,
    #     epochs=10,
    #     cos_lr=True,
    #     data="waymo-noConf-noDist-vid.yaml",
    #     mosaic=0.0,
    #     lr0=0.0005,
    #     workers=8,
    #     temporal_cls=0.5,
    #     cls=0.0,
    #     batch=1,
    # )

    # train_model_v(
    #     name="waymo-yolo11nv-4tw-1s",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     temporal_window=4,
    #     temporal_stride=1,
    #     model_load_path="",
    #     temporal_freeze=True,
    #     epochs=10,
    #     cos_lr=True,
    #     data="waymo-noConf-noDist-vid.yaml",
    #     mosaic=0.0,
    #     batch=1,
    # )
    # train_model_v(
    #     name="waymo-yolo11nv-4tw-1s-0.0005lr0-0.5temporal_cls-109e",
    #     model="dlt-models/yolo11n-temporal.yaml",
    #     temporal_window=4,
    #     temporal_stride=1,
    #     model_load_path="runs/detect/waymo-yolo11n-bdd100k_night-50e-lr0.001-lrf0.01-freeze0-SGD/weights/last.pt",
    #     temporal_freeze=True,
    #     epochs=109,
    #     cos_lr=True,
    #     data="waymo-noConf-noDist-vid.yaml",
    #     mosaic=0.0,
    #     lr0=0.0005,
    #     workers=8,
    #     temporal_cls=0.5,
    #     batch=1,
    # )
