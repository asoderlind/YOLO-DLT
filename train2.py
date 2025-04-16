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


def train_model(
    name="{data}-{model}-{fe}-{augment}-{epochs}e",
    model="yolo11n.yaml",
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
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.001,
    #     freeze=10,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night",
    #     model="runs/detect/waymo-noConf-noDist-vid-yolo11n-bdd100k_night4/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.005,
    #     freeze=10,
    #     resume=True,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-lr0.01",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.01,
    #     freeze=10,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-lr0.005",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.005,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-30e-lr0.001",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=30,
    #     lr0=0.001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-lr0.005",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.0005,
    # )

    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-headTune-20e",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=20,
    #     lr0=0.001,
    #     freeze=22,
    #     optimizer="AdamW",
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-headTune-30e",
    #     model="runs/detect/waymo-noConf-noDist-vid-yolo11n-bdd100k_night-headTune-30e/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     resume=True,
    #     epochs=20,
    #     lr0=0.001,
    #     freeze=22,
    #     optimizer="AdamW",
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-headTune-20e",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=20,
    #     lr0=0.001,
    #     freeze=22,
    # )

    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-spdconv-bdd100k_night-50e",
    #     model="runs/detect/waymo-noConf-noDist-vid-yolo11n-spdconv-bdd100k_night-50e/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     resume=True,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-AdamW-50e-lr0.001",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-AdamW-50e-lr0.0001",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=50,
    #     lr0=0.0001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-AdamW-40e-lr0.001",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=40,
    #     lr0=0.001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-AdamW-30e-lr0.001",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=30,
    #     lr0=0.001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-40e",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=40,
    #     lr0=0.001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-30e",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=30,
    #     lr0=0.001,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-10e-lr0.001-lrf0.1",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=10,
    #     lr0=0.001,
    #     lrf=0.1,
    # )
    # train_model(
    #     name="waymo-noConf-noDist-vid-yolo11n-bdd100k_night-noFreeze-AdamW-10e-lr0.001-lrf0.1",
    #     model="runs/detect/bdd100k_night-yolo11n-seed-test-0/weights/last.pt",
    #     data="waymo-noConf-noDist-vid.yaml",
    #     epochs=10,
    #     lr0=0.001,
    #     lrf=0.1,
    #     optimizer="AdamW",
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-mpdiou",
    #     model="runs/detect/bdd100k_night-yolo11n-spdconv-mpdiou/weights/last.pt",
    #     resume=True,
    #     iou_type="mpdiou",
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-bic-reduced-channel-sppfcsp",
    #     model="dlt-models/yolo11n-bic-reduced-channel-sppfcsp.yaml",
    # )

    # train_model(
    #     name="bdd100k_night-yolo11n-bic-reduced-channel-simam",
    #     model="dlt-models/yolo11n-bic-reduced-channel-simam.yaml",
    # )

    # train_model(
    #     name="bdd100k_night-yolo11n-bic-afr-skip-reduced-channel",
    #     model="dlt-models/yolo11n-bic-afr-skip-reduced-channel.yaml",
    # )

    # train_model(
    #     name="bdd100k_night-yolo11n-bic-reduced-channel-gsconv",
    #     model="dlt-models/yolo11n-bic-reduced-channel-gsconv.yaml",
    # )

    # train_model(
    #     name="bdd100k_night-yolo11n-bic-reduced-channel-rfac3k2",
    #     model="dlt-models/yolo11n-bic-reduced-channel-rfac3k2.yaml",
    # )

    train_model(
        name="bdd100k_night-yolo11n-bic-reduced-channel-repc3k2",
        model="dlt-models/yolo11n-bic-reduced-channel-repc3k2.yaml",
    )

    train_model(
        name="bdd100k_night-yolo11n-bic-reduced-channel-enhancedc3k2",
        model="dlt-models/yolo11n-bic-reduced-channel-enhancedc3k2.yaml",
    )

    train_model(
        name="bdd100k_night-yolo11n-bic-afr-reduced-channel",
        model="runs/detect/bdd100k_night-yolo11n-bic-afr-reduced-channel/weights/last.pt",
        resume=True,
    )
    train_model(
        name="bdd100k_night-yolo11n-bic-afr-skip-reduced-channel-full-mosaic",
        model="dlt-models/yolo11n-bic-afr-skip-reduced-channel.yaml",
        close_mosaic=0,
    )

    train_model(
        name="bdd100k_night-yolo11n-biformer",
        model="runs/detect/bdd100k_night-yolo11n-biformer/weights/last.pt",
        resume=True,
    )

    # train_model(
    #     name="carla_yolo-yolo11n-spdconv",
    #     model="runs/detect/carla_yolo-yolo11n-spdconv/weights/last.pt",
    #     resume=True,
    #     data="carla-yolo.yaml",
    # )
    # train_model(
    #     name="carla_yolo-yolo11n",
    #     model="yolo11n.yaml",
    #     data="carla-yolo.yaml",
    # )
