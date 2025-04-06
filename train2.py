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
    # train_model(name="bdd100k_night-yolo11n-cl", model="runs/detect/bdd100k_night-yolo11n-cl/weights/last.pt", resume=True)
    # train_model(name="bdd100k_night-yolo11n-simsppf", model="dlt-models/yolo11n-SimSPPF.yaml")
    # train_model(name="bdd100k_night-yolo11n-se", model="runs/detect/bdd100k_night-yolo11n-se/weights/last.pt", resume=True)
    # train_model(name="bdd100k_night-yolo11n-eca", model="runs/detect/bdd100k_night-yolo11n-eca/weights/last.pt", resume=True)
    # train_model(name="bdd100k_night-yolo11n-simam", model="dlt-models/yolo11n-SimAM.yaml")
    # train_model(name="bdd100k_night-yolo11n-se-skip", model="runs/detect/bdd100k_night-yolo11n-se-skip/weights/last.pt", resume=True)
    # train_model(name="bdd100k_night-yolo11n-eca-skip", model="dlt-models/yolo11n-ECA-skip.yaml")
    # train_model(name="bdd100k_night-yolo11n-simam-skip", model="dlt-models/yolo11n-SimAM-skip.yaml")
    # train_model(name="bdd100k_night-yolo11n-efficient-fem", model="dlt-models/yolo11n-EfficientFEM.yaml") # TODO: Figure out why this is so slow
    # train_model(name="bdd100k_night-yolo11n-fem-2", model="dlt-models/yolo11n-FEM2.yaml")

    # test some IOU types
    # iou_type: ciou # (str) iou loss type, i.e. "iou", "giou", "diou", "ciou", "nwd", "wiou", "wiou1", "wiou2", "wiou3", "siou", "ciou+nwd (80/20)", "eiou", "focal-eiou", "isiou", "thiou"
    # train_model(name="bdd100k_night-yolo11n-ciou", model="dlt-models/yolo11n.yaml", iou_type="ciou")
    # train_model(name="bdd100k_night-yolo11n-giou", model="dlt-models/yolo11n.yaml", iou_type="giou")
    # train_model(name="bdd100k_night-yolo11n-diou", model="dlt-models/yolo11n.yaml", iou_type="diou")
    # train_model(name="bdd100k_night-yolo11n-nwd", model="runs/detect/bdd100k_night-yolo11n-nwd/weights/last.pt", resume=True, iou_type="nwd")
    # train_model(name="bdd100k_night-yolo11n-wiou", model="yolo11n.yaml", iou_type="wiou1")
    # train_model(name="bdd100k_night-yolo11n-wiou2", model="yolo11n.yaml", iou_type="wiou2")
    # train_model(name="bdd100k_night-yolo11n-wiou3", model="runs/detect/bdd100k_night-yolo11n-wiou3/weights/last.pt", resume=True, iou_type="wiou3")
    # train_model(name="bdd100k_night-yolo11n-EfficientFEM", model="runs/detect/bdd100k_night-yolo11n-EfficientFEM/weights/last.pt", resume=True)
    # train_model(name="bdd100k_night-yolo11n-EfficientSPDConv", model="dlt-models/yolo11n-EfficientSPDConv.yaml")
    # train_model(name="bdd100k_night-yolo11n-SPPFFEM", model="dlt-models/yolo11n-SPPFFEM.y
    # train_model(name="bdd100k_night-yolo11n-thiou", model="yolo11n.yaml", iou_type="thiou")
    # train_model(name="bdd100k_night-yolo11n-eiou", model="yolo11n.yaml", iou_type="eiou")
    # train_model(name="bdd100k_night-yolo11n-dwspdconv", model="dlt-models/yolo11n-DWSPDConv.yaml")
    # train_model(name="bdd100k_night-yolo11n-SPDConv2", model="dlt-models/yolo11n-SPDConv-2.yaml")
    # train_model(
    #     name="bdd100k_night-yolo11n-SPDConv3",
    #     model="runs/detect/bdd100k_night-yolo11n-SPDConv3/weights/last.pt",
    #     resume=True,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-repc3k2",
    #     model="runs/detect/bdd100k_night-yolo11n-repc3k2/weights/last.pt",
    #     resume=True,
    # )
    # train_model(name="bdd100k_night-yolo11n-rfac3k2", model="dlt-models/yolo11n-RFAC3k2.yaml")
    # train_model(
    #     name="bdd100k_night-yolo11n-progressive-spdconv",
    #     model="runs/detect/bdd100k_night-yolo11n-progressive-spdconv/weights/last.pt",
    #     resume=True,
    # )
    # These are not critical
    # train_model(name="bdd100k_night-yolo11n-focal-eiou", model="yolo11n.yaml", iou_type="focal-eiou")
    # train_model(name="bdd100k_night-yolo11n-wiou3", model="yolo11n.yaml", iou_type="wiou3")

    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-rfac3k2",
    #     model="runs/detect/bdd100k_night-yolo11n-spdconv-rfac3k2/weights/last.pt",
    #     resume=True,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-repc3k2",
    #     model="runs/detect/bdd100k_night-yolo11n-spdconv-repc3k2/weights/last.pt",
    #     resume=True,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-fa",
    #     model="runs/detect/bdd100k_night-yolo11n-spdconv-fa/weights/last.pt",
    #     resume=True,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-full-mosaic",
    #     model="dlt-models/yolo11n-SPDConv-3.yaml",
    #     close_mosaic=0,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-cl",
    #     model="runs/detect/bdd100k_night-yolo11n-spdconv-cl/weights/last.pt",
    #     resume=True,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-augs",
    #     model="dlt-models/yolo11n-SPDConv-3.yaml",
    #     degrees=15,
    #     shear=10,
    #     crop_fraction=0.75,
    #     hsv_v=0.6,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-SPDConv-3-thiou",
    #     model="dlt-models/yolo11n-SPDConv-3.yaml",
    #     iou_type="thiou",
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-SPDConv-3-thiou-gain",
    #     model="runs/detect/bdd100k_night-yolo11n-SPDConv-3-thiou-gain/weights/last.pt",
    #     resume=True,
    #     iou_type="thiou",
    #     box=9.0,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-4",
    #     model="dlt-models/yolo11n-spdconv-4.yaml",
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-partial-rfac3k2",
    #     model="runs/detect/bdd100k_night-yolo11n-spdconv-partial-rfac3k2/weights/last.pt",
    #     resume=True,
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-final-1",
    #     model="runs/detect/bdd100k_night-yolo11n-final-1/weights/last.pt",
    #     resume=True,
    #     iou_type="thiou",
    # )
    # train_model(
    #     name="bdd100k_night-yolo11n-gc+newconv",
    #     model="dlt-models/yolo11n-GC+NewConv.yaml",
    # )

    # train_model(
    #     name="bdd100k_night-yolo11n-final-ciou",
    #     model="runs/detect/bdd100k_night-yolo11n-final-ciou/weights/last.pt",
    #     resume=True,
    # )

    # train_model(
    #     name="bdd100k_night-yolo11n-spdconv-fa-cl",
    #     model="runs/detect/bdd100k_night-yolo11n-spdconv-fa-cl/weights/last.pt",
    #     resume=True,
    # )
    train_model(
        name="bdd100k-yolo11n-spdconv-3",
        model="dlt-models/yolo11n-SPDConv-3.yaml",
        data="bdd100k.yaml",
    )
    train_model(
        name="bdd100k-yolo11n-spdconv-rfac3k2",
        model="dlt-models/yolo11n-spdconv-rfac3k2.yaml",
        data="bdd100k.yaml",
    )

    train_model(
        name="bdd100k_night-yolo11n-siou",
        model="runs/detect/bdd100k_night-yolo11n-siou/weights/last.pt",
        resume=True,
        iou_type="siou",
    )
    train_model(name="bdd100k_night-yolo11n-isiou", model="yolo11n.yaml", iou_type="isiou")
    train_model(name="bdd100k_night-yolo11n-ciou+nwd", model="yolo11n.yaml", iou_type="ciou+nwd")
    train_model(name="bdd100k_night-yolo11n-nwd", model="yolo11n.yaml", iou_type="nwd")
