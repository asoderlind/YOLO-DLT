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
    WAYMO_TEMPORAL_BASE = "runs/detect/waymo_dark-yolo11n-temporal-base/weights/last.pt"
    BDD100K_NIGHT_TEMPORAL_BASE = "runs/detect/bdd100k_night-yolo11n-temporal-base/weights/last.pt"
    # All runs have 16 nbs, 10% warmup, 4.0 gain, freeze true, cos_lr,

    train_model_v(
        name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-fixed-2",
        model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
        model_load_path=WAYMO_TEMPORAL_BASE,
        data="waymo_dark.yaml",
        gframe=16,
        batch=1,
        temporal_freeze=True,
        temporal_cls=4.0,
        epochs=40,
        warmup_epochs=4,
        cos_lr=True,
        dataset_type="temporal",
        lr0=0.005,
        nbs=16,
    )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-0.5_tau-fixed",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-12.5_attn_scale-fixed",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-12.5-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-5.0_attn_scale-fixed",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-5.0-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-2.5_attn_scale",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-2.5-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-0.25_attn_scale",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-0.25-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # frame ablations

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-mosaic-fixed",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=8,
    #     gframe=8,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_stride=1,  # Using temporal stride of 1
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-mosaic-fixed",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=8,
    #     gframe=8,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_stride=2,  # Using temporal stride of 1
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_lframe-mosaic-fixed",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_stride=1,  # Using temporal stride of 1
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-fixed-preload_vid_cls-full_freeze",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-fixed",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # model_temp = YOLO(
    #     "runs/detect/waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-0.001_lr0-0.75_nms-temporal_dl-mosaic/weights/last.pt"
    # )
    # model_temp.val(
    #     name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-0.001_lr0-0.75_nms-temporal_dl-mosaic_val",
    #     data="waymo_dark.yaml",
    #     batch=1,
    #     device="cuda",
    #     gframe=16,
    #     dataset_type="temporal",
    # )

    # model = YOLO(WAYMO_TEMPORAL_BASE)
    # model.val(
    #     name="waymo_dark-yolo11n-temporal-base_val",
    #     data="waymo_dark.yaml",
    #     batch=1,
    #     device="cuda",
    #     gframe=16,
    #     dataset_type="temporal",
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-temporal-base-freeze-run",
    #     model=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="default",
    #     nbs=16,
    #     lr0=0.005,
    #     # temporal_freeze=True,
    #     box=0.0,  # (float) box loss gain
    #     cls=0.0,  # (float) cls loss gain (scale with pixels)
    #     dfl=0.0,  # (float) dfl loss gain
    #     lambda_c=0.0,  # (float) loss weight for class label smoothing
    #     temporal_cls=0.0,  # (float) temporal class loss gain
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-0.001_lr0-0.75_nms-temporal_dl-mosaic",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=8,
    #     gframe=8,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_stride=2,  # Using temporal stride of 2
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-0.005_lr0-0.75_nms-temporal_dl-mosaic",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=8,
    #     gframe=8,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-0.005_lr0-0.75_nms-temporal_dl-mosaic",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=8,
    #     gframe=8,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale-0.5_tau",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # lr ablation runs

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.0025_lr0-0.75_nms-temporal_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.0025,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.001_lr0-0.75_nms-temporal_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.001,
    #     nbs=16,
    # )

    # temporal gain ablation runs

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-2.0_gain",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=2.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-8.0_gain",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=8.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # model = YOLO(
    #     "runs/detect/waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale/weights/last.pt"
    # )

    # model_base = YOLO(WAYMO_TEMPORAL_BASE)

    # model.val(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale_val",
    #     data="waymo_dark.yaml",
    #     batch=1,
    #     device="cuda",
    #     gframe=16,
    #     dataset_type="temporal",
    # )

    # model.val(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale_val_car",
    #     data="waymo_dark.yaml",
    #     batch=1,
    #     device="cuda",
    #     gframe=16,
    #     dataset_type="temporal",
    #     classes=0,  # Only validate on car class
    # )

    # model_base.val(
    #     name="waymo_dark-yolo11n-temporal-base_val",
    #     data="waymo_dark.yaml",
    #     batch=1,
    #     device="cuda",
    #     gframe=16,
    #     dataset_type="temporal",
    # )

    # model_base.val(
    #     name="waymo_dark-yolo11n-temporal-base_val_car",
    #     data="waymo_dark.yaml",
    #     batch=1,
    #     device="cuda",
    #     gframe=16,
    #     dataset_type="temporal",
    #     classes=0,  # Only validate on car class
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_lframe-0.005_lr0-0.75_nms-temporal_dl-mosaic",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_lframe-2_tmp_stride-0.005_lr0-0.75_nms-temporal_dl-mosaic",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_stride=2,  # Using temporal stride of 2
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-8_lframe-8_gframe-2_tmp_stride-0.005_lr0-0.75_nms-temporal_dl-mosaic",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=8,
    #     gframe=8,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_stride=2,  # Using temporal stride of 2
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_lframe-0.005_lr0-0.75_nms-temporal_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     mosaic=0.0,  # No mosaic augmentation
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_lframe-2_tmp_stride-0.005_lr0-0.75_nms-temporal_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     lframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_stride=2,  # Using temporal stride of 2
    #     mosaic=0.0,  # No mosaic augmentation
    # )

    # ## Attention scale ablations

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-0.25_attn_scale",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-0.25-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-2.5_attn_scale",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-2.5-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-5.0_attn_scale",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-5.0-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-12.5_attn_scale",
    #     model="dlt-models/yolo11n-temporal-nms-0.75-12.5-scale.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.75_nms-temporal_dl-25.0_attn_scale",
    #     model="dlt-models/yolo11n-temporal-nms-0.75.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # Unfinished runs
    ##################################################################################

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.05_thresh-temporal_dl-4.0_gain",
    #     model="dlt-models/yolo11n-temporal-thresh-0.05.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-40_e-16_gframe-0.005_lr0-0.07_thresh-temporal_dl-4.0_gain",
    #     model="dlt-models/yolo11n-temporal-thresh-0.07.yaml",
    #     model_load_path=WAYMO_TEMPORAL_BASE,
    #     data="waymo_dark.yaml",
    #     gframe=16,
    #     batch=1,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    # )

    ########################################################3
    # train_model_v(
    #     name="waymo_dark-yolo11n-temporal-base",
    #     model="yolo11n.yaml",
    #     model_load_path="runs/detect/bdd100k_night-yolo11n-temporal-base/weights/last.pt",
    #     epochs=20,
    #     data="waymo_dark.yaml",
    #     dataset_type="default",
    #     lr0=0.001,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_40_e-8_we-16_gframe-0.0025_lr0_rescue_zone-default_dl-4.0_gain",
    #     model="dlt-models/yolo11n-temporal-nms-0.4.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     gframe=16,
    #     batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     dataset_type="default",
    #     lr0=0.0025,
    #     nbs=16,
    #     temporal_freeze=True,
    #     temporal_cls=4.0,
    # )

    # train_model_v(
    #     name="bdd100k_night-yolo11n-temporal-base",
    #     model="yolo11n.yaml",
    #     model_load_path="",
    #     batch=16,
    #     epochs=200,
    #     data="bdd100k_night.yaml",
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.0025_lr0_0.4_nms-default_dl-0.5_gain",
    #     model="dlt-models/yolo11n-temporal-nms-0.4.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     gframe=16,
    #     batch=16,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     dataset_type="default",
    #     lr0=0.0025,
    #     nbs=16,
    #     temporal_freeze=True,
    #     temporal_cls=0.5,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.0025_lr0_0.4_nms-temporal_dl-8.0_gain",
    #     model="dlt-models/yolo11n-temporal-nms-0.4.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     gframe=16,
    #     batch=1,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     dataset_type="temporal",
    #     lr0=0.0025,
    #     nbs=16,
    #     temporal_freeze=True,
    #     temporal_cls=8.0,
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.4_nms-temporal_dl-2.5_gain",
    #     model="dlt-models/yolo11n-temporal-nms-0.4.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     gframe=16,
    #     batch=1,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     dataset_type="temporal",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    #     temporal_cls=2.5,
    #     resume=True,
    # )

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
    #     dataset_type="temporal",
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.4_nms-temporal_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.4.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     gframe=16,
    #     batch=1,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    #     dataset_type="temporal",
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.02_thresh-temporal_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.02.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     gframe=16,
    #     batch=1,
    #     epochs=80,
    #     warmup_epochs=8,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    #     dataset_type="temporal",
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_40_e-8_we-16_gframe-0.005_lr0_0.04_nms_default_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.04.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     batch=16,
    #     epochs=40,
    #     warmup_epochs=4,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    #     dataset_type="default",
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_60_e-8_we-16_gframe-0.005_lr0_0.04_nms_default_dl",
    #     model="dlt-models/yolo11n-temporal-nms-0.04.yaml",
    #     model_load_path="runs/detect/waymo_dark-yolo11n3/weights/last.pt",
    #     batch=16,
    #     epochs=60,
    #     warmup_epochs=6,
    #     cos_lr=True,
    #     data="waymo_dark.yaml",
    #     lr0=0.005,
    #     nbs=16,
    #     temporal_freeze=True,
    #     dataset_type="default",
    # )

    # train_model_v(
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.4_nms-0.4_nms_val-default_dl",
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
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.02_thresh-default_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.02.yaml",
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
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.05_thresh-default_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.05.yaml",
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
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.06_thresh-default_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.06.yaml",
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
    #     name="waymo_dark-yolo11n-16_nbs_80_e-8_we-16_gframe-0.005_lr0_0.07_thresh-default_dl",
    #     model="dlt-models/yolo11n-temporal-thresh-0.07.yaml",
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
