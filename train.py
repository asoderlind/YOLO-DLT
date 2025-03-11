import os
import time
from datetime import datetime
from argparse import ArgumentParser

from ultralytics import YOLO


# Create a function to measure training time
def train_with_timing(args):
    model_path = args.model
    data_path = args.data
    epochs = args.epochs
    device = args.device
    name = args.tag
    use_fe = args.use_fe
    augment = args.augment
    pretrained = args.pretrained
    iou_type = args.iou_type

    # Load the model
    model = YOLO(model_path)

    # Record start time
    start_time = time.time()

    # Build name
    dataset_name = os.path.basename(data_path).split(".")[0]
    _fe = "-fe" if use_fe else ""
    _model = model_path.split(".")[0]
    name = f"{dataset_name}-{_model}-{_fe}-{name}"

    # Train the model
    results = model.train(
        data=data_path,
        batch=16,
        epochs=epochs,
        device=device,
        pretrained=pretrained,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.9,
        use_fe=use_fe,
        augment=augment,
        name=name,
        iou_type=iou_type,
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Format time as hours, minutes, seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    output_path = f"runs/detect/{name}/"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = os.path.join(output_path, "timing.txt")

    # Save timing information to file
    with open(output_file, "w") as f:
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total training time: {formatted_time}\n")
        f.write(f"Total seconds: {elapsed_time:.2f}\n")

    print(f"\nTraining completed in {formatted_time}")
    print(f"Timing information saved to: {output_path}")

    return results


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--param_path", type=str, default="default.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.yaml")
    parser.add_argument("--data", type=str, default="exDark128-yolo.yaml")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tag", type=str, default="noName")
    parser.add_argument("--use_fe", type=bool, default=False)
    parser.add_argument("--augment", type=bool, default=True)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--iou_type", type=str, default="ciou")
    args = parser.parse_args()
    train_with_timing(args)


if __name__ == "__main__":
    main()
