import os
import time
from datetime import datetime

from ultralytics import YOLO


# Create a function to measure training time
def train_with_timing(
    model_path,
    data_path,
    batch_size,
    epochs,
    device,
    name,
    use_fe=False,
    augment=True,
    optimizer="auto",
    pretrained=False,
    iou_type="iou",
):
    # Load the model
    model = YOLO(model_path)

    # Record start time
    start_time = time.time()

    # Train the model
    results = model.train(
        data=data_path,
        batch=batch_size,
        epochs=epochs,
        device=device,
        pretrained=pretrained,
        optimizer=optimizer,
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
