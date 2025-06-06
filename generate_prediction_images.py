import argparse
from ultralytics import YOLO
from train_conf import DEVICE
import os
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run qualitative comparison of distance and feature extraction models."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the model weights.", choices=["kitti-yolo", "waymo_dark"]
    )
    parser.add_argument("--image_name", type=str, required=True, help="Path to the image for testing.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU threshold for non-max suppression.")

    args = parser.parse_args()

    if args.dataset == "kitti-yolo":
        model = YOLO("weights/best_kitti_distance_01.pt")  # Adjust the path to your model weights
    elif args.dataset == "waymo_dark":
        model = YOLO("weights/best_waymo_distance_001.pt")  # Adjust the path to your model weights
    else:
        raise ValueError("Unsupported dataset. Choose either 'kitti-yolo' or 'waymo_dark'.")

    dataset = args.dataset
    if not os.path.exists(f"{dataset}_predictions"):
        os.makedirs(f"{dataset}_predictions")
    image_name = args.image_name

    if "*" in image_name:
        image_paths = glob.glob(f"../yolo-testing/datasets/{dataset}/images/val/{image_name}")
    else:
        image_paths = [f"../yolo-testing/datasets/{dataset}/images/val/{image_name}"]

    print(f"Using images: {image_paths}")

    results = model.predict(image_paths, conf=args.conf, iou=args.iou, device=DEVICE, max_dist=140)

    for i, r in enumerate(results):
        r.save(f"{dataset}_predictions/{image_paths[i].split('/')[-1].split('.')[0]}_predictions.jpg")
