import os
import glob
from tqdm import tqdm
from notebooks.utils import get_dln_model, enhance_image
import cv2
import argparse


def main(dataset: str):
    # Define paths
    dataset_path = f"../yolo-testing/datasets/{dataset}/images"
    target_train_folder = f"../yolo-testing/datasets/{dataset}/images/train_enhanced"
    target_val_folder = f"../yolo-testing/datasets/{dataset}/images/val_enhanced"

    formats = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]

    # Get raw image lists
    raw_train_images = []
    raw_val_images = []
    for format in formats:
        raw_train_images += glob.glob(f"{dataset_path}/train/*.{format}")
        raw_val_images += glob.glob(f"{dataset_path}/val/*.{format}")

    print(f"Found {len(raw_train_images)} training images and {len(raw_val_images)} validation images.")

    # Get enhanced image lists
    enhanced_train_images = glob.glob(f"{target_train_folder}/*")
    enhanced_val_images = glob.glob(f"{target_val_folder}/*")
    print(
        f"Found {len(enhanced_train_images)} enhanced training images and {len(enhanced_val_images)} enhanced validation images."
    )

    # Check if the number of enhanced images matches the number of raw images
    if len(enhanced_train_images) != len(raw_train_images):
        print(
            f"Warning: {len(raw_train_images) - len(enhanced_train_images)} training images are missing in the enhanced folder."
        )

    if len(enhanced_val_images) != len(raw_val_images):
        print(
            f"Warning: {len(raw_val_images) - len(enhanced_val_images)} validation images are missing in the enhanced folder."
        )

    # Only keep the images that are not already enhanced
    raw_train_images = [img for img in raw_train_images if img not in enhanced_train_images]
    raw_val_images = [img for img in raw_val_images if img not in enhanced_val_images]

    print(
        f"After filtering, {len(raw_train_images)} training images and {len(raw_val_images)} validation images remain."
    )

    prompt = input(
        f"Do you want to enhance {len(raw_train_images)} training images and {len(raw_val_images)} validation images? (y/n): "
    )
    if prompt.lower() != "y":
        print("Exiting without enhancing images.")
        exit()

    # Create target directories if they don't exist
    if not os.path.exists(target_train_folder):
        os.makedirs(target_train_folder)
    if not os.path.exists(target_val_folder):
        os.makedirs(target_val_folder, exist_ok=True)

    dln = get_dln_model()

    # Enhance training images with progress bar
    print("Enhancing training images:")
    for image in tqdm(raw_train_images, desc="Train", unit="img"):
        path = f"{target_train_folder}/{os.path.basename(image)}"
        if os.path.exists(path):
            continue
        enhanced_image = enhance_image(image, dln)
        if enhanced_image is not None:
            cv2.imwrite(path, enhanced_image)

    # Enhance validation images with progress bar
    print("Enhancing validation images:")
    for image in tqdm(raw_val_images, desc="Validation", unit="img"):
        path = f"{target_val_folder}/{os.path.basename(image)}"
        if os.path.exists(path):
            continue
        enhanced_image = enhance_image(image, dln)
        if enhanced_image is not None:
            cv2.imwrite(path, enhanced_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance images using DLN model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bdd100k_night-dln",
        help="Dataset name (default: bdd100k_night-dln)",
    )
    args = parser.parse_args()
    main(args.dataset)
