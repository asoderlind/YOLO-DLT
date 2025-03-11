import os
import glob
from notebooks.utils import enhance_image
import cv2
from tqdm import tqdm

# Define paths
dataset_path = "../yolo-testing/datasets/exDark128-yolo/images"
target_train_folder = "../yolo-testing/datasets/exDark128-yolo/enhanced_images/train"
target_val_folder = "../yolo-testing/datasets/exDark128-yolo/enhanced_images/val"

# Create target directories if they don't exist
os.makedirs(target_train_folder, exist_ok=True)
os.makedirs(target_val_folder, exist_ok=True)

# Get raw image lists
raw_train_images = glob.glob(f"{dataset_path}/train/*.jpg")
raw_val_images = glob.glob(f"{dataset_path}/val/*.jpg")

print(f"Found {len(raw_train_images)} training images and {len(raw_val_images)} validation images.")

# Enhance training images with progress bar
print("Enhancing training images:")
for image in tqdm(raw_train_images, desc="Train", unit="img"):
    enhanced_image = enhance_image(image)
    cv2.imwrite(f"{target_train_folder}/{os.path.basename(image)}", enhanced_image)

# Enhance validation images with progress bar
print("Enhancing validation images:")
for image in tqdm(raw_val_images, desc="Validation", unit="img"):
    enhanced_image = enhance_image(image)
    cv2.imwrite(f"{target_val_folder}/{os.path.basename(image)}", enhanced_image)
