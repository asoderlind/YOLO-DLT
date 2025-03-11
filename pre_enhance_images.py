import os
import glob
from notebooks.utils import enhance_image
import cv2
from tqdm import tqdm

dataset = "exDark-yolo"

# Define paths
dataset_path = f"../yolo-testing/datasets/{dataset}/images"
target_train_folder = f"../yolo-testing/datasets/{dataset}/images/train_enhanced"
target_val_folder = f"../yolo-testing/datasets/{dataset}/images/val_enhanced"

formats = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]

print(dataset_path)

# Get raw image lists
raw_train_images = []
raw_val_images = []
for format in formats:
    raw_train_images += glob.glob(f"{dataset_path}/train/*.{format}")
    raw_val_images += glob.glob(f"{dataset_path}/val/*.{format}")

print(f"Found {len(raw_train_images)} training images and {len(raw_val_images)} validation images.")

# Create target directories if they don't exist
os.makedirs(target_train_folder, exist_ok=True)
os.makedirs(target_val_folder, exist_ok=True)

# Enhance training images with progress bar
print("Enhancing training images:")
for image in tqdm(raw_train_images, desc="Train", unit="img"):
    path = f"{target_train_folder}/{os.path.basename(image)}"
    if os.path.exists(path):
        print(f"Skipping {path}")
        continue
    enhanced_image = enhance_image(image)
    if enhanced_image is not None:
        cv2.imwrite(path, enhanced_image)

# Enhance validation images with progress bar
print("Enhancing validation images:")
for image in tqdm(raw_val_images, desc="Validation", unit="img"):
    path = f"{target_val_folder}/{os.path.basename(image)}"
    if os.path.exists(path):
        print(f"Skipping {path}")
        continue
    enhanced_image = enhance_image(image)
    if enhanced_image is not None:
        cv2.imwrite(path, enhanced_image)
