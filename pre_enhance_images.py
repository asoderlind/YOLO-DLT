import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics.DLN.model import DLN
import torchvision.transforms as transforms
import torch

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

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
model = DLN()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("ultralytics/DLN/DLN_finetune_LOL.pth", map_location=lambda storage, loc: storage))
model.eval()
model.to(device)


def enhance_image(filename, dln_chpt="ultralytics/DLN/DLN_finetune_LOL.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    trans = transforms.ToTensor()
    torch.manual_seed(0)
    with torch.no_grad():
        test_image_in = Image.open(filename).convert("RGB")
        test_image_in_tensor = trans(test_image_in)
        test_image_in_tensor = test_image_in_tensor.unsqueeze(0)
        test_image_in_tensor = test_image_in_tensor.to(device)  # shape [1, 3, 720, 1280]
        prediction = model(test_image_in_tensor)  # shape [1, 3, 720, 1280]
        prediction = prediction.data[0].cpu().numpy().transpose((1, 2, 0))
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)  # shape [720, 1280, 3]
    return np.uint8(prediction)


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
