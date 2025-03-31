import os
import glob
import argparse

BASE_PATH = "/home/phoawb/repos/yolo-testing/datasets"

def clean_yolo_dataset(dataset_name: str, train_set: str, dry_run=False):
    """Removes unmatched labels or images in a YOLO dataset."""

    labels_dir = f"{BASE_PATH}/{dataset_name}/labels/{train_set}"
    images_dir = f"{BASE_PATH}/{dataset_name}/images/{train_set}"
    
    # Get lists of filenames (without extensions)
    print("labels_dir: ", labels_dir)
    label_files = glob.glob(f"{labels_dir}/*.txt")
    print(f"{len(label_files)} labels")

    print("images_dir: ", images_dir)
    image_files = glob.glob(f"{images_dir}/*.png")
    print(f"{len(image_files)} images")

    label_files = {os.path.basename(f)[:-4] for f in label_files}
    image_files = {os.path.basename(f)[:-4] for f in image_files}

    # Find unmatched labels
    unmatched_labels = label_files - image_files
    # Find unmatched images
    unmatched_images = image_files - label_files

    print(f"Unmatched labels: {len(unmatched_labels)}")
    print(f"Unmatched images: {len(unmatched_images)}")

    # Remove unmatched labels
    for label in unmatched_labels:
        label_path = os.path.join(labels_dir, label + ".txt")
        if dry_run:
            print(f"[DRY RUN] Would delete: {label_path}")
        else:
            os.remove(label_path)
            print(f"Deleted: {label_path}")

    # Remove unmatched images
    for image in unmatched_images:
        for ext in (".jpg", ".png"):
            image_path = os.path.join(images_dir, image + ext)
            if os.path.exists(image_path):  # Ensure the file exists before trying to delete
                if dry_run:
                    print(f"[DRY RUN] Would delete: {image_path}")
                else:
                    os.remove(image_path)
                    print(f"Deleted: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean YOLO dataset by removing unmatched labels or images.")
    parser.add_argument("name", type=str, help="Name of dataset")
    parser.add_argument("train_set", type=str, help="The train set train/val/test")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be deleted without actually deleting files")

    args = parser.parse_args()
    clean_yolo_dataset(args.name, args.train_set, dry_run=args.dry_run)
