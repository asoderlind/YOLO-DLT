"""
This script sorts the label indices and removes gaps in the class indices
for YOLO format labels. For example, if the labels are: 0,1,2,3,4,5 but
the labels 3 and 4 are missing, the script will change the labels to:
0 -> 0, 1 -> 1, 2 -> 2, 5 -> 3.
"""

import os
import glob
import argparse


def fix_labels(dataset_path: str, dataset_name: str, dry_run: bool = False):
    """
    Fixes the labels in the dataset by removing gaps in the class indices.
    """

    labels_path = os.path.join(dataset_path, dataset_name, "labels")
    if not os.path.exists(labels_path):
        print(f"Labels path {labels_path} does not exist.")
        return

    # Get all label files
    label_files = glob.glob(os.path.join(labels_path, "train", "*.txt")) + glob.glob(
        os.path.join(labels_path, "val", "*.txt")
    )

    print(f"Found {len(label_files)} label files")

    classes = set()

    # Loop through each label file
    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Loop through each line and create the mapping
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id not in classes:
                classes.add(cls_id)

    # Create a mapping from old class indices to new class indices
    sorted_classes = sorted(classes)
    class_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_classes)}
    print(f"Class mapping: {class_mapping}")

    # Loop through each label file again to update the labels
    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id in class_mapping:
                new_cls_id = class_mapping[cls_id]
                new_line = f"{new_cls_id} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)

        # Write the updated labels back to the file
        if not dry_run:
            with open(label_file, "w") as f:
                f.writelines(new_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix YOLO labels by removing gaps in class indices.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    dry_run = args.dry_run

    fix_labels(dataset_path, dataset_name, dry_run)
