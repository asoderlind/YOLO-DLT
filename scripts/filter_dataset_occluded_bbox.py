"""
Filter dataset by removing images with occluded bounding boxes above a certain threshold.
This script processes the dataset and removes annotations with occluded bounding boxes.
"""

import os
import shutil
import tqdm
import glob
from PIL import Image
from utils import filter_occluded_boxes


def main(dataset_path: str, dataset_name: str, occlusion_threshold: float, image_format="jpg"):
    """
    Filters and creates new dataset with filtered images and labels.
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
    print(label_files[:1])

    output_dataset_name = dataset_name + f"-occ{occlusion_threshold}"
    output_dataset_path = os.path.join(dataset_path, output_dataset_name)
    print(f"Output dataset path: {output_dataset_path}")
    os.makedirs(output_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(output_dataset_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_path, "labels", "val"), exist_ok=True)

    loader = tqdm.tqdm(total=len(label_files), desc="Processing labels", unit="file")

    for label_path in label_files:
        loader.update(1)
        img_path = label_path.replace("labels", "images").replace(".txt", f".{image_format}")
        height, width = Image.open(img_path).size

        annotations = filter_occluded_boxes(label_path, height, width, occlusion_threshold=occlusion_threshold)
        new_lines = []
        for annotation in annotations:
            new_lines.append(
                f"{int(annotation[0])} {annotation[1]:.4f} {annotation[2]:.4f} {annotation[3]:.4f} {annotation[4]:.4f} {annotation[5]:.4f}\n"
            )

        new_label_path = label_path.replace(dataset_name, output_dataset_name)
        new_image_path = img_path.replace(dataset_name, output_dataset_name)

        with open(new_label_path, "w") as f:
            f.writelines(new_lines)

        shutil.copy(img_path, new_image_path)


if __name__ == "__main__":
    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser(description="Filter dataset with occluded bounding boxes")
    parser.add_argument("--dataset-path", type=str, help="Dataset path", default="../../yolo-testing/datasets")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name")
    parser.add_argument("--occlusion-threshold", type=float, default=0.5, help="Occlusion threshold")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    occlusion_threshold = args.occlusion_threshold

    print(f"Dataset path: {dataset_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Occlusion threshold: {occlusion_threshold}")

    main(dataset_path, dataset_name, occlusion_threshold)
