import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_statistics(dataset_path, image_formats=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]) -> dict:
    statistics = {
        "total": -1,
        "training": -1,
        "validation": -1,
        "test": -1,
        "train/val split": -1,
        "total_background": -1,
        "bg_training": -1,
        "bg_validation": -1,
        "num_classes": -1,
    }
    training_images = []
    for image_format in image_formats:
        training_images += glob.glob(f"{dataset_path}/images/train/*.{image_format}")

    validation_images = []
    for image_format in image_formats:
        validation_images += glob.glob(f"{dataset_path}/images/val/*.{image_format}")

    test_images = []
    for image_format in image_formats:
        test_images += glob.glob(f"{dataset_path}/images/test/*.{image_format}")

    training_labels = glob.glob(f"{dataset_path}/labels/train/*.txt")
    validation_labels = glob.glob(f"{dataset_path}/labels/val/*.txt")
    test_labels = glob.glob(f"{dataset_path}/labels/test/*.txt")

    total_images = len(training_images) + len(validation_images) + len(test_images)
    total_labels = len(training_labels) + len(validation_labels) + len(test_labels)

    if total_images != total_labels:
        print(f"Warning: total images ({total_images}) and total labels ({total_labels}) do not match")

    total_empty_training_labels = 0
    for label in training_labels:
        with open(label, "r") as f:
            if not f.read():
                total_empty_training_labels += 1

    total_empty_validation_labels = 0
    for label in validation_labels:
        with open(label, "r") as f:
            if not f.read():
                total_empty_validation_labels += 1

    total_empty_test_labels = 0
    for label in test_labels:
        with open(label, "r") as f:
            if not f.read():
                total_empty_test_labels += 1

    if len(training_images) == 0:
        raise Exception("No training images found")

    for label in training_labels:
        # Count the number of classes
        with open(label, "r") as f:
            for line in f:
                class_id = line.split()[0]
                if int(class_id) > statistics["num_classes"]:
                    statistics["num_classes"] = int(class_id)

    statistics["total"] = total_images
    statistics["training"] = len(training_images)
    statistics["validation"] = len(validation_images)
    statistics["test"] = len(test_images)
    statistics["train/val split"] = (len(validation_images) / len(training_images)) * 100
    statistics["total_background"] = (
        total_empty_training_labels + total_empty_validation_labels + total_empty_test_labels
    )
    statistics["bg_training"] = total_empty_training_labels
    statistics["bg_validation"] = total_empty_validation_labels
    statistics["bg_test"] = total_empty_test_labels
    if "Australia" in dataset_path or "China" in dataset_path:
        statistics["num_classes"] += 0
    else:
        statistics["num_classes"] += 1
    return statistics


def distance_distribution_histogram(dataset_path):
    # Set the maximum distance based on the dataset
    if "kitti" in dataset_path.lower():
        max_distance = 150
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
    elif "carla" in dataset_path.lower():
        max_distance = 100
        classes = [0, 1, 2, 3, 4, 5]
    elif "waymo-noconf" in dataset_path.lower():
        max_distance = 85
        classes = [1, 2, 3, 4]
    else:
        raise Exception("Unknown dataset")

    def parse_label_file(file):
        with open(file, "r") as f:
            distances = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        class_id = int(parts[0])
                        dist = float(parts[-1])  # Assume distance is the last value on the line
                        if class_id in classes and dist > 0:
                            distances.append(dist)
                    except ValueError:
                        continue
            return distances

    def bin_distances(distances, bin_size=5, max_distance=90):
        bins = list(range(0, max_distance + bin_size, bin_size))
        counts, _ = np.histogram(distances, bins=bins)
        return counts, bins[:-1]

    def load_split(split):
        label_files = glob.glob(f"{dataset_path}/labels/{split}/*.txt")
        distances = []
        for file in label_files:
            distances.extend(parse_label_file(file))
        return distances

    train_distances = load_split("train")
    val_distances = load_split("val")

    train_distances = [dist * max_distance for dist in train_distances]
    val_distances = [dist * max_distance for dist in val_distances]

    train_counts, bins = bin_distances(train_distances)
    val_counts, _ = bin_distances(val_distances)

    bar_width = 2
    x = np.array(bins) + bar_width + 0.5
    plt.bar(
        x - bar_width / 2, train_counts, width=bar_width, color="#DAE8FC", edgecolor="#6C8EBF", label="Training dataset"
    )
    plt.bar(
        x + bar_width / 2, val_counts, width=bar_width, color="#F8CECC", edgecolor="#B85450", label="Validation dataset"
    )
    plt.xlabel("Groundtruth distance in meters")
    plt.ylabel("Number of boxes")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(np.arange(0, 95, 5))
    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser(description="Run statistics on dataset")

parser.add_argument("name", type=str, default="bdd100k_night", help="Name of the dataset to run statistics on")

args = parser.parse_args()

stats = get_statistics(f"../../yolo-testing/datasets/{args.name}")

for key, value in stats.items():
    print(f"{key}: {value}")

if args.name.lower() in ["kitti-yolo", "carla-yolo", "waymo-noconf"]:
    distance_distribution_histogram(f"../../yolo-testing/datasets/{args.name}")
