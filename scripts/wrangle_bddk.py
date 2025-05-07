"""
Example usage:
python wrangle_bddk.py ~/repos/yolo-testing/datasets/bdd100k_images_100k ~/repos/yolo-testing/datasets/bdd100k_labels ~/repos/yolo-testing/datasets/bdd100k-night-yolo --timeofdays "night"
"""

import argparse
import os

import glob
import shutil
import json
import tqdm

IMG_WIDTH, IMG_HEIGHT = 1280, 720

cls2id = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "bus": 3,
    "truck": 4,
    "bike": 5,
    "motor": 6,
    "tl_green": 7,
    "tl_red": 8,
    "tl_yellow": 9,
    "tl_none": 10,
    "traffic_sign": 11,
    "train": 12,
}


def num_frame_statistics(labels):
    """
    Get the number of frames in the labels.
    """
    stats = {}
    for label in labels:
        nf = len(label["frames"])  # Number of frames in the label
        stats[nf] = stats.get(nf, 0) + 1
    return stats


def convert_labels_to_yolo(images_dir: str, labels: list, output_dir: str, timeofdays=[], save=False):
    # Filter labels based on time of day
    if len(timeofdays) > 0:
        print("Filtering labels based on time of day ...")
        print(f"Time of day: {timeofdays}")
        labels = [label for label in labels if label["attributes"]["timeofday"] in timeofdays]
        print(f"Found {len(labels)} labels after filtering")
    loader = tqdm.tqdm(labels, desc="Converting labels to YOLO format", unit="file")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_images_dir = output_dir.replace("labels", "images")
    os.makedirs(output_images_dir, exist_ok=True)

    # Convert labels to YOLO format
    for label in loader:
        name = label["name"]
        source_image = f"{images_dir}/{name}.jpg"
        out_file = f"{output_dir}/{name}.txt"
        yolo_annotations = list()
        objects = label["frames"][0]["objects"]
        for obj in objects:
            category = obj["category"]
            if "area" in category or "lane" in category:
                # Ignore area and lane objects
                continue
            if category == "traffic light":
                color = obj["attributes"]["trafficLightColor"]
                category = f"tl_{color}"
            if category == "traffic sign":
                category = "traffic_sign"
            if category not in cls2id.keys():
                raise ValueError(f"Could not find {category} in cls2id.keys()")
            box2d = obj["box2d"]
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            assert x1 >= 0 and x1 <= IMG_WIDTH, f"x1={x1} must be between 0 and {IMG_WIDTH}"
            assert x2 >= 0 and x2 <= IMG_WIDTH, f"x2={x2} must be between 0 and {IMG_WIDTH}"
            assert y1 >= 0 and y1 <= IMG_HEIGHT, f"y1={y1} must be between 0 and {IMG_HEIGHT}"
            assert y2 >= 0 and y2 <= IMG_HEIGHT, f"y2={y2} must be between 0 and {IMG_HEIGHT}"
            x_center = ((x1 + x2) / 2) / IMG_WIDTH
            y_center = ((y1 + y2) / 2) / IMG_HEIGHT
            width = (x2 - x1) / IMG_WIDTH
            height = (y2 - y1) / IMG_HEIGHT
            assert x_center >= 0 and x_center <= 1.0, f"x_center={x_center} must be between 0 and 1.0"
            assert y_center >= 0 and y_center <= 1.0, f"y_center={y_center} must be between 0 and 1.0"
            assert width >= 0 and width <= 1.0, f"width={width} must be between 0 and 1.0"
            assert height >= 0 and height <= 1.0, f"height={height} must be between 0 and 1.0"
            category_id = cls2id[category]
            yolo_annotations.append(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        if save:
            target_image = out_file.replace("labels", "images").replace(".txt", ".jpg")
            shutil.copy(source_image, target_image)
            with open(out_file, "w") as f:
                f.write("\n".join(yolo_annotations))


def wrangle_bddk(images_dir, labels_dir, output_dir, timeofdays_train=[]):
    """
    Wrangle the BDDK dataset by copying images, and parsing labels.
    [train, test] -> train
    [val] -> val
    """
    print("Wrangling BDDK dataset...")

    labels = {
        "train": glob.glob(os.path.join(labels_dir, "100k", "train", "*.json")),
        "test": glob.glob(os.path.join(labels_dir, "100k", "test", "*.json")),
        "val": glob.glob(os.path.join(labels_dir, "100k", "val", "*.json")),
    }

    print(f"Found {len(labels['train'])} train labels")
    assert len(labels["train"]) == 70000, "Train labels should be 70000"
    print(f"Found {len(labels['test'])} test labels")
    assert len(labels["test"]) == 20000, "Test labels should be 20000"
    print(f"Found {len(labels['val'])} val labels")
    assert len(labels["val"]) == 10000, "Val labels should be 10000"

    parsed_labels = {
        "train": [],
        "test": [],
        "val": [],
    }

    # for split in ["train", "test", "val"]:
    for split in ["val", "train", "test"]:
        loader = tqdm.tqdm(labels[split], desc=f"Parsing {split} labels", unit="file")
        for label_path in loader:
            # Get the image name from the label path
            with open(label_path, "r") as f:
                label = json.load(f)
                parsed_labels[split].append(label)

    convert_labels_to_yolo(
        f"{images_dir}/100k/train",
        parsed_labels["train"],
        f"{output_dir}/labels/train",
        timeofdays=timeofdays_train,
        save=True,
    )
    convert_labels_to_yolo(
        f"{images_dir}/100k/test",
        parsed_labels["test"],
        f"{output_dir}/labels/train",
        timeofdays=timeofdays_train,
        save=True,
    )
    convert_labels_to_yolo(
        f"{images_dir}/100k/val", parsed_labels["val"], f"{output_dir}/labels/val", timeofdays=["night"], save=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrangle BDDK dataset")
    parser.add_argument(
        "images_dir",
        type=str,
        help="Directory containing the BDDK images 100k/train, 100k/val, 100k/test",
    )
    parser.add_argument(
        "labels_dir",
        type=str,
        help="Directory containing the BDDK labels 100k/train, 100k/val, 100k/test",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the wrangled dataset",
    )
    parser.add_argument(
        "--timeofdays",
        type=str,
        help="Time of day to filter train labels",
    )
    args = parser.parse_args()

    # Check if the data directory exists
    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"Images directory {args.images_dir} does not exist.")

    # Check if the labels directory exists
    if not os.path.exists(args.labels_dir):
        raise FileNotFoundError(f"Labels directory {args.labels_dir} does not exist.")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.timeofdays is None:
        timeofdays = []
    else:
        timeofdays = args.timeofdays.split(",")

    print(f"Output directory: {args.output_dir}")
    print(f"Images directory: {args.images_dir}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Time of day: {timeofdays}")

    wrangle_bddk(args.images_dir, args.labels_dir, args.output_dir, timeofdays)
    print("Done!")
