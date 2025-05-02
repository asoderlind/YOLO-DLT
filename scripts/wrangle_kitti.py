"""
Example usage:
python wrangle_bddk.py ~/repos/yolo-testing/datasets/bdd100k_images_100k ~/repos/yolo-testing/datasets/bdd100k_labels ~/repos/yolo-testing/datasets/bdd100k-night-yolo --timeofdays "night"
"""

import argparse

import os
import glob
import math
from PIL import Image
import shutil
import tqdm

class2index = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
    "DontCare": 8,
}


def ltrb_to_xywh(img_w, img_h, l, t, r, b) -> tuple[float, float, float, float]:
    w = r - l
    h = b - t
    x = l + w / 2
    y = t + h / 2
    x_rel = x / img_w
    y_rel = y / img_h
    w_rel = w / img_w
    h_rel = h / img_h
    # truncate to 4 decimal places
    return round(x_rel, 4), round(y_rel, 4), round(w_rel, 4), round(h_rel, 4)


def load_images_and_labels(images_path, labels_path, format):
    images = []
    labels = []
    loader = tqdm.tqdm(
        glob.glob(images_path + "/*.{}".format(format)),
        desc="Loading images and labels",
        unit="file",
    )
    for image_path in loader:
        image_name = os.path.basename(image_path)
        label_path = os.path.join(labels_path, image_name.replace(format, "txt"))
        if not os.path.exists(label_path):
            continue
        with open(label_path) as f:
            label = []
            for line in f.read().strip().split("\n"):
                object_type, truncation, occlusion, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry = line.split(" ")
                annotation = {
                    "object_type": object_type,
                    "truncation": float(truncation),
                    "occlusion": int(occlusion),
                    "alpha": float(alpha),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "h": float(h),
                    "w": float(w),
                    "l": float(l),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "ry": float(ry),
                }
                label.append(annotation)
            labels.append((label_path, label))
        images.append(image_path)
    return images, labels


def wrangle_kitti(images_dir, labels_dir, new_dataset_dir):
    """ """
    kitti_images, kitti_labels = load_images_and_labels(images_dir, labels_dir, "png")

    print("Kitti images: ", len(kitti_images))
    print("Kitti labels: ", len(kitti_labels))

    print(kitti_images[:5])
    print(kitti_labels[:5])

    assert len(kitti_images) == len(kitti_labels)

    if not os.path.exists(new_dataset_dir):
        os.makedirs(new_dataset_dir)

    new_images_dir = os.path.join(new_dataset_dir, "images")
    if not os.path.exists(new_images_dir):
        os.makedirs(new_images_dir)

    new_labels_dir = os.path.join(new_dataset_dir, "labels")
    if not os.path.exists(new_labels_dir):
        os.makedirs(new_labels_dir)

    num_train_images = 5241
    num_val_images = 2240
    assert num_train_images + num_val_images == len(kitti_images)

    new_images_train_dir = os.path.join(new_images_dir, "train")
    new_images_val_dir = os.path.join(new_images_dir, "val")
    if not os.path.exists(new_images_train_dir):
        os.makedirs(new_images_train_dir, exist_ok=True)
    if not os.path.exists(new_images_val_dir):
        os.makedirs(new_images_val_dir, exist_ok=True)

    new_labels_train_dir = os.path.join(new_labels_dir, "train")
    new_labels_val_dir = os.path.join(new_labels_dir, "val")
    if not os.path.exists(new_labels_train_dir):
        os.makedirs(new_labels_train_dir, exist_ok=True)
    if not os.path.exists(new_labels_val_dir):
        os.makedirs(new_labels_val_dir, exist_ok=True)

    tqdm_counter = tqdm.tqdm(
        range(len(kitti_images)),
        desc="Copying images and labels",
        unit="file",
    )
    for label_path, label in kitti_labels[:num_train_images]:
        tqdm_counter.update(1)
        output_label_path = new_labels_train_dir + "/" + os.path.basename(label_path)
        image_path = label_path.replace("label", "image").replace("txt", "png")
        # get image dimensions with PIL

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        shutil.copy(image_path, new_images_train_dir)

        with open(output_label_path, "w") as f:
            for annotation in label:
                c = class2index[annotation["object_type"]]
                x, y, w, h = ltrb_to_xywh(
                    img_w, img_h, annotation["x1"], annotation["y1"], annotation["x2"], annotation["y2"]
                )
                if c == 8:  # DontCare
                    distance_to_object = 0
                else:
                    distance_to_object = math.sqrt(annotation["x"] ** 2 + annotation["y"] ** 2 + annotation["z"] ** 2)
                    # Clamp and normalize distance to object
                    if distance_to_object > 150:
                        distance_to_object = 150  # max distance in KITTI dataset
                    if distance_to_object < 0:
                        distance_to_object = 0  # min distance in KITTI dataset
                    distance_to_object = distance_to_object / 150  # normalize to [0, 1]
                    distance_to_object = round(distance_to_object, 4)  # truncate to 4 decimal places
                f.write("{} {} {} {} {} {}\n".format(c, x, y, w, h, distance_to_object))

    for label_path, label in kitti_labels[num_train_images:]:
        tqdm_counter.update(1)
        output_label_path = new_labels_val_dir + "/" + os.path.basename(label_path)
        image_path = label_path.replace("label", "image").replace("txt", "png")
        # get image dimensions with PIL

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        shutil.copy(image_path, new_images_val_dir)

        with open(output_label_path, "w") as f:
            for annotation in label:
                c = class2index[annotation["object_type"]]
                x, y, w, h = ltrb_to_xywh(
                    img_w, img_h, annotation["x1"], annotation["y1"], annotation["x2"], annotation["y2"]
                )
                if c == 8:  # DontCare
                    distance_to_object = 0
                else:
                    distance_to_object = math.sqrt(annotation["x"] ** 2 + annotation["y"] ** 2 + annotation["z"] ** 2)
                    # Clamp and normalize distance to object
                    if distance_to_object > 150:
                        distance_to_object = 150  # max distance in KITTI dataset
                    if distance_to_object < 0:
                        distance_to_object = 0  # min distance in KITTI dataset
                    distance_to_object = distance_to_object / 150  # normalize to [0, 1]
                    distance_to_object = round(distance_to_object, 4)  # truncate to 4 decimal places
                f.write("{} {} {} {} {} {}\n".format(c, x, y, w, h, distance_to_object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrangle KITTI dataset")
    parser.add_argument(
        "images_dir", type=str, help="Directory containing the KITTI images e.g. data_object_image_2/training/image_2"
    )
    parser.add_argument(
        "labels_dir", type=str, help="Directory containing the BDDK labels e.g. data_object_label_2/training/label_2"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the wrangled dataset e.g datasets/kitti-yolo",
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

    print(f"Output directory: {args.output_dir}")
    print(f"Images directory: {args.images_dir}")
    print(f"Labels directory: {args.labels_dir}")

    wrangle_kitti(args.images_dir, args.labels_dir, args.output_dir)
    print("Done!")
