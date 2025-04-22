import random
import glob
import os
import PIL
import argparse
from utils import draw_yolo_bboxes

path = "../../yolo-testing/datasets/"

KITTI = {
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

BDD100K = {
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
    "traffic_signal": 11,
    "train": 12,
}

WAYMO = {
    "unknown": 0,
    "vehicle": 1,
    "pedestrian": 2,
    "sign": 3,
    "cyclist": 4,
}

CARLA = {
    "Car": 0,
    "Truck": 1,
    "Van": 2,
    "Bus": 3,
    "Motorcycle": 4,
    "Bicycle": 5,
}


def get_id2cls(dataset: str):
    # switch statement
    id2cls = {}
    if dataset == "kitti-yolo":
        class2index = KITTI
    elif dataset == "bdd100k_night":
        class2index = BDD100K
    elif dataset == "waymo-noConf":
        class2index = WAYMO
    elif dataset == "carla-yolo" or "carla-town06-yolo-v3":
        class2index = CARLA
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    id2cls = {v: k for k, v in class2index.items()}
    return id2cls


def main(dataset: str, img_name: str | None = None):
    # switch statement
    if dataset == "kitti-yolo":
        dataset_path = os.path.join(path, "kitti-yolo")
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        id2cls = get_id2cls(dataset)
        max_dist = 150.0
    elif dataset == "bdd100k_night":
        dataset_path = os.path.join(path, "bdd100k_night")
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        id2cls = get_id2cls(dataset)
        max_dist = -1.0
    elif dataset == "waymo-noConf":
        dataset_path = os.path.join(path, "waymo-noConf")
        classes = [1, 2, 3, 4]
        id2cls = get_id2cls(dataset)
        max_dist = 85.0
    elif dataset == "carla-yolo" or dataset == "carla-town06-yolo-v3":
        dataset_path = os.path.join(path, dataset)
        classes = [0, 1, 2, 3, 4, 5]
        id2cls = get_id2cls(dataset)
        max_dist = 100.0
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Example usage
    all_imgs_path = f"{dataset_path}/images/train"
    print(f"Path: {all_imgs_path}")

    all_imgs = glob.glob(f"{all_imgs_path}/*")

    print("Example images:")
    print(all_imgs[:3])

    print(f"Number of images: {len(all_imgs)}")

    print("img_name:", img_name)

    if img_name is not None:
        img_path = os.path.join(all_imgs_path, img_name)
        if not os.path.exists(img_path):
            raise ValueError(f"Image {img_name} not found in {all_imgs_path}")
    else:
        img_path = random.sample(all_imgs, 1)[0]

    image_name = (
        os.path.basename(img_path).replace(".jpg", "").replace(".png", "").replace(".jpeg", "").replace(".JPEG", "")
    )
    label_path = f"{dataset_path}/labels/train/{image_name}.txt"

    print(img_path, label_path)

    # get img_w and img_h from the image
    img = PIL.Image.open(img_path)
    img_w, img_h = img.size
    print(img_w, img_h)

    draw_yolo_bboxes(img_path, label_path, img_w, img_h, id2cls, classes, max_dist=max_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding boxes on images")
    parser.add_argument(
        "dataset",
        type=str,
        default="kitti-yolo",
        choices=["kitti-yolo", "bdd100k_night", "waymo-noConf", "carla-yolo", "carla-town06-yolo-v3"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--img",
        type=str,
        help="Name of the image to use",
    )
    args = parser.parse_args()
    main(args.dataset, img_name=args.img)
