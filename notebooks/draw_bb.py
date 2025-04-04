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


def get_id2cls(dataset: str):
    # switch statement
    id2cls = {}
    if dataset == "kitti-yolo":
        class2index = KITTI
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    id2cls = {v: k for k, v in class2index.items()}
    return id2cls


def main(dataset: str):
    # switch statement
    if dataset == "kitti-yolo":
        dataset_path = os.path.join(path, "kitti-yolo")
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        id2cls = get_id2cls(dataset)
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Example usage
    img_path = random.sample(glob.glob(f"{dataset_path}/images/train/*"), 1)[0]
    image_name = os.path.basename(img_path)[:-4]
    label_path = f"{dataset_path}/labels/train/{image_name}.txt"

    print(img_path, label_path)

    # get img_w and img_h from the image
    img = PIL.Image.open(img_path)
    img_w, img_h = img.size
    print(img_w, img_h)

    draw_yolo_bboxes(img_path, label_path, img_w, img_h, id2cls, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding boxes on images")
    parser.add_argument(
        "dataset",
        type=str,
        default="kitti-yolo",
        choices=["kitti-yolo"],
        help="Dataset to use",
    )
    args = parser.parse_args()
    main(args.dataset)
