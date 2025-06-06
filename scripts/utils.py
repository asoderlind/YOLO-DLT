import cv2
import matplotlib.pyplot as plt
import os

import numpy as np


def xywh2xyxy(x_center, y_center, w_box, h_box, width, height):
    """
    Convert YOLO format (x_center, y_center, w_box, h_box) to (x1, y1, x2, y2)
    """
    x1 = int((x_center - w_box / 2) * width)
    y1 = int((y_center - h_box / 2) * height)
    x2 = int((x_center + w_box / 2) * width)
    y2 = int((y_center + h_box / 2) * height)

    # Clamp the coordinates to be within the image dimensions
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return x1, y1, x2, y2


def xyxy2xywh(x1, y1, x2, y2, width, height):
    """
    Convert (x1, y1, x2, y2) to YOLO format (x_center, y_center, w_box, h_box)
    """
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    w_box = (x2 - x1) / width
    h_box = (y2 - y1) / height

    return x_center, y_center, w_box, h_box


def filter_occluded_boxes(label_file: str, img_height: int, img_width: int, occlusion_threshold: float = 0.5):
    # Make annotations array
    with open(label_file, "r") as f:
        lines = f.readlines()

    num_lines = len(lines)
    annotations = np.zeros((num_lines, 6), dtype=np.float32)

    for i, line in enumerate(lines):
        # Split the line into components
        components = line.strip().split()
        # Extract the bounding box coordinates
        cls, x_center, y_center, w_box, h_box, dist = map(float, components[:6])

        # Convert to (x1, y1, x2, y2) format
        x1, y1, x2, y2 = xywh2xyxy(x_center, y_center, w_box, h_box, img_width, img_height)

        # Append the bounding box to the list
        annotations[i, 0] = cls
        annotations[i, 1] = x1
        annotations[i, 2] = y1
        annotations[i, 3] = x2
        annotations[i, 4] = y2
        annotations[i, 5] = dist

    # Sort the annotations by distance, largest to smallest
    annotations = annotations[np.argsort(annotations[:, 5])[::-1]]

    # Create a mock image with the same height and width as the image
    # and a channel for classification
    mock_image = np.zeros((img_height, img_width), dtype=np.int8)
    mock_image.fill(-1)

    total_area_per_annotation = np.zeros(num_lines, dtype=np.float32)
    visible_area_per_annotation = np.zeros(num_lines, dtype=np.float32)

    # Iterate over each annotation
    for i in range(num_lines):
        # Get the coordinates of the bounding box
        x1, y1, x2, y2 = annotations[i, 1:5].astype(int)

        # Calculate the area of the bounding box
        total_area = (x2 - x1) * (y2 - y1)
        total_area_per_annotation[i] = total_area

        # Add the mask to the mock image
        mock_image[y1:y2, x1:x2] = i

    for i in range(num_lines):
        # Compare the total area with the visible area
        visible_area = np.sum(mock_image == i)
        visible_area_per_annotation[i] = visible_area

    visibility_ratio_per_annotation = visible_area_per_annotation / total_area_per_annotation

    # Remove annotations with ratio less than the occlusion threshold
    valid_indices = np.where(visibility_ratio_per_annotation >= occlusion_threshold)[0]
    annotations = annotations[valid_indices]

    # Convert the annotations back to YOLO format
    for i in range(len(annotations)):
        x1, y1, x2, y2 = annotations[i, 1:5].astype(int)
        x_center, y_center, w_box, h_box = xyxy2xywh(x1, y1, x2, y2, img_width, img_height)
        annotations[i, 1:5] = [x_center, y_center, w_box, h_box]

    return annotations


def draw_yolo_bboxes(
    image_path: str,
    label_path: str,
    w: int,
    h: int,
    id2cls: dict,
    classes: list[int],
    max_dist=-1,
    text_size=1,
    text_thickness=2,
):
    # print("image_path:", image_path)
    # print("label_path:", label_path)
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path, "rt") as f:
        data_lines = f.readlines()

    labels: list[tuple[int, int, int, int, str, str]] = []

    for item in range(len(data_lines)):
        if len(data_lines[item].split(" ")) == 5:
            cls, x_center, y_center, w_box, h_box = data_lines[item].split(" ")
            cls = int(cls)
            dist = -1.0
        else:
            cls, x_center, y_center, w_box, h_box, dist = data_lines[item].split(" ")
            cls = int(cls)
            dist = f"{float(dist) * max_dist:.2f}"

        x1 = int((float(x_center) - float(w_box) / 2) * float(w))
        y1 = int((float(y_center) - float(h_box) / 2) * float(h))
        x2 = int((float(x_center) + float(w_box) / 2) * float(w))
        y2 = int((float(y_center) + float(h_box) / 2) * float(h))

        if cls not in classes:
            continue
        else:
            _class = id2cls[cls] if cls in id2cls else str(cls)
            labels.append((x1, y1, x2, y2, _class, dist))

    # print("labels:", labels)
    for label in labels:
        x1, y1, x2, y2, cls, dist = label
        color = {
            "Pedestrian": (255, 100, 100),
            "Car": (100, 255, 255),
            "Cyclist": (150, 150, 255),
            "Tram": (255, 150, 150),
        }
        cv2.rectangle(img, (x1, y1), (x2, y2), color[cls] if cls in color else (255, 255, 255), 2)
        cv2.putText(
            img,
            f"{cls}{'/' + str(dist) if dist != -1 else ''}",
            (x1, y1 - 5),
            0,
            text_size,
            color[cls] if cls in color else (255, 255, 255),
            text_thickness,
        )

    plt.figure(figsize=(10, 6))  # Set figure size
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    # tight
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    dataset = image_path.split("/")[-4]
    img_name = image_path.split("/")[-1].split(".")[0]

    if not os.path.exists(f"../{dataset}_labels/"):
        os.makedirs(f"../{dataset}_labels/")

    plt.savefig(
        f"../{dataset}_labels/labels-{dataset}-{img_name}.png", bbox_inches="tight", pad_inches=0.0
    )  # Save the figure

    plt.close()  # Close the figure to free memory
