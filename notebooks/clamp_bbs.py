import os
import glob

dataset_path = "../../yolo-testing/datasets"


def clamp_all_bbs(dataset_name: str, dry: bool = False):
    """
    Clamp all bounding boxes to the image size and save the new labels.
    :param dataset_name: The name of the dataset.
    :return: None
    """
    # Get the paths for the images and labels
    path_val = f"{dataset_path}/{dataset_name}/images/val/*"
    images = glob.glob(path_val)
    if not images:
        raise ValueError(f"No images found in {path_val}")
    path_train = f"{dataset_path}/{dataset_name}/images/train/*"
    images += glob.glob(path_train)

    print(len(images), "images found")

    # Loop through the images
    for img in images:
        # Get the image name without extension
        img_name = os.path.splitext(img)[0]

        # Get the label file path
        label_file = (
            img.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt")
        )

        # Check if the label file exists
        if not os.path.exists(label_file):
            print(f"Label file {label_file} does not exist")
            continue

        # Read the label file
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Loop through the lines and clamp the bounding boxes
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
            distance = float(parts[5]) if len(parts) > 5 else -1.0

            if x_center > 1 or x_center < 0:
                print(f"Warning: x_center {x_center} out of bounds for {img_name}")

            if y_center > 1 or y_center < 0:
                print(f"Warning: y_center {y_center} out of bounds for {img_name}")

            if box_width > 1 or box_width < 0:
                print(f"Warning: box_width {box_width} out of bounds for {img_name}")

            if box_height > 1 or box_height < 0:
                print(f"Warning: box_height {box_height} out of bounds for {img_name}")

            # Clamp the values between 0 and 1
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            box_width = min(max(box_width, 0.0), 1.0)
            box_height = min(max(box_height, 0.0), 1.0)
            if distance != -1.0:
                distance = min(max(distance, 0.0), 1.0)

            new_lines.append(f"{cls_id} {x_center} {y_center} {box_width} {box_height} {distance}\n")

        # Write the new labels to same file
        if not dry:
            with open(label_file, "w") as f:
                f.writelines(new_lines)
            print(f"Clamped bounding boxes for {img_name} and saved to {label_file}")


clamp_all_bbs("carla-yolo", dry=False)
