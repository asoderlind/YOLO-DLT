import glob
import argparse


def get_statistics(dataset_path, image_formats=["jpg"]) -> dict:
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


#stats = get_statistics('../../yolo-testing/datasets/bdd100k_night')
# stats = get_statistics('../../yolo-testing/datasets/waymo_open_dataset', image_format="jpeg")
# stats = get_statistics('../../yolo-testing/datasets/night_Australia')
# stats = get_statistics('../../yolo-testing/datasets/night_only_Canada')
# stats = get_statistics('../../yolo-testing/datasets/night_only_China')
parser = argparse.ArgumentParser(description="Run statistics on dataset")

parser.add_argument("name", type=str, default="bdd100k_night")

args = parser.parse_args()

stats = get_statistics(
    f"../../yolo-testing/datasets/{args.name}", image_formats=["jpg", "jpeg", "png"]
)

for key, value in stats.items():
    print(f"{key}: {value}")
