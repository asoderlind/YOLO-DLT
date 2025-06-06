import matplotlib.pyplot as plt
from PIL import Image


def plot_dataset_vertical(labels, predictions, figsize=(10, None), save_path=None):
    """
    Plots label images on top of prediction images in a single column for each image pair.
    No overall title is added.

    Args:
        labels (list): List of file paths for label images.
        predictions (list): List of file paths for prediction images.
        figsize (tuple): Figure size as (width, height). If height is None, it defaults to 3 inches per image
                         (each pair takes 2 rows, so 2 * 3 inches per pair).
        save_path (str, optional): If provided, the figure is saved to this path.
    """
    n = len(labels)
    # Default height: each image gets 3 inches, and each pair has 2 images.
    if figsize[1] is None:
        figsize = (figsize[0], n * 2 * 3)

    # Create subplots: 2 rows per pair => total rows=2*n, in 1 column.
    fig, axes = plt.subplots(nrows=2 * n, ncols=1, figsize=figsize)

    # If there's only one pair, make axes a list for easy indexing
    if 2 * n == 1:
        axes = [axes]

    # Plot each pair: label on top, prediction below.
    for i in range(n):
        # Top row: Label image
        label_img = Image.open(labels[i])
        axes[2 * i].imshow(label_img)
        axes[2 * i].axis("off")

        # Bottom row: Prediction image
        pred_img = Image.open(predictions[i])
        axes[2 * i + 1].imshow(pred_img)
        axes[2 * i + 1].axis("off")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()


def plot_dataset_grid(labels, predictions, figsize=(10, None), save_path=None):
    """
    Plots label images in the left column and prediction images in the right column,
    with subtitles above each column.

    Args:
        labels (list): List of file paths for label images.
        predictions (list): List of file paths for prediction images.
        title (str): Overall title for the plot.
        figsize (tuple): Figure size as (width, height). If height is None, rows * 3 is used.
        save_path (str, optional): If provided, the figure is saved to this path.
    """
    n = len(predictions)
    # if height is not provided, set a default (3 inches per row)
    if figsize[1] is None:
        figsize = (figsize[0], n * 3)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=figsize)
    # if there is only one row, ensure axes is iterable of two axes
    if n == 1:
        axes = [axes]

    # Set subtitles in the top row for each column
    axes[0][0].set_title("Ground truth", fontsize=16)
    axes[0][1].set_title("Predictions", fontsize=16)

    for i in range(n):
        # Load the label and prediction images
        label_img = Image.open(labels[i])
        pred_img = Image.open(predictions[i])

        # Left column: Label image
        axes[i][0].imshow(label_img)
        axes[i][0].axis("off")

        # Right column: Prediction image
        axes[i][1].imshow(pred_img)
        axes[i][1].axis("off")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()


if __name__ == "__main__":
    # --- Plot for KITTI dataset ---
    kitti_predictions = [
        "kitti-yolo_predictions/000010_predictions.jpg",
        "kitti-yolo_predictions/000602_predictions.jpg",
    ]
    kitti_labels = [
        "kitti-yolo_labels/labels-kitti-yolo-000010.png",
        "kitti-yolo_labels/labels-kitti-yolo-000602.png",
    ]
    plot_dataset_vertical(
        kitti_labels,
        kitti_predictions,
        save_path="grids/kitti_predictions_grid.png",
    )

    # --- Plot for Waymo dataset ---
    waymo_predictions = [
        "waymo_dark_predictions/vid_01_frame_170_image_0367_predictions.jpg",
        "waymo_dark_predictions/vid_04_frame_015_image_0797_predictions.jpg",
        "waymo_dark_predictions/vid_05_frame_174_image_1152_predictions.jpg",
        "waymo_dark_predictions/vid_10_frame_121_image_2083_predictions.jpg",
    ]
    waymo_labels = [
        "waymo_dark_labels/labels-waymo_dark-vid_01_frame_170_image_0367.png",
        "waymo_dark_labels/labels-waymo_dark-vid_04_frame_015_image_0797.png",
        "waymo_dark_labels/labels-waymo_dark-vid_05_frame_174_image_1152.png",
        "waymo_dark_labels/labels-waymo_dark-vid_10_frame_121_image_2083.png",
    ]
    plot_dataset_grid(
        waymo_labels,
        waymo_predictions,
        save_path="grids/waymo_predictions_grid.png",
    )
