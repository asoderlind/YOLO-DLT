import argparse
import enum
import warnings
from pathlib import Path
from typing import TypedDict, TypeVar, Union

import cv2  # type: ignore
import dask.dataframe as dd
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from gcsfs import GCSFileSystem  # type: ignore
from tqdm import tqdm  # type: ignore

pd.set_option("display.max_columns", None)

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

DATASET_BUCKET = "gs://waymo_open_dataset_v_2_0_1"
MAX_DISTANCE = 85.0
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1280
DEFAULT_DATASET_PATH = "/mnt/machine-learning-storage/ML1/ClusterOutput/MLC-499/Datasets/GENAI-6807_Waymo/"
# DEFAULT_DATASET_PATH = "../yolo-testing/datasets/"


class Stats(TypedDict):
    day_images_training: int
    dawn_dusk_images_training: int
    night_images_training: int
    dawn_dusk_images_validation: int
    night_images_validation: int


stats: Stats = {
    "day_images_training": 0,
    "dawn_dusk_images_training": 0,
    "night_images_training": 0,
    "dawn_dusk_images_validation": 0,
    "night_images_validation": 0,
}

_AnyDataFrame = Union[pd.DataFrame, dd.DataFrame]
_DataFrame = TypeVar("_DataFrame", pd.DataFrame, dd.DataFrame)


class BoxType(enum.Enum):
    """Object type of a box (a.k.a. class/category)."""

    TYPE_UNKNOWN = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_SIGN = 3
    TYPE_CYCLIST = 4


def _how(left_nullable: bool = False, right_nullable: bool = False):
    if left_nullable and right_nullable:
        return "outer"
    elif left_nullable and not right_nullable:
        return "right"
    elif not left_nullable and right_nullable:
        return "left"
    else:
        return "inner"


def _cast_keys(src, dst, keys):
    for key in keys:
        if dst.dtypes[key] != src.dtypes[key]:
            dst[key] = dst[key].astype(src[key].dtype)


def _group_by(src: _DataFrame, keys: set[str]):
    dst = src.groupby(list(keys)).agg(list).reset_index()
    # Fix key types automatically created from the MultiIndex
    _cast_keys(src, dst, keys)
    return dst


def _select_key_columns(df: _AnyDataFrame, prefix: str) -> set[str]:
    return set([c for c in df.columns if c.startswith(prefix)])


def merge(
    left: _DataFrame,
    right: _DataFrame,
    left_nullable: bool = False,
    right_nullable: bool = False,
    left_group: bool = False,
    right_group: bool = False,
    key_prefix: str = "key.",
) -> _DataFrame:
    """Merges two tables using automatically select columns.

    This operation is called JOIN in SQL, but we use "merge" to make it consistent
    with Pandas and Dask.

    If the sets of key columns in the left and right tables do not match it will
    group by shared columns first to avoid unexpected cross products of unmatched
    columns.

    When both `left_nullable` and `right_nullable` are set to True it will perform
    an outer JOIN and output all rows from both tables in the output. When both
    set to False it will perform INNER join, otherwise - LEFT or RIGHT joins
    accordingly.

    Args:
      left: a left table.
      right: a right table.
      left_nullable: if True output may contain rows where only right columns are
        present, while left columns are null.
      right_nullable: if True output may contain rows where only left columns are
        present, while right columns are null.
      left_group: If True it will group records in the left table by common keys.
      right_group: If True it will group records in the right table by common
        keys.
      key_prefix: a string prefix used to select key columns.

    Returns:
      A new table which is the result of the join operation.
    """
    left_keys = _select_key_columns(left, key_prefix)
    right_keys = _select_key_columns(right, key_prefix)
    common_keys = left_keys.intersection(right_keys)
    if left_group and left_keys != common_keys:
        left = _group_by(left, common_keys)
    if right_group and right_keys != common_keys:
        right = _group_by(right, common_keys)
    return left.merge(right, on=list(common_keys), how=_how(left_nullable, right_nullable))


# main read function to process the data
def read(
    split: str = "training",
    tag: str = "stats",
    context_name: str = "",
    columns: list | None = None,
    filters: list | None = None,
) -> dd.DataFrame | pd.DataFrame:
    """Creates a DataFrame for the component specified by its tag."""
    if context_name:
        # Read specific context
        path = f"{DATASET_BUCKET}/{split}/{tag}/{context_name}.parquet"
        paths = [path]
    else:
        # Read all contexts for this component
        fs = GCSFileSystem()
        paths = fs.glob(f"{DATASET_BUCKET}/{split}/{tag}/*.parquet")
        paths = [f"gs://{path}" for path in paths]

    # Create the Dask DataFrame
    df = dd.read_parquet(
        paths,
        columns=columns,
        filters=filters,
    )

    return df


def get_stats_df(split: str, include_day: bool) -> pd.DataFrame:
    stats_night_df = read(
        split=split,
        tag="stats",
        columns=[
            "key.segment_context_name",
            "key.frame_timestamp_micros",
            "[StatsComponent].time_of_day",
        ],
        filters=[("[StatsComponent].time_of_day", "==", "Night")],
    ).compute()

    stats[f"night_images_{split}"] = len(stats_night_df)  # type: ignore

    if not include_day:
        return stats_night_df

    stats_dusk_dawn_df = read(
        split=split,
        tag="stats",
        columns=[
            "key.segment_context_name",
            "key.frame_timestamp_micros",
            "[StatsComponent].time_of_day",
        ],
        filters=[("[StatsComponent].time_of_day", "==", "Dawn/Dusk")],
    ).compute()

    stats[f"dawn_dusk_images_{split}"] = len(stats_dusk_dawn_df)  # type: ignore

    # We don't need to pad the validation set with day data
    if split == "validation":
        return pd.concat([stats_night_df, stats_dusk_dawn_df])

    day_head = 20656  # pad out to 50k
    stats_day_df: pd.DataFrame = (
        read(
            split=split,
            tag="stats",
            columns=[
                "key.segment_context_name",
                "key.frame_timestamp_micros",
                "[StatsComponent].time_of_day",
            ],
            filters=[("[StatsComponent].time_of_day", "==", "Day")],
        )
        .compute()
        .head(day_head)
    )

    stats["day_images_training"] = len(stats_day_df)

    stats_df = pd.concat([stats_night_df, stats_dusk_dawn_df, stats_day_df])
    return stats_df


def get_dataframes(split: str, include_day: bool, verbose: bool, save_checkpoint: bool = False):
    if verbose:
        print("getting stats")

    stats_df: pd.DataFrame = get_stats_df(split=split, include_day=include_day)
    segment_context_names: list[str] = stats_df["key.segment_context_name"].unique().tolist()
    if verbose:
        print(f"Found {len(segment_context_names)} unique segment context names.")

    if verbose:
        print("Getting bounding boxes")

    camera_box_df: pd.DataFrame = read(
        split=split,
        tag="camera_box",
        columns=[
            "key.segment_context_name",
            "key.camera_name",
            "key.frame_timestamp_micros",
            "key.camera_object_id",
            "[CameraBoxComponent].box.center.x",
            "[CameraBoxComponent].box.center.y",
            "[CameraBoxComponent].box.size.x",
            "[CameraBoxComponent].box.size.y",
            "[CameraBoxComponent].type",
        ],
        filters=[
            ("key.camera_name", "==", 1),
            ("key.segment_context_name", "in", segment_context_names),
        ],
    ).compute()

    if verbose:
        print("Getting LiDAR boxes")

    lidar_box_df: pd.DataFrame = read(
        split=split,
        tag="lidar_box",
        columns=[
            "key.segment_context_name",
            "key.frame_timestamp_micros",
            "key.laser_object_id",
            "[LiDARBoxComponent].box.center.x",
            "[LiDARBoxComponent].box.center.y",
            "[LiDARBoxComponent].box.center.z",
            "[LiDARBoxComponent].type",
        ],
        filters=[("key.segment_context_name", "in", segment_context_names)],
    ).compute()

    if verbose:
        print("Getting camera to LiDAR box association")

    camera_to_lidar_box_association_df: pd.DataFrame = read(
        split=split,
        tag="camera_to_lidar_box_association",
        filters=[
            ("key.camera_name", "==", 1),
            ("key.segment_context_name", "in", segment_context_names),
        ],
    ).compute()

    if verbose:
        print("Getting projected LiDAR boxes")

    projected_lidar_box_df: pd.DataFrame = read(
        split="training",
        tag="projected_lidar_box",
        filters=[
            ("key.camera_name", "==", 1),
            ("key.segment_context_name", "in", segment_context_names),
        ],
    ).compute()

    if verbose:
        print("Getting camera images...")

    # schema = pa.schema(
    #     [
    #         ("key.segment_context_name", pa.large_string()),
    #         ("key.camera_name", pa.int8()),
    #         ("key.frame_timestamp_micros", pa.int64()),
    #         ("[CameraImageComponent].image", pa.binary()),  # Specify binary type here
    #         ("index", pa.large_string()),
    #     ]
    # )
    camera_image_df: pd.DataFrame = read(
        split=split,
        tag="camera_image",
        context_name="",
        columns=[
            "key.segment_context_name",
            "key.camera_name",
            "key.frame_timestamp_micros",
            "[CameraImageComponent].image",
        ],
        filters=[
            ("key.camera_name", "==", 1),
            ("key.segment_context_name", "in", segment_context_names),
        ],
    ).compute()

    if save_checkpoint:
        print("Saving the dataframes to parquet files...")
        camera_box_df.to_parquet(f"camera_box_df_checkpoint_{split}.parquet")
        lidar_box_df.to_parquet(f"lidar_box_df_checkpoint_{split}.parquet")
        camera_to_lidar_box_association_df.to_parquet(f"camera_to_lidar_box_association_df_checkpoint_{split}.parquet")
        projected_lidar_box_df.to_parquet(f"projected_lidar_box_df_checkpoint_{split}.parquet")
        camera_image_df.to_parquet(f"camera_image_df_{split}.parquet")

    return (
        camera_box_df,
        lidar_box_df,
        camera_to_lidar_box_association_df,
        projected_lidar_box_df,
        camera_image_df,
    )


def calculate_vectorized_iou(camera_boxes: pd.DataFrame, projected_boxes: pd.DataFrame) -> np.ndarray:
    # Calculate corners from center and size for camera boxes
    cam_x1 = (
        camera_boxes["[CameraBoxComponent].box.center.x"] - (camera_boxes["[CameraBoxComponent].box.size.x"] / 2)
    ).to_numpy()
    cam_y1 = (
        camera_boxes["[CameraBoxComponent].box.center.y"] - (camera_boxes["[CameraBoxComponent].box.size.y"] / 2)
    ).to_numpy()
    cam_x2 = (
        camera_boxes["[CameraBoxComponent].box.center.x"] + (camera_boxes["[CameraBoxComponent].box.size.x"] / 2)
    ).to_numpy()
    cam_y2 = (
        camera_boxes["[CameraBoxComponent].box.center.y"] + (camera_boxes["[CameraBoxComponent].box.size.y"] / 2)
    ).to_numpy()

    # Calculate corners from center and size for projected boxes
    proj_x1 = (
        projected_boxes["[ProjectedLiDARBoxComponent].box.center.x"]
        - (projected_boxes["[ProjectedLiDARBoxComponent].box.size.x"] / 2)
    ).to_numpy()
    proj_y1 = (
        projected_boxes["[ProjectedLiDARBoxComponent].box.center.y"]
        - (projected_boxes["[ProjectedLiDARBoxComponent].box.size.y"] / 2)
    ).to_numpy()
    proj_x2 = (
        projected_boxes["[ProjectedLiDARBoxComponent].box.center.x"]
        + (projected_boxes["[ProjectedLiDARBoxComponent].box.size.x"] / 2)
    ).to_numpy()
    proj_y2 = (
        projected_boxes["[ProjectedLiDARBoxComponent].box.center.y"]
        + (projected_boxes["[ProjectedLiDARBoxComponent].box.size.y"] / 2)
    ).to_numpy()

    # Reshape for broadcasting
    # Each camera box vs all projected boxes
    cam_x1 = cam_x1.reshape(-1, 1)
    cam_y1 = cam_y1.reshape(-1, 1)
    cam_x2 = cam_x2.reshape(-1, 1)
    cam_y2 = cam_y2.reshape(-1, 1)

    # Calculate areas once
    cam_area = (cam_x2 - cam_x1) * (cam_y2 - cam_y1)
    proj_area = (proj_x2 - proj_x1) * (proj_y2 - proj_y1)

    # Calculate intersection coordinates
    # For each camera box (rows) vs each projected box (columns)
    x1_intersection = np.maximum(cam_x1, proj_x1)
    y1_intersection = np.maximum(cam_y1, proj_y1)
    x2_intersection = np.minimum(cam_x2, proj_x2)
    y2_intersection = np.minimum(cam_y2, proj_y2)

    # Calculate intersection width and height
    # If boxes do not overlap, set to 0
    width = np.maximum(0, x2_intersection - x1_intersection)
    height = np.maximum(0, y2_intersection - y1_intersection)

    intersection_area = width * height
    union_area = cam_area + proj_area - intersection_area

    # Calculate IoU
    # Avoid division by zero
    iou = np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(union_area),
        where=union_area != 0,
    )
    return iou


def optimized_find_projected_matches(
    unmatched_camera_boxes: pd.DataFrame, projected_lidar_box_df: pd.DataFrame
) -> pd.DataFrame:
    # Pre-group projected boxes by timestamp AND type
    projected_boxes_grouped = projected_lidar_box_df.groupby(
        ["key.frame_timestamp_micros", "[ProjectedLiDARBoxComponent].type"]
    )

    # results container
    matches = []

    # Group camera boxes by timestamp and type for batch processing
    camera_boxes_grouped = unmatched_camera_boxes.groupby(["key.frame_timestamp_micros", "[CameraBoxComponent].type"])

    print("Starting the main matching loop...")
    total_camera_boxes = len(camera_boxes_grouped)
    for (timestamp, box_type), camera_boxes in tqdm(
        camera_boxes_grouped, total=total_camera_boxes, desc="Processing camera boxes"
    ):
        try:
            # Get projected boxes for this frame and type
            projected_boxes_group = projected_boxes_grouped.get_group((timestamp, box_type))
        except KeyError:
            # No projected boxes for this frame and type
            continue

        if projected_boxes_group.empty:
            continue

        iou_matrix = calculate_vectorized_iou(
            camera_boxes,
            projected_boxes_group,
        )

        # Find best matches for all camera boxes in this group at once
        # For each camera box (row in iou_matrix), find the projected box with highest IoU

        best_match_indices = np.argmax(iou_matrix, axis=1)
        best_match_scores = np.max(iou_matrix, axis=1)

        # Only keep matches above the threshold
        threshold = 0.5
        valid_matches = best_match_scores >= threshold

        # Create match records for valid matches
        for i, is_valid in enumerate(valid_matches):
            if not is_valid:
                continue
            matches.append(
                {
                    "key.camera_object_id": camera_boxes.iloc[i]["key.camera_object_id"],
                    "key.frame_timestamp_micros": timestamp,
                    "key.laser_object_id": projected_boxes_group.iloc[best_match_indices[i]]["key.laser_object_id"],
                    "match_score": float(best_match_scores[i]),
                }
            )

    return pd.DataFrame(matches) if matches else pd.DataFrame()


def optimized_associate_distances(
    camera_box_df: pd.DataFrame,
    camera_to_lidar_box_association_df: pd.DataFrame,
    lidar_box_df: pd.DataFrame,
    projected_lidar_box_df: pd.DataFrame,
) -> pd.DataFrame:
    # Keep only relevant columns in a clean order
    desired_columns = [
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "key.camera_object_id",
        "key.laser_object_id",
        "distance",
        "match_score",
        "[CameraBoxComponent].type",
        "[CameraBoxComponent].box.center.x",
        "[CameraBoxComponent].box.center.y",
        "[CameraBoxComponent].box.size.x",
        "[CameraBoxComponent].box.size.y",
        "[LiDARBoxComponent].type",
        "[LiDARBoxComponent].box.center.x",
        "[LiDARBoxComponent].box.center.y",
        "[LiDARBoxComponent].box.center.z",
    ]

    # 1. Handle official associations first
    official_matches: pd.DataFrame = merge(camera_box_df, camera_to_lidar_box_association_df, right_nullable=True)
    official_matches = merge(official_matches, lidar_box_df, right_nullable=True)

    # Vectorized distance calculation for official matches
    has_lidar = official_matches["key.laser_object_id"].notna()

    # Pre-allocate distance column with NaN
    official_matches["distance"] = np.nan

    # Only calculate distance for rows with LiDAR data
    if has_lidar.any():
        x = official_matches.loc[has_lidar, "[LiDARBoxComponent].box.center.x"]
        y = official_matches.loc[has_lidar, "[LiDARBoxComponent].box.center.y"]
        z = official_matches.loc[has_lidar, "[LiDARBoxComponent].box.center.z"]

        # Vectorized distance calculation
        official_matches.loc[has_lidar, "distance"] = np.sqrt(x**2 + y**2 + z**2)

    # Set match score (iou) to 1.0 for official matches with 2 decimals precision
    official_matches["match_score"] = has_lidar.astype(float).round(2)

    # 2. Handle projected matches for unmatched boxes
    unmatched_camera_boxes = official_matches[official_matches["distance"].isna()].copy()

    projected_matches = optimized_find_projected_matches(unmatched_camera_boxes, projected_lidar_box_df)

    if projected_matches.empty:
        official_matches["distance"] = official_matches["distance"].replace({np.nan: 0.0}).round(1)

        # Normalize distance to be between 0 and 1
        dist_mask = official_matches["distance"] > MAX_DISTANCE
        official_matches.loc[dist_mask, "distance"] = MAX_DISTANCE
        official_matches["distance"] = official_matches["distance"].divide(MAX_DISTANCE)

        # Fix any mismatched match_score values - ensure they're 0.0 for distances of 0.0
        mask = official_matches["distance"] == 0.0
        official_matches.loc[mask, "match_score"] = 0.0

        official_matches["match_score"] = official_matches["match_score"].round(2)
        return official_matches[desired_columns]

    # Get LiDAR box information for projected matches

    projected_with_lidar = merge(projected_matches, lidar_box_df, right_nullable=True)

    if len(projected_with_lidar) > 0:
        x = projected_with_lidar["[LiDARBoxComponent].box.center.x"]
        y = projected_with_lidar["[LiDARBoxComponent].box.center.y"]
        z = projected_with_lidar["[LiDARBoxComponent].box.center.z"]

        # Vectorized distance calculation
        projected_with_lidar["distance"] = np.sqrt(x**2 + y**2 + z**2)
    match_keys = ["key.camera_object_id", "key.frame_timestamp_micros"]
    official_matches.set_index(match_keys, inplace=True)

    if not projected_with_lidar.empty:
        projected_with_lidar.set_index(match_keys, inplace=True)
        official_matches.update(projected_with_lidar)

    official_matches.reset_index(inplace=True)

    # Normalize distance to be between 0 and 1
    dist_mask = official_matches["distance"] > MAX_DISTANCE
    official_matches.loc[dist_mask, "distance"] = MAX_DISTANCE
    official_matches["distance"] = official_matches["distance"].divide(MAX_DISTANCE)
    official_matches["distance"] = official_matches["distance"].replace({np.nan: 0.0})
    # Fix any mismatched match_score values - ensure they're 0.0 for distances of 0.0
    mask = official_matches["distance"] == 0.0
    official_matches.loc[mask, "match_score"] = 0.0
    official_matches["distance"] = official_matches["distance"].round(2)
    official_matches["match_score"] = official_matches["match_score"].round(2)

    return official_matches[desired_columns]


def draw_bounding_boxes_with_distance(
    image_w_box_distance_front_night_df_row: pd.Series,
) -> None:
    image_bytes = image_w_box_distance_front_night_df_row["[CameraImageComponent].image"]
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define colors for different classes (in RGB format now)
    colors = {
        BoxType.TYPE_UNKNOWN.value: (128, 128, 128),  # Gray
        BoxType.TYPE_VEHICLE.value: (255, 0, 0),  # Red
        BoxType.TYPE_PEDESTRIAN.value: (0, 255, 0),  # Green
        BoxType.TYPE_SIGN.value: (0, 0, 255),  # Blue
        BoxType.TYPE_CYCLIST.value: (0, 255, 255),  # Yellow
    }

    # Create figure and axis
    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    for i in range(len(image_w_box_distance_front_night_df_row["[CameraBoxComponent].box.center.x"])):
        start_x = int(
            image_w_box_distance_front_night_df_row["[CameraBoxComponent].box.center.x"][i]
            - image_w_box_distance_front_night_df_row["[CameraBoxComponent].box.size.x"][i] / 2
        )
        start_y = int(
            image_w_box_distance_front_night_df_row["[CameraBoxComponent].box.center.y"][i]
            - image_w_box_distance_front_night_df_row["[CameraBoxComponent].box.size.y"][i] / 2
        )
        width = image_w_box_distance_front_night_df_row["[CameraBoxComponent].box.size.x"][i]
        height = image_w_box_distance_front_night_df_row["[CameraBoxComponent].box.size.y"][i]
        color = colors.get(
            image_w_box_distance_front_night_df_row["[CameraBoxComponent].type"][i],
            colors[BoxType.TYPE_UNKNOWN.value],
        )
        color = tuple(c / 255 for c in color)  # type: ignore
        rect = plt.Rectangle((start_x, start_y), width, height, fill=False, edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)
        label = BoxType(image_w_box_distance_front_night_df_row["[CameraBoxComponent].type"][i]).name.replace(
            "TYPE_", ""
        )
        plt.text(
            start_x,
            start_y - 5,
            label,
            color=color,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0),
        )
        # Add distance as label
        plt.text(
            start_x,
            start_y - 20,
            f"{image_w_box_distance_front_night_df_row['distance'][i]}m"
            if pd.notna(image_w_box_distance_front_night_df_row["distance"][i])
            else "",
            color=color,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0),
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_as_yolo_format(
    df: pd.DataFrame,
    output_dir: Path,
    split: str = "training",
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> None:
    """
    Save dataset in YOLO format with structured, zero-padded filenames.

    Args:
        df: DataFrame containing camera images and bounding box info
        output_dir: Base directory to save the images and labels
        split: Train/val/test split name
        image_width: Width of the original images
        image_height: Height of the original images
    """
    print(f"Saving {split} part of the dataset in YOLO format...")
    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir = output_dir / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    # Group by segment and frame timestamp
    grouped = df.groupby(["key.segment_context_name", "key.frame_timestamp_micros"])

    # Get unique segments and create a mapping from segment name to video number
    unique_segments = sorted(df["key.segment_context_name"].unique())
    segment_to_vid = {seg: i for i, seg in enumerate(unique_segments)}

    # Calculate padding requirements
    vid_pad = len(str(len(unique_segments)))

    # Calculate image padding requirements
    image_pad = len(str(len(df)))
    image_idx = 0

    # Add tqdm to the loop for a progress bar
    total_groups = len(grouped)
    for (segment_name, frame_timestamp), group_df in tqdm(grouped, total=total_groups, desc="Processing segments"):
        # Get the video number for this segment
        vid_num = segment_to_vid[segment_name]

        # Get all frames for this segment to determine frame numbering
        segment_frames = sorted(
            df[df["key.segment_context_name"] == segment_name]["key.frame_timestamp_micros"].unique()
        )
        frame_to_num = {frame: i + 1 for i, frame in enumerate(segment_frames)}
        frame_num = frame_to_num[frame_timestamp]
        frame_pad = len(str(len(segment_frames)))

        if len(group_df) != 1:
            breakpoint()
        row = group_df.iloc[0]

        # Create filename with the requested format and zero padding
        filename = f"vid_{str(vid_num).zfill(vid_pad)}_frame_{str(frame_num).zfill(frame_pad)}_image_{str(image_idx).zfill(image_pad)}"
        image_idx += 1

        # Save the image
        image_bytes = row["[CameraImageComponent].image"]
        image_path = image_dir / f"{filename}.jpeg"

        # Convert bytes to image and save
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(str(image_path), img)

        # Save the corresponding labels with matching filename
        label_path = label_dir / f"{filename}.txt"
        with open(label_path, "w") as f:
            # Check if box data exists and is in expected format
            if not (
                isinstance(row.get("[CameraBoxComponent].box.center.x"), (list, np.ndarray))
                and len(row["[CameraBoxComponent].box.center.x"]) > 0
            ):
                continue
            for j in range(len(row["[CameraBoxComponent].box.center.x"])):
                # Normalize coordinates for YOLO format
                x = round((row["[CameraBoxComponent].box.center.x"][j] / image_width), 6)
                y = round((row["[CameraBoxComponent].box.center.y"][j] / image_height), 6)
                w = round((row["[CameraBoxComponent].box.size.x"][j] / image_width), 6)
                h = round((row["[CameraBoxComponent].box.size.y"][j] / image_height), 6)
                box_type = row["[CameraBoxComponent].type"][j]

                # Include distance and match score if available
                distance = row["distance"][j] if "distance" in row and j < len(row["distance"]) else 0

                # skip match score in the labels
                # match_score = (
                #     row["match_score"][j]
                #     if "match_score" in row and j < len(row["match_score"])
                #     else 0
                # )

                # Write YOLO format line with additional data
                f.write(f"{box_type} {x} {y} {w} {h} {distance}\n")

    print(f"Saved {split} dataset in YOLO format to {output_dir}")
    print("Saved dataset with format: vid_XXX_frame_YYY_image_ZZZ.[jpeg/txt]")
    print(f"  - {len(unique_segments)} videos (padded to {vid_pad} digits)")
    print("  - Variable frames per video (padded based on each video's frame count)")
    print("  - Variable images per frame (padded based on each frame's image count)")


def wrangle_data(
    split: str = "training",
    include_day: bool = False,
    verbose: bool = True,
    save_checkpoint: bool = False,
    load_from_checkpoint: bool = False,
    output_dir: Path = Path("yolo_dataset"),
):
    print(f"Starting to wrangle the {split} data...")

    if not load_from_checkpoint:
        (
            camera_box_df,
            lidar_box_df,
            camera_to_lidar_box_association_df,
            projected_lidar_box_df,
            camera_image_df,
        ) = get_dataframes(
            split=split,
            include_day=include_day,
            verbose=verbose,
            save_checkpoint=save_checkpoint,
        )
    else:
        print("loading the checkpointed dataframes...")
        camera_box_df = pd.read_parquet("camera_box_df_checkpoint.parquet")
        lidar_box_df = pd.read_parquet("lidar_box_df_checkpoint.parquet")
        camera_to_lidar_box_association_df = pd.read_parquet("camera_to_lidar_box_association_df_checkpoint.parquet")
        projected_lidar_box_df = pd.read_parquet("projected_lidar_box_df_checkpoint.parquet")
        camera_image_df = pd.read_parquet("camera_image_df.parquet")
        print("Loaded dataframes from checkpoint.")

    # drop rows with [CameraBoxComponent].type that is 0 or 3
    type_mask = camera_box_df["[CameraBoxComponent].type"].isin([0, 3])
    camera_box_df = camera_box_df[~type_mask]
    # Remap the types so that 1 -> 0, 2 -> 1, 4 -> 2
    camera_box_df["[CameraBoxComponent].type"] = camera_box_df["[CameraBoxComponent].type"].replace({1: 0, 2: 1, 4: 2})

    print("Starting to associate distances...")
    camera_boxes_with_distance_df = optimized_associate_distances(
        camera_box_df,
        camera_to_lidar_box_association_df,
        lidar_box_df,
        projected_lidar_box_df,
    )

    # Merge the camera boxes with distance and the camera image data
    # Group them per image
    image_w_box_distance_df: pd.DataFrame = merge(
        camera_image_df,
        camera_boxes_with_distance_df,
        right_group=True,
        right_nullable=True,
    )

    potentially_null_columns = [
        "key.camera_object_id",
        "[CameraBoxComponent].box.center.x",
        "[CameraBoxComponent].box.center.y",
        "[CameraBoxComponent].box.size.x",
        "[CameraBoxComponent].box.size.y",
        "[CameraBoxComponent].type",
    ]

    def convert_empty_list(x):
        """Convert non-lists to empty lists"""
        return [] if not isinstance(x, list) else x

    # Apply to each column
    for col in potentially_null_columns:
        image_w_box_distance_df[col] = image_w_box_distance_df[col].map(convert_empty_list)

    # Skip the debug step
    # draw_bounding_boxes_with_distance(
    #     image_w_box_distance_front_night_df_row=image_w_box_distance_df.iloc[0]
    # )

    # Save the data in YOLO format
    save_as_yolo_format(
        df=image_w_box_distance_df,
        output_dir=output_dir,
        split=split,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )

    print(f"Finished wrangling the {split} data!")


def waymo(
    include_day_train: bool = False,
    include_day_val: bool = False,
    verbose: bool = True,
    save_checkpoint: bool = False,
    load_from_checkpoint: bool = False,
    dataset_name: str = "",
    mode: str = "both",
):
    if not dataset_name:
        if include_day_train and include_day_val:
            dataset_name = "waymo_day"
        elif include_day_train:
            dataset_name = "waymo"
        elif include_day_val:
            print("WARNING: You have chosen day data only for validation set, are you sure?")
            dataset_name = "waymo_day_val"
        else:
            dataset_name = "waymo_night"
    if mode in ["train", "both"]:
        wrangle_data(
            split="training",
            include_day=include_day_train,
            verbose=verbose,
            save_checkpoint=save_checkpoint,
            load_from_checkpoint=load_from_checkpoint,
            output_dir=Path(DEFAULT_DATASET_PATH + dataset_name),
        )
    if mode in ["val", "both"]:
        wrangle_data(
            split="validation",
            include_day=include_day_val,
            verbose=verbose,
            save_checkpoint=save_checkpoint,
            load_from_checkpoint=load_from_checkpoint,
            output_dir=Path(DEFAULT_DATASET_PATH + dataset_name),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waymo Open Dataset Wrangling Script")

    # Add arguments

    parser.add_argument(
        "--mode",
        choices=["train", "val", "both"],
        default="both",
        help="Specify which part of the dataset to process (train, val, or both)",
    )

    parser.add_argument(
        "--include-day-train",
        action="store_true",
        default=False,
        help="Include daytime images in training dataset",
    )

    parser.add_argument(
        "--include-day-val",
        action="store_true",
        default=False,
        help="Include daytime images in validation dataset",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information",
    )

    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Run without printing detailed progress",
    )

    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save intermediate processing checkpoints",
    )

    parser.add_argument(
        "--load-from-checkpoint",
        action="store_true",
        help="Load data from previously saved checkpoints if available",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Specify a custom dataset name (optional)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    waymo(
        include_day_train=args.include_day_train,
        include_day_val=args.include_day_val,
        verbose=args.verbose,
        save_checkpoint=args.save_checkpoint,
        load_from_checkpoint=args.load_from_checkpoint,
        dataset_name=args.dataset_name,
        mode=args.mode,
    )
