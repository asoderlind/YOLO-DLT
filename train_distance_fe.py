from ultralytics import YOLO
import argparse
import os
from train_conf import (
    DEVICE,
    KITTI_CLASSES,
    OPTIMIZER,
    MOMENTUM,
    BATCH,
    IOU_TYPE,
    LR0,
    WARMUP_BIAS_LR,
    PRETRAINED,
    MODEL,
    EPOCHS,
    SEED,
    MAX_DIST_WAYMO,
    CLUSTER_BDDK_WEIGHT_PATH,
    CLUSTER_OUTPUT_PATH,
)


def train(
    data_path: str,
    model_path: str = MODEL,
    seed: int = SEED,
    epochs: int = EPOCHS,
    lr0: float = LR0,
    optimizer: str = OPTIMIZER,
    name=None,
    project="runs/detect",
    **kwargs,
) -> str:
    classes = kwargs.get("classes", [])
    class_string = "".join([str(c) for c in classes])

    use_fe = kwargs.get("use_fe", False)  # default: False
    lambda_c = kwargs.get("lambda_c", 0.5)  # default: 0.5
    use_dist = kwargs.get("use_dist", False)  # default: False
    dist = kwargs.get("dist", 0.05)  # default: 0.05
    # freeze = kwargs.get("freeze", None)  # default: 0

    model = YOLO(model_path)

    data_no_yaml = data_path.split(".yaml")[0]
    model_no_yaml = model_path.split("/")[-1]

    scale = 0.0 if use_dist else 0.5
    hsv_h = 0.0 if use_fe else 0.015
    hsv_s = 0.0 if use_fe else 0.7
    hsv_v = 0.0 if use_fe else 0.4

    if name is None:
        name = f"{data_no_yaml}/{model_no_yaml}-c{class_string}-lr0={lr0}-{epochs}e-{'dist' if use_dist else 'noDist'}-d={dist}-{'fe' if use_fe else 'noFe'}-lc={lambda_c}-s={seed}_"
        last_weight = f"{project}/{name}/weights/last.pt"
        if os.path.exists(last_weight):
            model = YOLO(last_weight)
            kwargs["resume"] = True

    try:
        model.train(
            project=project,
            name=name,
            data=data_path,
            pretrained=PRETRAINED,
            scale=scale,
            epochs=epochs,
            device=DEVICE,
            batch=BATCH,
            momentum=MOMENTUM,
            lr0=lr0,
            iou_type=IOU_TYPE,
            warmup_bias_lr=WARMUP_BIAS_LR,
            optimizer=optimizer,
            seed=seed,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            **kwargs,
        )
    except AssertionError as e:
        print("Already finished: ", e)
        return name

    return name


if __name__ == "__main__":
    ######
    # FE #
    ######

    # exDark

    """
    for data in ["exDark-yolo-mbllen.yaml"]:
        # All loss
        for lambda_c in [1.0]:
            for s in [2]:
                train(
                    data_path=data,
                    use_fe=True,
                    lambda_c=lambda_c,
                    seed=s
                )

    # BDD100k-night
    for s in [2]:
        # MBLLEN
        train(
            data_path="bdd100k_night-mbllen.yaml",
            use_fe=True,
            lambda_c=0.1,
            seed=s,
        )
    """

    ############
    # Distance #
    ############

    # KITTI

    # Test without dist
    # train(data_path="kitti.yaml", use_dist=True, dist=0.5, classes=KITTI_CLASSES, epochs=200)

    # Waymo-night

    for s in [0, 1, 2]:
        train(
            model_path=CLUSTER_BDDK_WEIGHT_PATH,
            data_path="waymo_cluster_night.yaml",
            use_dist=False,
            dist=0.00,
            max_dist=MAX_DIST_WAYMO,
            seed=s,
            lr0=0.001,
            epochs=50,
            project=CLUSTER_OUTPUT_PATH,
        )
        """
        train(
            data_path="waymo_cluster_night.yaml",
            use_dist=True,
            dist=0.01,
            max_dist=MAX_DIST_WAYMO,
            seed=s,
        )
        train(
            data_path="waymo_cluster_night.yaml",
            use_dist=True,
            dist=0.05,
            max_dist=MAX_DIST_WAYMO,
            seed=s,
        )
        train(
            data_path="waymo_cluster_night.yaml",
            use_dist=True,
            dist=0.1,
            max_dist=MAX_DIST_WAYMO,
            seed=s,
        )
        train(
            data_path="waymo_cluster_night.yaml",
            use_dist=True,
            dist=0.5,
            max_dist=MAX_DIST_WAYMO,
            seed=s,
        )
        train(
            data_path="waymo_cluster_night.yaml",
            use_dist=True,
            dist=1.0,
            max_dist=MAX_DIST_WAYMO,
            seed=s,
        )
        """
