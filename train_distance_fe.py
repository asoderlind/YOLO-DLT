from ultralytics import YOLO
import argparse
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
)


def train(
    data_path: str,
    model_path: str = MODEL,
    seed: int = SEED,
    epochs: int = EPOCHS,
    name=None,
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

    if name is None:
        name = f"{data_no_yaml}-c{class_string}-{model_no_yaml}-{epochs}e-{seed}s-{'dist' if use_dist else 'noDist'}-d={dist}-{'fe' if use_fe else 'noFe'}-lc={lambda_c}_"
        name = name.replace("/", "-")

    model.train(
        name=name,
        data=data_path,
        pretrained=PRETRAINED,
        scale=scale,
        epochs=epochs,
        device=DEVICE,
        batch=BATCH,
        momentum=MOMENTUM,
        lr0=LR0,
        iou_type=IOU_TYPE,
        warmup_bias_lr=WARMUP_BIAS_LR,
        optimizer=OPTIMIZER,
        seed=seed,
        **kwargs,
    )
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="distance", help="Mode to run: distance, fe")
    parser.add_argument(
        "--data",
        type=str,
        default="kitti.yaml",
        help="Path to the data yaml file",
        choices=["kitti", "waymo", "exdark", "bdd100k"],
    )
    args = parser.parse_args()

    if args.mode == "distance" and args.data == "waymo":
        #########
        # WAYMO #
        #########
        """
        train(
            data_path="waymo-noConf.yaml",
            use_dist=False,
            dist=0.00,
            scale=0.5,
            max_dist=MAX_DIST_WAYMO,
        )
        train(
            data_path="waymo-noConf.yaml",
            use_dist=True,
            dist=0.01,
            max_dist=MAX_DIST_WAYMO,
        )
        # SPDConv
        train(
            model_path="dlt-models/yolo11n-SPDConv-3.yaml",
            data_path="waymo-noConf.yaml",
            use_dist=False,
            dist=0.00,
            scale=0.5,
            max_dist=MAX_DIST_WAYMO,
        )
        train(
            model_path="dlt-models/yolo11n-SPDConv-3.yaml",
            data_path="waymo-noConf.yaml",
            use_dist=True,
            dist=0.01,
            max_dist=MAX_DIST_WAYMO,
        )
        """
        pass

    if args.mode == "distance" and args.data == "kitti":
        #########
        # KITTI #
        #########

        # Test without dist
        """
        train(data_path="kitti.yaml", use_dist=True, dist=0.5, classes=KITTI_CLASSES, epochs=200)
        train(
            data_path="kitti.yaml", model_path="runs/detect/kitti.yaml-dlt-models-yolo11n-SPDConv-3.yaml-200e-SGD-dist-scale0.0-mosaic1.0-c01234567-d0.5_5/weights/last.pt", use_dist=True, dist=0.5, classes=KITTI_CLASSES, epochs=200, resume=True
        )
        train(
            data_path="kitti.yaml", model_path="dlt-models/yolo11n-SPDConv-3.yaml", use_dist=True, dist=1.0, classes=KITTI_CLASSES, epochs=200)
        """

        # no dist
        for s in [1]:
            for d in [1.0]:
                train(
                    data_path="kitti.yaml",
                    use_dist=True,
                    dist=d,
                    seed=s,
                    classes=KITTI_CLASSES,
                )

        for s in [2]:
            train(
                data_path="kitti.yaml",
                use_dist=False,
                dist=d,
                seed=s,
                classes=KITTI_CLASSES,
            )
            for d in [0.01, 0.05, 0.1, 1.0]:
                train(
                    data_path="kitti.yaml",
                    use_dist=True,
                    dist=d,
                    seed=s,
                    classes=KITTI_CLASSES,
                )

        # train(data_path="kitti.yaml", use_dist=True, dist=0.5, seed=seed, classes=KITTI_CLASSES, epochs=200, scale=0.0, mosaic=0.0)
        # train(data_path="kitti.yaml", use_dist=True, dist=0.5, seed=seed, classes=KITTI_CLASSES, epochs=200, scale=0.0, mosaic=1.0)

    if args.mode == "fe" and args.data == "exdark":
        ##########
        # EXDARK #
        ##########

        # Baseline exDark-yolo
        train(
            data_path="exDark-yolo.yaml",
            use_fe=False,
            lambda_c=0.0,
            val=True,
        )

        # All loss on exDark with DLN
        for lambda_c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            train(
                data_path="exDark-yolo.yaml",
                use_fe=True,
                lambda_c=lambda_c,
            )

        # Curriculum on exDark with DLN
        for lambda_c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            first_layer_trained = train(
                data_path="exDark-yolo.yaml",
                use_fe=True,
                lambda_c=lambda_c,
                epochs=10,
                box=0.0,
                cls=0.0,
                dfl=0.0,
                val=False,
            )
            train(
                model_name=f"runs/detect/{first_layer_trained}/weights/last.pt",
                data_path="exDark-yolo.yaml",
                use_fe=False,
                lambda_c=0.0,
                freeze=1,
            )

    if args.mode == "fe" and args.data == "bdd100k":
        # Curriculum on bdd100k with mirnet
        """
        first_layer_trained = train(
            data_path="bdd100k_night_mirnet.yaml",
            use_fe=True,
            epochs=10,
            lambda_c=0.5,
            box=0.0,
            cls=0.0,
            dfl=0.0,
            val=False
        )
        train(
            model_name=f"runs/detect/dlt-models/yolo11n.yaml-10e-bdd100k_night_mirnet.yaml-fe-lambda_c0.5_2/weights/last.pt",
            data_path="bdd100k_night_mirnet.yaml",
            use_fe=False,
            lambda_c=0.0,
            val=False,
            freeze=1
        )

        # Curriculum on bdd100k with DLN
        first_layer_trained = train(
            data_path="bdd100k_night.yaml", use_fe=True, epochs=10, lambda_c=0.5, box=0.0, cls=0.0, dfl=0.0, val=False
        )
        train(
            model_name=f"runs/detect/{first_layer_trained}/weights/last.pt",
            data_path="bdd100k_night.yaml",
            use_fe=False,
            lambda_c=0.0,
            val=False,
            freeze=1,
        )

        # All loss on bdd100k with mirnet
        train(
            data_path="bdd100k_night_mirnet.yaml",
            use_fe=True,
            lambda_c=0.5,
        )

        # All loss on bdd100k with DLN
        train(
            data_path="bdd100k_night.yaml",
            use_fe=True,
            lambda_c=0.5,
        )
        """
