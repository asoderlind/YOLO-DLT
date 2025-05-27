from ultralytics import YOLO
from train_conf import CLUSTER_OUTPUT_PATH, MAX_DIST_WAYMO, DEVICE


def eval_model(model_path, data_path, **kwargs):
    model_name = f"val_distance/{model_path.split('/')[-3]}"
    model = YOLO(model_path)
    model.val(
        device=DEVICE,
        data=data_path,
        name=model_name,
        **kwargs,
    )


if __name__ == "__main__":
    ###################
    # BDD100k-cluster #
    ###################

    for s in [0, 1, 2]:
        for dist in [0.01, 0.05, 0.1, 0.5, 1.0]:
            eval_model(
                f"{CLUSTER_OUTPUT_PATH}/waymo_dark/yolo11n.yaml-c-200e-dist-d={dist}-noFe-lc=0.5-s={s}_/weights/best.pt",
                "waymo_dark.yaml",
                use_dist=True,
                max_dist=MAX_DIST_WAYMO,
                project=CLUSTER_OUTPUT_PATH,
            )

    ##########
    # EXDARK #
    ##########
    """
    for enhancer in ["mbllen"]:
        for lc in [0.7, 0.8, 0.9, 1.0]:
            for s in [1, 2]:
                eval_model(f"runs/detect/exDark-yolo-{enhancer}/yolo11n.yaml-c-200e-noDist-d=0.05-fe-lc={lc}-s={s}_/weights/best.pt", f"exDark-yolo-{enhancer}.yaml")

    s = 2
    lc = 0.8
    eval_model(f"runs/detect/exDark-yolo-dln-c-yolo11n.yaml-200e-{s}s-noDist-d=0.05-fe-lc={lc}_/weights/best.pt", "exDark-yolo-dln.yaml")
    """

    #########
    # KITTI #
    #########

    """

    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-1s-SGD-noDist-scale0.5-mosaic1.0-c01234567-d0.0_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-1s-SGD-dist-scale0.0-mosaic1.0-c01234567-d0.01_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-1s-SGD-dist-scale0.0-mosaic1.0-c01234567-d0.05_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-2s-SGD-dist-scale0.5-mosaic1.0-c01234567-d0.5_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-noPre-SGD-dist-scale0.0-mosaic1.0-c01234567-d0.05_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-noPre-SGD-noDist-scale0.5-mosaic1.0-c01234567-d0.0_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-noPre-SGD-dist-scale0.0-mosaic1.0-c01234567-d0.01_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    eval_distance(
        "runs/detect/kitti.yaml-dlt-models-yolo11n.yaml-200e-noPre-SGD-dist-scale0.0-mosaic1.0-c01234567-d0.05_/weights/best.pt",
        "kitti.yaml",
        True,
        classes=KITTI_CLASSES,
        max_dist=150,
    )
    """

    #########
    # WAYMO #
    #########
    """
    eval_model(
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n.yaml-100e-noPre-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt",
        "waymo-noConf.yaml",
        use_dist=True,
        max_dist=85,
    )

    eval_distance(
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n.yaml-200e-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n-SPDConv-3.yaml-200e-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n.yaml-100e-noPre-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n-SPDConv-3.yaml-100e-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n-bic-reduced-channel.yaml-100e-noPre-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n.yaml-100e-noPre-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d1.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d2.0_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    eval_distance(
        "runs/detect/waymo-noConf.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.01_/weights/best.pt",
        "waymo-noConf.yaml",
        True,
        max_dist=85,
    )
    """
