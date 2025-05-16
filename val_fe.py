from ultralytics import YOLO
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ENV = os.getenv("ENV", "LOCAL")
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6]


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
    ##########
    # EXDARK #
    ##########
    for enhancer in ["dln", "mbllen"]:
        for lc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for s in [1, 2]:
                eval_model(f"runs/detect/exDark-yolo-{enhancer}-c-yolo11n.yaml-200e-{s}s-noDist-d=0.05-fe-lc={lc}_/weights/best.pt", "exDark-yolo-dln.yaml")

    #########
    # CARLA #
    #########
    """
    eval_distance(
        "runs/detect/carla-town06-night.yaml-yolo11n.pt-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt",
        "carla-town06-night.yaml",
        True,
        max_dist=100,
    )
    eval_distance(
        "runs/detect/carla-town06-night.yaml-yolo11n.pt-200e-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt",
        "carla-town06-night.yaml",
        True,
        max_dist=100,
    )
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
    '''

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
        "runs/detect/waymo-noConf.yaml-runs-detect-carla-town06-all-night.yaml-dlt-models-yolo11n.yaml-100e-noPre-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_-weights-best.pt-100e-noPre-SGD-noDist-scale0.5-mosaic1.0-c-d0.0_/weights/best.pt",
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
        "runs/detect/waymo-noConf.yaml-dlt-models-yolo11n.yaml-100e-noPre-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt",
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
    '''
