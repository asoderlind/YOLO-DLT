from .train2 import run_seeded_train, run_seeded_val

if __name__ == "__main__":
    classes = 0
    run_seeded_train(
        model_name="waymo_dark-yolo11n-bic-repc3k2-simsppf",
        model="dlt-models/yolo11n-bic-repc3k2-simsppf.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        model_name="waymo_dark-yolo11n-bic-repc3k2-sppfcsp",
        model="dlt-models/yolo11n-bic-repc3k2-sppfcsp.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        model_name="waymo_dark-yolo11n-bic-repc3k2-simsppfcsp",
        model="dlt-models/yolo11n-bic-repc3k2-simsppfcsp.yaml",
        dataset="waymo_dark.yaml",
    )

    run_seeded_val(
        model_name="waymo_dark-yolo11n-bic-repc3k2-simsppf",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        model_name="waymo_dark-yolo11n-bic-repc3k2-sppfcsp",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        model_name="waymo_dark-yolo11n-bic-repc3k2-simsppfcsp",
        data="waymo_dark.yaml",
        classes=classes,
    )
