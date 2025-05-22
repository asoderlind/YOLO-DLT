from .train2 import run_seeded_train, run_seeded_val

if __name__ == "__main__":
    classes = 0
    run_seeded_train(
        model_name="waymo_dark-yolo11n-bic-repc3k2-se",
        model="dlt-models/yolo11n-bic-repc3k2-se.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        model_name="waymo_dark-yolo11n-bic-repc3k2-eca",
        model="dlt-models/yolo11n-bic-repc3k2-eca.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        model_name="waymo_dark-yolo11n-bic-repc3k2-simam",
        model="dlt-models/yolo11n-bic-repc3k2-simam.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        model_name="waymo_dark-yolo11n-bic-repc3k2-ca",
        model="dlt-models/yolo11n-bic-repc3k2-ca.yaml",
        dataset="waymo_dark.yaml",
    )

    run_seeded_val(
        model_name="waymo_dark-yolo11n-bic-repc3k2-se",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        model_name="waymo_dark-yolo11n-bic-repc3k2-eca",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        model_name="waymo_dark-yolo11n-bic-repc3k2-simam",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        model_name="waymo_dark-yolo11n-bic-repc3k2-ca",
        data="waymo_dark.yaml",
        classes=classes,
    )
