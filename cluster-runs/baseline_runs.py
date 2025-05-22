from .train2 import run_seeded_train, run_seeded_val

if __name__ == "__main__":
    classes = 0
    run_seeded_train(
        name="waymo_dark-yolo11n",
        model="yolo11n.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        name="waymo_dark-yolo11n-bic-repc3k2",
        model="dlt-models/yolo11n-bic-repc3k2.yaml",
        dataset="waymo_dark.yaml",
    )
    run_seeded_train(
        name="waymo_dark-yolo11s",
        model="yolo11s.yaml",
        dataset="waymo_dark.yaml",
    )

    run_seeded_val(
        name="waymo_dark-yolo11n",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        name="waymo_dark-yolo11n-bic-repc3k2",
        data="waymo_dark.yaml",
        classes=classes,
    )
    run_seeded_val(
        name="waymo_dark-yolo11s",
        data="waymo_dark.yaml",
        classes=classes,
    )
