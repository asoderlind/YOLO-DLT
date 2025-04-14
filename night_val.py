from ultralytics import YOLO


def val_night(model_path: str) -> None:
    model = YOLO(model_path)
    model.val(data="bdd100k_night.yaml", device="cuda", batch=16)


paths = [
    "runs/detect/bdd100k-yolo11n-spdconv-3/weights/last.pt",
    "runs/detect/bdd100k-yolo11n-spdconv-rfac3k2/weights/last.pt",
    "runs/detect/bdd100k-yolo11n-spdconv-cl/weights/last.pt",
]

if __name__ == "__main__":
    for path in paths:
        val_night(path)
