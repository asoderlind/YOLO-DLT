from .train2 import train_model

if __name__ == "__main__":
    train_model(
        name="waymo_dark-yolo11n-cache",
        model="yolo11n.yaml",
        dataset="waymo_dark.yaml",
        epochs=1,
        batch_size=16,
        use_dist=True,
    )
