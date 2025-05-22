from ultralytics import YOLO
from train_conf import (
    DEVICE,
)
import cv2
import glob

model_path = (
    "runs/waymo-noConf.yaml-dlt-models-yolo11n-SPDConv-3.yaml-100e-SGD-dist-scale0.0-mosaic1.0-c-d0.05_/weights/best.pt"
)

# Load a pretrained YOLO11n model
model = YOLO(model_path)  # load a pretrained model (recommended for best results)

# Get all image paths in the folder
img_paths = sorted(glob.glob("../yolo-testing/datasets/waymo-noConf/images/val/*.jpeg"))

for img_path in img_paths:
    results = model(img_path, conf=0.50, iou=0.65, device=DEVICE)

    for r in results:
        img = cv2.imread(img_path)
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        boxes = r.boxes.data
        boxes = boxes[boxes[:, 5] == 0, :4]  # Filter boxes with class 0

        overlay = img.copy()
        alpha = 0.5  # Transparency factor (0 = transparent, 1 = opaque)

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = map(int, boxes[i])
            # Draw a vertical black rectangle over the entire height
            cv2.rectangle(overlay, (x1, 0), (x2, img.shape[0]), (0, 0, 0), -1)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.imshow("YOLO Detection", img)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
