import cv2
import matplotlib.pyplot as plt


def draw_yolo_bboxes(
    image_path: str,
    label_path: str,
    w: int,
    h: int,
    id2cls: dict,
    classes: list[int],
    max_dist=-1,
    text_size=1,
    text_thickness=2,
):
    print("image_path:", image_path)
    print("label_path:", label_path)
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path, "rt") as f:
        data_lines = f.readlines()

    labels: list[tuple[int, int, int, int, str, str]] = []

    for item in range(len(data_lines)):
        if len(data_lines[item].split(" ")) == 5:
            cls, x_center, y_center, w_box, h_box = data_lines[item].split(" ")
            cls = int(cls)
            dist = -1.0
        else:
            cls, x_center, y_center, w_box, h_box, dist = data_lines[item].split(" ")
            cls = int(cls)
            dist = f"{float(dist) * max_dist:.2f}"

        x1 = int((float(x_center) - float(w_box) / 2) * float(w))
        y1 = int((float(y_center) - float(h_box) / 2) * float(h))
        x2 = int((float(x_center) + float(w_box) / 2) * float(w))
        y2 = int((float(y_center) + float(h_box) / 2) * float(h))

        if cls not in classes:
            continue
        else:
            _class = id2cls[cls] if cls in id2cls else str(cls)
            labels.append((x1, y1, x2, y2, _class, dist))

    print("labels:", labels)
    for label in labels:
        x1, y1, x2, y2, cls, dist = label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls}{',' + str(dist) if dist != -1 else ''}",
            (x1, y1 - 5),
            0,
            text_size,
            (0, 255, 0),
            text_thickness,
        )

    plt.figure(figsize=(10, 6))  # Set figure size
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()
