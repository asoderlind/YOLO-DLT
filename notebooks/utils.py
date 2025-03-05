import cv2
import matplotlib.pyplot as plt


def draw_yolo_bboxes(image_path, label_path, w=1280, h=720, id2cls={0: "person"}):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path, "rt") as f:
        data_lines = f.readlines()

    bboxes = []

    for item in range(len(data_lines)):
        cl = int(data_lines[item].split(" ")[0])
        a = float(data_lines[item].split(" ")[1])
        b = float(data_lines[item].split(" ")[2])
        c = float(data_lines[item].split(" ")[3])
        d = float(data_lines[item].split(" ")[4][:-1])

        x1 = int((a - c / 2) * w)
        y1 = int((b - d / 2) * h)
        x2 = int((a + c / 2) * w)
        y2 = int((b + d / 2) * h)

        bboxes.append([x1, y1, x2, y2, cl])

    for c in bboxes:
        cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), (0, 255, 0), 5)
        _class = id2cls[c[4]] if c[4] in id2cls else str(c[4])
        cv2.putText(img, str(_class), (int((c[0] + c[2]) / 2), int((c[1] + c[3]) / 2)), 0, 1, (255, 255, 255), 3)

    plt.figure(figsize=(10, 6))  # Set figure size
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()
