import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics.DLN.model import DLN
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np


MAX_DIST = 150


def draw_yolo_bboxes(
    image_path, label_path, w: int, h: int, id2cls: dict, classes: list[int], text_size=1, text_thickness=2
):
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
            dist = f"{float(dist) * MAX_DIST:.2f}"

        x1 = int((float(x_center) - float(w_box) / 2) * float(w))
        y1 = int((float(y_center) - float(h_box) / 2) * float(h))
        x2 = int((float(x_center) + float(w_box) / 2) * float(w))
        y2 = int((float(y_center) + float(h_box) / 2) * float(h))

        if cls not in classes:
            continue
        else:
            _class = id2cls[cls] if cls in id2cls else str(cls)
            labels.append((x1, y1, x2, y2, _class, dist))

    for label in labels:
        x1, y1, x2, y2, cls, dist = label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls},{dist if dist != -1 else ''}",
            (x1, y1 - 5),
            0,
            text_size,
            (255, 255, 255),
            text_thickness,
        )

    plt.figure(figsize=(10, 6))  # Set figure size
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()


def compare_images(image1, image2):
    fig, axs = plt.subplots(1, 2, figsize=(5, 5))
    axs[0].imshow(image1)
    axs[0].axis("off")
    axs[0].set_title("Input")
    axs[1].imshow(image2)
    axs[1].axis("off")
    axs[1].set_title("Output")
    plt.show()


def get_dln_model(dln_chpt="ultralytics/DLN/DLN_finetune_LOL.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    model = DLN()
    checkpoint = torch.load(dln_chpt, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model


def enhance_image(filename, dln):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    trans = transforms.ToTensor()
    torch.manual_seed(0)
    with torch.no_grad():
        test_image_in = Image.open(filename).convert("RGB")
        test_image_in_tensor = trans(test_image_in)
        test_image_in_tensor = test_image_in_tensor.unsqueeze(0)
        test_image_in_tensor = test_image_in_tensor.to(device)  # shape [1, 3, 720, 1280]
        prediction = dln(test_image_in_tensor)  # shape [1, 3, 720, 1280]
        prediction = prediction.data[0].cpu().numpy().transpose((1, 2, 0))
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)  # shape [720, 1280, 3]
    return np.uint8(prediction)
