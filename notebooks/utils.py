import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics.DLN.model import DLN
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np


def draw_yolo_bboxes(image_path, label_path, w=1280, h=720, id2cls={0: "person"}, text_size=1, text_thickness=2):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path, "rt") as f:
        data_lines = f.readlines()

    bboxes = []
    distances = []

    for item in range(len(data_lines)):
        if len(data_lines[item].split(" ")) == 5:
            cl, a, b, c, d = data_lines[item].split(" ")
            cl = int(cl)
            e = None
        if len(data_lines[item].split(" ")) == 6:
            cl, a, b, c, d, e = data_lines[item].split(" ")
            cl = int(cl)
            e = float(e)

        x1 = int((float(a) - float(c) / 2) * float(w))
        y1 = int((float(b) - float(d) / 2) * float(h))
        x2 = int((float(a) + float(c) / 2) * float(w))
        y2 = int((float(b) + float(d) / 2) * float(h))

        bboxes.append([x1, y1, x2, y2, cl])
        distances.append(e)

    for c, d in zip(bboxes, distances):
        cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), (0, 255, 0), 2)
        _class = id2cls[c[4]] if c[4] in id2cls else str(c[4])
        cv2.putText(
            img,
            f"{_class},{d if d is not None else ''}",
            (int((c[0] + c[2]) / 2), int((c[1] + c[3]) / 2)),
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
