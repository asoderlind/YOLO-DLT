import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics.DLN.model import DLN
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np


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
