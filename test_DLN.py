from __future__ import print_function
from PIL import Image
import torch
from ultralytics.DLN.model import DLN
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import glob
import random
import time

seed = 1337  # classic number
model_name = "DLN_finetune_LOL"
modelfile = f"ultralytics/DLN/{model_name}.pth"
filename = random.sample(glob.glob("../yolo-testing/datasets/exDark-yolo/images/train/*.jpg"), 1)[0]

trans = transforms.ToTensor()

# model
torch.manual_seed(seed)
model = DLN()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(modelfile, map_location=lambda storage, loc: storage))

print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

model.eval()
model.to("cpu")


def compare_images(image1, image2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image1)
    axs[0].axis("off")
    axs[0].set_title("Input")
    axs[1].imshow(image2)
    axs[1].axis("off")
    axs[1].set_title("Output")
    plt.show()


with torch.no_grad():
    test_image_in = Image.open(filename).convert("RGB")
    test_image_in_tensor = trans(test_image_in)
    test_image_in_tensor = test_image_in_tensor.unsqueeze(0)
    test_image_in_tensor = test_image_in_tensor.to("cpu")  # shape [1, 3, 720, 1280]

    t0 = time.time()
    prediction = model(test_image_in_tensor)  # shape [1, 3, 720, 1280]
    print(f"Inference time taken: {time.time() - t0:.2f} s")

    prediction = prediction.data[0].cpu().numpy().transpose((1, 2, 0))
    prediction = prediction * 255.0
    prediction = prediction.clip(0, 255)  # shape [720, 1280, 3]

    compare_images(np.array(test_image_in), np.uint8(prediction))
