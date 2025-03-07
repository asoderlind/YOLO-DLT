from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from notebooks.utils import enhance_image, get_dln_model
import random

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def get_edge_map(img):
    # Pre-blur with a small Gaussian kernel to reduce noise
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply the Scharr operator for a more sensitive edge detection
    grad_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr_edges = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize the edge map to [0,1]
    edge_norm = cv2.normalize(scharr_edges, None, 0, 1, cv2.NORM_MINMAX)

    # Optional: threshold to obtain higher contrast edges
    _, edge_thresh = cv2.threshold((edge_norm * 255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
    return edge_thresh


def show_loss(imgs, model, dataset, img_name):
    model.model.eval()
    print("Model:", model.model_name)

    dln = get_dln_model()

    imgs = torch.tensor(imgs, dtype=torch.float32)
    imgs = imgs / 255.0  # normalize the image
    imgs = imgs.unsqueeze(0)  # add batch dimension
    imgs = imgs.permute(0, 3, 1, 2)  # change the order of the dimensions
    imgs = imgs.to(device)

    enhanced_imgs = dln(imgs)  # shape [bs, 3, h, w]
    enhanced_imgs = enhanced_imgs.to(device)

    predictions = model.model.model[0](imgs)  # shape [bs, 16, h/2, w/2]

    enhanced_img_resized = torch.nn.functional.interpolate(
        enhanced_imgs, size=predictions.shape[-2:]
    )  # shape [bs, 3, h/2, w/2]

    # pool images
    avg_pool_enhanced_imgs = torch.mean(enhanced_img_resized, dim=1)  # shape [bs, 1, h/2, w/2]
    max_pool_enhanced_imgs = torch.max(enhanced_img_resized, dim=1).values  # shape [bs, 1, h/2, w/2]

    # pool predictions
    avg_pool_predictions = torch.mean(predictions, dim=1)  # shape [bs, 1, h/2, w/2]
    max_pool_predictions = torch.max(predictions, dim=1).values  # shape [bs, 1, h/2, w/2]

    try:
        avg_pool_loss = torch.nn.functional.mse_loss(avg_pool_enhanced_imgs, avg_pool_predictions)
        max_pool_loss = torch.nn.functional.mse_loss(max_pool_enhanced_imgs, max_pool_predictions)
    except RuntimeError as e:
        raise RuntimeError(
            "ERROR ❌ Enhanced images and predictions must have the same shape for consistency loss."
        ) from e
    print(f"Avg pool loss {avg_pool_loss:.2f}")
    print(f"Max pool loss {max_pool_loss:.2f}")

    # visualize the difference between the enhanced image and the prediction
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Model: {model.model_name}")

    axs[0, 0].imshow(avg_pool_enhanced_imgs[0].detach().cpu().numpy(), cmap="gray")
    axs[0, 0].set_title("Avg pool enhanced image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(avg_pool_predictions[0].detach().cpu().numpy(), cmap="gray")
    axs[0, 1].set_title("Avg pool prediction")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(max_pool_enhanced_imgs[0].detach().cpu().numpy(), cmap="gray")
    axs[1, 0].set_title("Max pool enhanced image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(max_pool_predictions[0].detach().cpu().numpy(), cmap="gray")
    axs[1, 1].set_title("Max pool prediction")
    axs[1, 1].axis("off")

    # save the figure
    model_name = model.model_name
    model_name = model_name.split("/")[-1]
    plt.savefig(f"../yolo-testing/feature_visualization/{img_name}--{dataset}-{model_name}.png")

    plt.show()


def get_img_features_pools_edges(_model, _img):
    print(f"Model: {_model.model_name}")
    _model.model.eval()
    _blocks = _model.model.model[0]
    _imgs = torch.tensor(_img, dtype=torch.float32)
    _imgs = _imgs / 255.0  # normalize the image
    _imgs = _imgs.unsqueeze(0)  # add batch dimension
    _imgs = _imgs.permute(0, 3, 1, 2)  # change the order of the dimensions
    _predictions = _blocks(_imgs)
    _img_resized = cv2.resize(
        _img, (_predictions[0].shape[2], _predictions[0].shape[1])
    )  # resize the image to the feature map size
    _avg_pool_img_resized_downsampled = np.mean(_img_resized, axis=2)
    _max_pool_img_resized_downsampled = np.max(_img_resized, axis=2)

    _feature_map = _predictions[0].detach().cpu().numpy()
    _channel_feature = np.sum(_feature_map, axis=0)  # sum all the channels
    _avg_pool_channel_feature = np.mean(_feature_map, axis=0)
    _max_pool_channel_feature = np.max(_feature_map, axis=0)
    return (
        _img,
        get_edge_map(_img),
        _avg_pool_img_resized_downsampled,
        _max_pool_img_resized_downsampled,
        _channel_feature,
        _avg_pool_channel_feature,
        _max_pool_channel_feature,
    )


# loss
# avg_pool_loss = F.mse_loss(torch.tensor(avg_pool_channel_feature), torch.tensor(avg_pool_img_resized_downsampled))


def visualize_features(_dataset, _img_name, _model, images):
    # img_recolored, avg_pool_img_resized_downsampled, max_pool_img_resized_downsampled, channel_feature, avg_pool_channel_feature, max_pool_channel_feature, img_edge = get_img_features_pools_edges(model, img)
    # img_enhanced_recolored, _, _, channel_feature_enhanced, _, _, img_enhanced_edge = get_img_features_pools_edges(model, img_enhanced)

    print("Normal image")
    img1_output = get_img_features_pools_edges(_model, images[0])  # normal
    print("Enhanced image")
    img2_output = get_img_features_pools_edges(_model, images[1])  # enhanced

    _avg_pool_predictions = img1_output[5]
    _max_pool_predictions = img1_output[6]

    _avg_pool_enhanced_imgs = img2_output[2]
    _max_pool_enhanced_imgs = img2_output[3]

    avg_pool_loss = torch.nn.functional.mse_loss(
        torch.tensor(_avg_pool_predictions), torch.tensor(_avg_pool_enhanced_imgs)
    )
    print(f"Avg pool loss {avg_pool_loss:.2f}")
    max_pool_loss = torch.nn.functional.mse_loss(
        torch.tensor(_max_pool_predictions), torch.tensor(_max_pool_enhanced_imgs)
    )
    print(f"Max pool loss {max_pool_loss:.2f}")

    names = ["img", "img_e"]
    outputs = [img1_output, img2_output]
    output_labels = ["og", "edge", "avg_pool", "max_pool", "feat_sum", "feat_avg_pool", "feat_max_pool"]

    fig, axs = plt.subplots(len(outputs), len(outputs[0]), figsize=(13, 4))
    # title model name
    fig.suptitle(f"Model: {_model.model_name}")

    for i, output in enumerate(outputs):
        for j, feature in enumerate(output):
            axs[i][j].imshow(feature, cmap="gray")
            axs[i][j].axis("off")
            axs[i][j].set_title(f"{names[i]}_{output_labels[j]}")

    # save the figure
    model_name = _model.model_name
    model_name = model_name.split("/")[-1]
    # plt.savefig(f"../yolo-testing/feature_visualization/{_img_name}--{_dataset}-{model_name}.png")
    # plt.show()


############################################################################################################

target_checkpoints = [
    "runs/detect/exDark128-yolo-fe-20-only-conv1-lr0-0.001/weights/best.pt",
    "yolo11n.pt",
    "runs/detect/exDark128-yolo-fe-20-only-conv1-lr0-0.001/weights/last.pt",
]

# load dataset
# dataset = "bdd128_DLN"
dataset = "exDark128-yolo"
images = glob.glob(f"../yolo-testing/datasets/{dataset}/images/train/2015_00112.jpg")
img_filename = random.sample(images, 1)[0]
name = img_filename.split("/")[-1]
img_enhanced_filename = f"../yolo-testing/datasets/{dataset}/images_enhanced_VOC_LOL/train/{name[:-4]}_enhanced.png"

# load images
img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_enhanced = enhance_image(img_filename)

# compare_images(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_enhanced)

for target_checkpoint in target_checkpoints:
    model = YOLO(target_checkpoint).to(device)
    model.info()
    show_loss(img, model, dataset, name)
    # visualize_features(dataset, name, model, [img, img_enhanced])
