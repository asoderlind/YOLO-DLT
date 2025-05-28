from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

import os


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
        None,
        _avg_pool_img_resized_downsampled,
        _max_pool_img_resized_downsampled,
        _channel_feature,
        _avg_pool_channel_feature,
        _max_pool_channel_feature,
    )


def visualize_features(_model, normal_img, enhanced_img):
    # img_recolored, avg_pool_img_resized_downsampled, max_pool_img_resized_downsampled, channel_feature, avg_pool_channel_feature, max_pool_channel_feature, img_edge = get_img_features_pools_edges(model, img)
    # img_enhanced_recolored, _, _, channel_feature_enhanced, _, _, img_enhanced_edge = get_img_features_pools_edges(model, img_enhanced)

    print("Normal image")
    img1_output = get_img_features_pools_edges(_model, normal_img)  # normal
    print("Enhanced image")
    img2_output = get_img_features_pools_edges(_model, enhanced_img)  # enhanced

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


def main(name: str):
    name_no_ext = name.split(".")[0]

    # load original image
    og_image = cv2.imread(f"sample_images/original/{name}")

    # load dln enhanced image
    dln_image_bgr = cv2.imread(f"sample_images/dln/{name}")
    dln_image = cv2.cvtColor(dln_image_bgr, cv2.COLOR_BGR2RGB)

    # load mbllen enhanced image
    mbllen_image = cv2.imread(f"sample_images/mbllen/{name}")

    # load original model
    og_model = YOLO("runs/bdd100k_night-yolo11n-seed-test-1/weights/best.pt")

    # load dln guided exdark model
    dln_model = YOLO("runs/dln-yolo11n.yaml-c-200e-noDist-d=0.05-fe-lc=0.2-s=0_3/weights/best.pt")

    # load mbllen guided exdark model
    mbllen_model = YOLO("runs/mbllen-yolo11n.yaml-c-200e-noDist-d=0.05-fe-lc=0.1-s=0_2/weights/best.pt")

    if not os.path.exists(f"sample_images/inference/og/{name_no_ext}/image0.jpg"):
        # run inference on the original image with the og model
        og_model(og_image, conf=0.50, iou=0.65, save=True, project="sample_images/inference/og", name=name_no_ext)

    if not os.path.exists(f"sample_images/inference/dln-guided/{name_no_ext}/image0.jpg"):
        # run inference on the dln enhanced image with the dln model
        dln_model(
            og_image, conf=0.50, iou=0.65, save=True, project="sample_images/inference/dln-guided", name=name_no_ext
        )

    if not os.path.exists(f"sample_images/inference/mbllen-guided/{name_no_ext}/image0.jpg"):
        # run inference on the mbllen enhanced image with the mbllen model
        mbllen_model(
            og_image, conf=0.50, iou=0.65, save=True, project="sample_images/inference/mbllen-guided", name=name_no_ext
        )

    og_inference_image = cv2.imread(
        f"sample_images/inference/og/{name_no_ext}/image0.jpg"
    )  # read the inference image from the og model
    dln_inference_image = cv2.imread(
        f"sample_images/inference/dln-guided/{name_no_ext}/image0.jpg"
    )  # read the inference image from the dln model
    mbllen_inference_image = cv2.imread(
        f"sample_images/inference/mbllen-guided/{name_no_ext}/image0.jpg"
    )  # read the inference image from the mbllen model

    # get og model features from original image
    _, _, _, _, _, avg_pool_og, max_pool_og = get_img_features_pools_edges(og_model, og_image)

    # get dln model features from dln enhanced image
    _, _, _, _, _, avg_pool_dln, max_pool_dln = get_img_features_pools_edges(dln_model, og_image)

    # get mbllen model features from mbllen enhanced image
    _, _, _, _, _, avg_pool_mbllen, max_pool_mbllen = get_img_features_pools_edges(mbllen_model, og_image)

    # Plot og, dln_enhaced, mbllen_enhanced on one row
    # avg_pool_og, avg_pool_dln, avg_pool_mbllen on second row
    # max_pool_og, max_pool_dln, max_pool_mbllen on third row

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs[0][0].imshow(og_image[..., ::-1])  # Convert BGR to RGB
    axs[0][0].set_title("Original")
    axs[0][0].axis("off")

    axs[0][1].imshow(dln_image[..., ::-1])  # Convert BGR to RGB
    axs[0][1].set_title("DLN Enhanced")
    axs[0][1].axis("off")

    axs[0][2].imshow(mbllen_image[..., ::-1])  # Convert BGR to RGB
    axs[0][2].set_title("MBLLEN Enhanced")
    axs[0][2].axis("off")

    axs[1][0].imshow(max_pool_og, cmap="gray")
    axs[1][0].set_title("Baseline Max-Pooled Features of Original")
    axs[1][0].axis("off")

    axs[1][1].imshow(max_pool_dln, cmap="gray")
    axs[1][1].set_title("DLN-Guided Max-Pooled Features of Original")
    axs[1][1].axis("off")

    axs[1][2].imshow(max_pool_mbllen, cmap="gray")
    axs[1][2].set_title("MBLLEN-Guided Max-Pooled Features of Original")
    axs[1][2].axis("off")

    axs[2][0].imshow(og_inference_image[..., ::-1])  # Convert BGR to RGB
    axs[2][0].set_title("Inference on Original using Baseline Model")
    axs[2][0].axis("off")

    axs[2][1].imshow(dln_inference_image[..., ::-1])  # Convert BGR to RGB
    axs[2][1].set_title("Inference on Original using DLN-Guided Model")
    axs[2][1].axis("off")

    axs[2][2].imshow(mbllen_inference_image[..., ::-1])  # Convert BGR to RGB
    axs[2][2].set_title("Inference Image on Original using MBLLEN-Guided Model")
    axs[2][2].axis("off")

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"feature_visualization_{name[:-4]}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Visualization for YOLO Models")
    parser.add_argument("--name", type=str, default="2015_01899.jpg", help="Name of the image to visualize features")
    args = parser.parse_args()
    main(args.name)
