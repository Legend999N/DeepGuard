import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_last_conv_layer(model):
    # 🔥 Automatically find last Conv2d layer
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise Exception("No Conv2d layer found in model")

def generate_heatmap(model, input_tensor):

    model.eval()

    # Enable gradients
    input_tensor = input_tensor.requires_grad_(True)

    # ✅ Auto-detect correct layer
    target_layer = get_last_conv_layer(model)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Convert tensor → image
    img = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return heatmap