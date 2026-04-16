import torch
import numpy as np
import cv2


def generate_gradcam(input_tensor, model, target_layer_name="features"):
    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    target_layer = dict(model.named_modules())[target_layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(input_tensor)
    target = output.argmax(dim=1)
    output[0, target].backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))[0, 0]
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
