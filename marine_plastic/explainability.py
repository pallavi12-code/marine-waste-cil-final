"""Grad-CAM for the ResNet50 layer4 feature maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from marine_plastic.models import MarineClassifier
from marine_plastic.preprocessing import transform_uploaded_image


def grad_cam(
    model: MarineClassifier,
    image: Image.Image,
    target_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    activations: list[torch.Tensor] = []
    layer = model.cam_target_layer
    handle = layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        model.eval()
        input_tensor = transform_uploaded_image(image).to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            feature_maps = activations[-1]
            gradients = torch.autograd.grad(logits[0, target_index], feature_maps)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
        image_size = input_tensor.shape[-2:]
        heatmap = torch.nn.functional.interpolate(
            heatmap, size=image_size, mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = heatmap.detach().cpu().numpy()
        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum
        original = image.convert("RGB").resize((image_size[1], image_size[0]))
        return np.asarray(original), heatmap
    finally:
        handle.remove()


def overlay_grad_cam(image_array: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    from matplotlib import colormaps

    colored = colormaps["jet"](heatmap)[..., :3]
    base = image_array.astype(np.float32) / 255.0
    blended = np.clip(0.55 * base + 0.45 * colored, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def save_grad_cam(
    model: MarineClassifier,
    record_path: Path,
    target_index: int,
    device: torch.device,
    output_path: Path,
) -> None:
    with Image.open(record_path) as image:
        original, heatmap = grad_cam(model, image, target_index, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_grad_cam(original, heatmap).save(output_path)
