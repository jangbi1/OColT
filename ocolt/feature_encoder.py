from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

ENCODER_MODEL_NAME = "vit_small_patch16_224"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform():
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as error:
        raise RuntimeError("torchvision is required for embedding extraction") from error
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_encoder(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    try:
        import timm
    except ImportError as error:
        raise RuntimeError("timm is required for GastroNet ViT extraction") from error

    checkpoint_path = Path(checkpoint_path)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("encoder checkpoint must be a tensor state_dict")

    model = timm.create_model(
        ENCODER_MODEL_NAME,
        pretrained=False,
        num_classes=0,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_batch(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    result = model(images.to(device, non_blocking=True)).float()
    if result.ndim != 2 or result.shape[1] != 384:
        raise RuntimeError(f"expected ViT features (B,384), got {tuple(result.shape)}")
    return result.cpu()
