from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .model import OColT


def load_state_dict(path: str | Path) -> Mapping[str, torch.Tensor]:
    checkpoint_path = Path(path)
    try:
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        value = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(value, Mapping) and "model_state_dict" in value:
        value = value["model_state_dict"]
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"checkpoint is not a non-empty state_dict: {checkpoint_path}")
    if not all(isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        raise ValueError(f"checkpoint contains non-tensor state entries: {checkpoint_path}")
    return value


def load_model(
    checkpoint_path: str | Path,
    model: OColT,
    device: torch.device,
) -> OColT:
    state = load_state_dict(checkpoint_path)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
