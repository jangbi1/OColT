from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .labels import CLASS_NAMES, IGNORE_INDEX


N_CLASSES = len(CLASS_NAMES)


def confusion_matrix(prediction: torch.Tensor, target: np.ndarray) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    result = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.long)
    if not valid.any():
        return result
    pred = prediction.detach().to(device="cpu", dtype=torch.long)
    target_tensor = torch.from_numpy(target[valid]).long()
    encoded = target_tensor * N_CLASSES + pred[torch.from_numpy(valid)]
    return torch.bincount(encoded, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    values = confusion.float()
    true_positive = torch.diag(values)
    false_positive = values.sum(0) - true_positive
    false_negative = values.sum(1) - true_positive
    support = values.sum(1)
    precision = true_positive / torch.clamp(true_positive + false_positive, min=1.0)
    recall = true_positive / torch.clamp(true_positive + false_negative, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    iou = true_positive / torch.clamp(
        true_positive + false_positive + false_negative, min=1.0
    )
    weights = support / torch.clamp(support.sum(), min=1.0)
    prediction_count = values.sum(0)
    wmape = (prediction_count - support).abs().sum().item() / max(support.sum().item(), 1.0)
    return {
        "wF1": float((weights * f1).sum().item()),
        "wJacc": float((weights * iou).sum().item()),
        "WMAPE": float(wmape),
        "support": {name: int(value) for name, value in zip(CLASS_NAMES, support.tolist())},
        "per_class_f1": {name: float(value) for name, value in zip(CLASS_NAMES, f1.tolist())},
        "per_class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou.tolist())},
    }


def count_transitions(prediction: torch.Tensor) -> int:
    values = prediction.detach().to(device="cpu", dtype=torch.long)
    return 0 if values.numel() < 2 else int((values[1:] != values[:-1]).sum().item())


def add_transition_summary(
    metrics: dict[str, Any], counts: Sequence[int]
) -> dict[str, Any]:
    values = np.asarray(counts, dtype=np.float64)
    metrics = dict(metrics)
    metrics["transitions_mean"] = float(values.mean()) if len(values) else 0.0
    metrics["transitions_total"] = int(values.sum()) if len(values) else 0
    return metrics
