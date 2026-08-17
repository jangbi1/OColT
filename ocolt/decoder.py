from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OCDConfig:
    backward_penalty: float = 10.0
    skip_penalty: float = 5.0
    start_state: int = -1
    smoothing_window: int = 3

    @classmethod
    def from_mapping(cls, value: dict) -> "OCDConfig":
        return cls(
            backward_penalty=float(value.get("backward_penalty", 10.0)),
            skip_penalty=float(value.get("skip_penalty", 5.0)),
            start_state=int(value.get("start_state", -1)),
            smoothing_window=int(value.get("smoothing_window", 3)),
        )

    def validate(self, class_count: int) -> None:
        if self.backward_penalty < 0 or self.skip_penalty < 0:
            raise ValueError("OCD penalties must be non-negative")
        if self.start_state < -1 or self.start_state >= class_count:
            raise ValueError(f"start_state must be -1 or 0..{class_count - 1}")
        if self.smoothing_window < 1 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer")


def smooth_probabilities(probabilities: torch.Tensor, window: int) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (T,C)")
    if window <= 1:
        return probabilities
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    length = probabilities.shape[0]
    values = probabilities.transpose(0, 1).unsqueeze(0)
    values = F.pad(values, (window // 2, window // 2), mode="replicate")
    return F.avg_pool1d(values, kernel_size=window, stride=1)[..., :length].squeeze(0).transpose(0, 1)


@torch.no_grad()
def ordinal_constrained_decode(
    logits: torch.Tensor, config: OCDConfig | None = None
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"expected logits (T,C), got {tuple(logits.shape)}")
    length, class_count = logits.shape
    if length == 0:
        raise ValueError("cannot decode an empty sequence")
    cfg = config or OCDConfig()
    cfg.validate(class_count)

    probabilities = torch.softmax(logits, dim=1)
    probabilities = smooth_probabilities(probabilities, cfg.smoothing_window)
    log_probabilities = torch.log(probabilities + 1e-8)

    transition = torch.zeros(
        (class_count, class_count), dtype=logits.dtype, device=logits.device
    )
    # transition[current, previous]
    for previous in range(class_count):
        for current in range(class_count):
            if current == previous or current == previous + 1:
                penalty = 0.0
            elif current < previous:
                penalty = -cfg.backward_penalty * (previous - current)
            else:
                penalty = -cfg.skip_penalty * (current - previous - 1)
            transition[current, previous] = penalty

    scores = torch.full(
        (length, class_count), -1e18, dtype=logits.dtype, device=logits.device
    )
    predecessor = torch.full(
        (length, class_count), -1, dtype=torch.long, device=logits.device
    )
    if cfg.start_state < 0:
        scores[0] = log_probabilities[0]
    else:
        scores[0, cfg.start_state] = log_probabilities[0, cfg.start_state]

    for frame in range(1, length):
        candidates = scores[frame - 1].view(1, class_count) + transition
        best_score, best_previous = candidates.max(dim=1)
        scores[frame] = best_score + log_probabilities[frame]
        predecessor[frame] = best_previous

    path = torch.zeros(length, dtype=torch.long, device=logits.device)
    path[-1] = scores[-1].argmax()
    for frame in range(length - 1, 0, -1):
        path[frame - 1] = predecessor[frame, path[frame]]
    return path
