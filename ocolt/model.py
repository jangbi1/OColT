from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn.utils import weight_norm


@dataclass(frozen=True)
class OColTConfig:
    input_size: int = 384
    output_size: int = 7
    feature_sizes: tuple[int, ...] = (64,) * 13
    kernel_size: int = 7
    num_convolutions: int = 2
    dropout: float = 0.5
    residual: bool = True
    last_layer: str = "conv"

    @classmethod
    def from_mapping(cls, value: dict) -> "OColTConfig":
        return cls(
            input_size=int(value.get("input_size", 384)),
            output_size=int(value.get("output_size", 7)),
            feature_sizes=tuple(int(x) for x in value.get("feature_sizes", [64] * 13)),
            kernel_size=int(value.get("kernel_size", 7)),
            num_convolutions=int(value.get("num_convolutions", 2)),
            dropout=float(value.get("dropout", 0.5)),
            residual=bool(value.get("residual", True)),
            last_layer=str(value.get("last_layer", "conv")),
        )


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        num_of_convs: int = 2,
        dropout: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if num_of_convs not in (1, 2):
            raise ValueError("num_of_convs must be 1 or 2")
        self.num_of_convs = num_of_convs
        self.residual = residual

        # Keep these aliases and the Sequential container: both names occur in
        # the historical state_dict and are required for strict loading.
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        if num_of_convs == 2:
            self.conv2 = weight_norm(
                nn.Conv1d(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )

        layers: list[nn.Module] = [self.conv1, nn.ReLU(), nn.Dropout(dropout)]
        if num_of_convs == 2:
            layers.extend([self.conv2, nn.ReLU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        output = self.net(inputs[0])
        if self.residual:
            residual = inputs[0] if self.downsample is None else self.downsample(inputs[0])
            output = self.relu(output + residual)
        if len(inputs) == 1:
            return [output]
        mask = inputs[1]
        return [output * mask.unsqueeze(1), mask]


class TemporalNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: Sequence[int],
        num_of_convs: int,
        kernel_size: int,
        dropout: float,
        residual: bool,
        last_layer: str,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels cannot be empty")
        if last_layer not in ("linear", "conv"):
            raise ValueError("last_layer must be 'linear' or 'conv'")
        self.last_layer_type = last_layer
        self.output_size = output_size

        blocks: list[nn.Module] = []
        for index, out_channels in enumerate(channels):
            dilation = 2**index
            in_channels = input_size if index == 0 else int(channels[index - 1])
            padding = (kernel_size - 1) * dilation // 2
            blocks.append(
                TemporalBlock(
                    in_channels,
                    int(out_channels),
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    num_of_convs=num_of_convs,
                    dropout=dropout,
                    residual=residual,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.last_layer = (
            nn.Linear(int(channels[-1]), output_size)
            if last_layer == "linear"
            else nn.Conv1d(int(channels[-1]), output_size, 1)
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = (
            self.network([inputs, mask])[0] * mask.unsqueeze(1)
            if mask is not None
            else self.network([inputs])[0]
        )
        if self.last_layer_type == "linear":
            logits = self.last_layer(features.permute(0, 2, 1)).float()
        else:
            logits = self.last_layer(features).float().permute(0, 2, 1)
        return logits * mask.unsqueeze(2) if mask is not None else logits


class OColT(nn.Module):
    def __init__(self, config: OColTConfig | None = None) -> None:
        super().__init__()
        self.config = config or OColTConfig()
        cfg = self.config
        self.conv_first_layer = nn.Conv1d(
            cfg.input_size, cfg.feature_sizes[0], kernel_size=1
        )
        self.conv_first_layer_relu = nn.ReLU()
        self.stage1 = TemporalNetwork(
            input_size=cfg.feature_sizes[0],
            output_size=cfg.output_size,
            channels=cfg.feature_sizes,
            kernel_size=cfg.kernel_size,
            num_of_convs=cfg.num_convolutions,
            dropout=cfg.dropout,
            residual=cfg.residual,
            last_layer=cfg.last_layer,
        )

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"expected (B,D,T), got {tuple(embeddings.shape)}")
        if embeddings.shape[1] != self.config.input_size:
            raise ValueError(
                f"expected embedding dimension {self.config.input_size}, "
                f"got {embeddings.shape[1]}"
            )
        features = self.conv_first_layer_relu(self.conv_first_layer(embeddings))
        return self.stage1(features, mask)
