"""No-edge additive correction model for three-head quantile predictions."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CorrectionMLP(nn.Module):
    """Predict an unbounded three-column delta from frozen base predictions."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden_channels), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.GELU(),
                ]
            )
        self.hidden = nn.Sequential(*layers)
        self.output_linear = nn.Linear(hidden_channels, 3)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(
        self,
        base_predictions: Tensor,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        del edge_index
        return self.output_linear(self.hidden(base_predictions))
