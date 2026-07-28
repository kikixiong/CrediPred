"""Utilities for taking immutable in-memory model checkpoints."""

from typing import Dict

from torch import Tensor, nn


def snapshot_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    """Clone a model state so later optimizer steps cannot mutate the snapshot."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
