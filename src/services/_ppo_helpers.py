"""Private helpers for ``PPOTrainer`` — extracted to honor CLAUDE.md §1 150-LOC cap.

Holds ``_pad`` (pad ``State.X`` to ``(1, V_max, F)`` + bool mask) so the trainer
stays focused on the Schulman 2017/2016 math. Not part of the public PPO API —
import path is intentionally underscored.
"""

from __future__ import annotations

import torch

from src.env.actions import V_MAX_DEFAULT
from src.env.state import State

_F_DIM, _FEAT_NDIM = 16, 2


def pad_state(state: State, v_max: int = V_MAX_DEFAULT) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``State.X`` to ``(1, v_max, F)`` + a ``(1, v_max)`` bool mask."""
    feat = state.X
    n, f = feat.shape if feat.ndim == _FEAT_NDIM else (0, _F_DIM)
    x, m = torch.zeros((1, v_max, f), dtype=torch.float32), torch.zeros((1, v_max), dtype=torch.bool)
    take = min(n, v_max)
    if take > 0:
        x[0, :take], m[0, :take] = feat[:take].float(), True
    return x, m
