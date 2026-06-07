"""GAE (Generalized Advantage Estimation) buffer -- Schulman 2016 Eq. 11 + 16.

Implements terminal-mask-correct advantage computation consumed by the PPO
trainer. Canonical hyperparameters live in ``config.ppo`` (gamma=0.99,
gae_lambda=0.95). This module exposes them as defaults but takes them as
explicit kwargs so the trainer can inject the loaded values for end-to-end
traceability.

Recurrence (per timestep, walking backward from T-1 to 0)::

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_hat_t = delta_t + gamma * gae_lambda * (1 - done_t) * A_hat_{t+1}

with ``A_hat_T := 0`` boundary and ``V(s_T) := last_value`` (bootstrap from
the final state when the rollout was truncated, or 0 when the episode
terminated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Trajectory:
    """Rollout buffer for a single trajectory of length T."""

    states: list[Any]  # length T — opaque State payloads (kept generic for testability)
    actions: list[int]  # length T
    log_probs: torch.Tensor  # shape (T,)
    rewards: torch.Tensor  # shape (T,)
    values: torch.Tensor  # shape (T,)
    dones: torch.Tensor  # shape (T,) bool
    masks: list[Any] | None = None  # shape (T, A_MAX_TOTAL) bool; stored at rollout time, replayed in _eval

    def __post_init__(self) -> None:
        t = len(self.states)
        for name, tensor in (
            ("log_probs", self.log_probs),
            ("rewards", self.rewards),
            ("values", self.values),
            ("dones", self.dones),
        ):
            if tensor.shape != (t,):
                raise ValueError(f"Trajectory.{name} must have shape ({t},), got {tuple(tensor.shape)}")
        if len(self.actions) != t:
            raise ValueError(f"Trajectory.actions length {len(self.actions)} != states length {t}")
        if self.masks is not None and len(self.masks) != t:
            raise ValueError(f"Trajectory.masks length {len(self.masks)} != states length {t}")


def compute_gae_advantages(
    trajectory: Trajectory,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> torch.Tensor:
    """Return advantage A_hat_t per timestep via GAE with terminal mask.

    Args:
        trajectory: Rollout with rewards/values/dones of length T.
        gamma: Discount factor (canonical 0.99 -- config.ppo.gamma).
        gae_lambda: GAE lambda (canonical 0.95 -- config.ppo.gae_lambda).
        last_value: V(s_T) bootstrap; 0.0 when the episode terminated.

    Returns:
        Tensor of shape (T,) dtype matching ``values``.
    """
    rewards = trajectory.rewards
    values = trajectory.values
    dones = trajectory.dones.to(dtype=values.dtype)
    t = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    next_value = float(last_value)
    next_advantage = 0.0
    for step in range(t - 1, -1, -1):
        mask = 1.0 - float(dones[step].item())
        delta = float(rewards[step].item()) + gamma * mask * next_value - float(values[step].item())
        next_advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[step] = next_advantage
        next_value = float(values[step].item())
    return advantages


def compute_returns(advantages: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Value-function target: R_t = Â_t + V(s_t) (PPO/A2C idiom)."""
    if advantages.shape != values.shape:
        raise ValueError(f"advantages shape {tuple(advantages.shape)} != values shape {tuple(values.shape)}")
    return advantages + values
