"""Tests for src/services/gae_buffer.py — GAE recurrence + terminal mask."""

from __future__ import annotations

import math

import pytest
import torch

from src.services.gae_buffer import Trajectory, compute_gae_advantages, compute_returns
from src.utils.config_loader import load_config


def _traj(
    *,
    rewards: list[float],
    values: list[float],
    dones: list[bool],
) -> Trajectory:
    t = len(rewards)
    return Trajectory(
        states=[None] * t,
        actions=[0] * t,
        log_probs=torch.zeros(t),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.bool),
    )


def test_gae_returns_correct_shape() -> None:
    traj = _traj(rewards=[1.0, 0.5, -0.2, 0.3], values=[0.1, 0.2, 0.0, 0.4], dones=[False] * 4)
    adv = compute_gae_advantages(traj, gamma=0.99, gae_lambda=0.95, last_value=0.0)
    assert adv.shape == (4,)
    assert adv.dtype == traj.values.dtype


def test_gae_terminal_mask() -> None:
    """done=True at t=T-1 → Â_{T-1} == r_{T-1} − V_{T-1} (bootstrap masked)."""
    traj = _traj(rewards=[0.0, 0.0, 1.5], values=[0.1, 0.2, 0.4], dones=[False, False, True])
    # last_value is non-zero on purpose; the mask must zero it out at the terminal step.
    adv = compute_gae_advantages(traj, gamma=0.99, gae_lambda=0.95, last_value=10.0)
    expected_last = 1.5 - 0.4
    assert adv[-1].item() == pytest.approx(expected_last, rel=1e-6)


def test_gae_lambda_zero_equals_td_residual() -> None:
    """λ=0 collapses GAE to the one-step TD residual δ_t."""
    rewards = [0.4, -0.1, 0.7, 0.2]
    values = [0.3, 0.5, 0.6, 0.1]
    dones = [False, False, False, False]
    traj = _traj(rewards=rewards, values=values, dones=dones)
    gamma = 0.99
    last_value = 0.25
    adv = compute_gae_advantages(traj, gamma=gamma, gae_lambda=0.0, last_value=last_value)
    next_values = [*values[1:], last_value]
    expected = [rewards[i] + gamma * next_values[i] - values[i] for i in range(len(rewards))]
    for got, want in zip(adv.tolist(), expected, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-7)


def test_gae_lambda_one_equals_monte_carlo() -> None:
    """λ=1, all done=False → Â_t == Σ_{k≥0} γ^k r_{t+k} (+ γ^{T-t}·last_value) − V_t."""
    rewards = [1.0, 0.5, -0.3, 0.8]
    values = [0.2, 0.1, 0.0, 0.4]
    dones = [False] * 4
    traj = _traj(rewards=rewards, values=values, dones=dones)
    gamma = 0.99
    last_value = 0.5
    adv = compute_gae_advantages(traj, gamma=gamma, gae_lambda=1.0, last_value=last_value)
    t = len(rewards)
    expected = []
    for start in range(t):
        future = 0.0
        for k, idx in enumerate(range(start, t)):
            future += (gamma**k) * rewards[idx]
        # Bootstrap V(s_T) discounted by gamma^{T-start}
        future += (gamma ** (t - start)) * last_value
        expected.append(future - values[start])
    for got, want in zip(adv.tolist(), expected, strict=True):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-7)


def test_compute_returns_equals_advantages_plus_values() -> None:
    adv = torch.tensor([0.3, -0.2, 1.1, 0.0])
    vals = torch.tensor([0.5, 0.4, -0.1, 0.2])
    returns = compute_returns(adv, vals)
    assert torch.allclose(returns, adv + vals)


def test_compute_returns_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_returns(torch.zeros(3), torch.zeros(4))


def test_trajectory_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="rewards"):
        Trajectory(
            states=[None, None],
            actions=[0, 0],
            log_probs=torch.zeros(2),
            rewards=torch.zeros(3),  # wrong length
            values=torch.zeros(2),
            dones=torch.zeros(2, dtype=torch.bool),
        )


def test_gae_uses_canonical_lambda_0_95() -> None:
    """End-to-end: with config.ppo.gae_lambda=0.95, recurrence matches by-hand math."""
    cfg = load_config()
    gamma = float(cfg["ppo"]["gamma"])
    lam = float(cfg["ppo"]["gae_lambda"])
    assert lam == 0.95, "Canonical gae_lambda sealed at 0.95 (brief 2.3)"
    assert gamma == 0.99, "Canonical gamma sealed at 0.99 (PRD-GAE FR-5)"
    rewards = [1.0, 0.0, -0.5]
    values = [0.2, 0.3, 0.1]
    dones = [False, False, False]
    traj = _traj(rewards=rewards, values=values, dones=dones)
    adv = compute_gae_advantages(traj, gamma=gamma, gae_lambda=lam, last_value=0.0)
    # Walk the recurrence by hand (Â_T=0, V(s_T)=last_value=0).
    next_value = 0.0
    next_adv = 0.0
    expected: list[float] = []
    for step in range(len(rewards) - 1, -1, -1):
        mask = 1.0
        delta = rewards[step] + gamma * mask * next_value - values[step]
        next_adv = delta + gamma * lam * mask * next_adv
        expected.append(next_adv)
        next_value = values[step]
    expected.reverse()
    for got, want in zip(adv.tolist(), expected, strict=True):
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-7)
