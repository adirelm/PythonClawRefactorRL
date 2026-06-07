"""Tests for src/services/ppo_trainer.py — PPO clipped surrogate + GAE."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.env.skills_graph_env import SkillsGraphEnv
from src.model.policy_net import PolicyNet
from src.services.gae_buffer import Trajectory
from src.services.ppo_trainer import PPOTrainer

_REQUIRED_METRIC_KEYS = {"policy_loss", "value_loss", "clip_fraction", "approx_kl"}
_SMOKE_N_STEPS = 8
_SMOKE_TOTAL_STEPS = 24


@pytest.fixture()
def tiny_source_tree(tmp_path: Path) -> Path:
    """Two-file Python tree → SkillsGraphEnv builds a graph with several nodes."""
    (tmp_path / "a.py").write_text(
        "def foo():\n    return 1\n\ndef bar():\n    return foo()\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "from a import foo\nclass C:\n    def m(self): foo()\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def env(tiny_source_tree: Path) -> SkillsGraphEnv:
    return SkillsGraphEnv(tiny_source_tree, seed=42, max_episode_steps=4)


@pytest.fixture()
def policy() -> PolicyNet:
    torch.manual_seed(0)
    return PolicyNet()


@pytest.fixture()
def trainer(env: SkillsGraphEnv, policy: PolicyNet) -> PPOTrainer:
    return PPOTrainer(env, policy, n_steps=_SMOKE_N_STEPS, n_epochs=2, batch_size=4)


def test_trainer_uses_canonical_clip_eps_0_2(trainer: PPOTrainer) -> None:
    """brief §2.3: clip_eps is FIXED at 0.2."""
    assert trainer.clip_eps == 0.2


def test_trainer_uses_canonical_gae_lambda_0_95(trainer: PPOTrainer) -> None:
    """PRD-GAE FR-4: gae_lambda is FIXED at 0.95."""
    assert trainer.gae_lambda == 0.95


def test_trainer_rejects_non_canonical_clip_eps(env: SkillsGraphEnv, policy: PolicyNet) -> None:
    with pytest.raises(ValueError, match="sealed"):
        PPOTrainer(env, policy, clip_eps=0.3)


def test_trainer_rejects_non_canonical_gae_lambda(env: SkillsGraphEnv, policy: PolicyNet) -> None:
    with pytest.raises(ValueError, match="sealed"):
        PPOTrainer(env, policy, gae_lambda=0.9)


def test_collect_rollout_returns_trajectory_of_length_n_steps(trainer: PPOTrainer) -> None:
    """``collect_rollout`` must yield exactly ``n_steps`` transitions."""
    traj = trainer.collect_rollout()
    assert isinstance(traj, Trajectory)
    assert len(traj.states) == _SMOKE_N_STEPS
    assert traj.rewards.shape == (_SMOKE_N_STEPS,)
    assert traj.values.shape == (_SMOKE_N_STEPS,)
    assert traj.log_probs.shape == (_SMOKE_N_STEPS,)
    assert traj.dones.shape == (_SMOKE_N_STEPS,)


def test_compute_loss_returns_two_scalars(trainer: PPOTrainer) -> None:
    """``compute_loss`` returns (policy_loss, value_loss) — both 0-d tensors."""
    traj = trainer.collect_rollout()
    advantages = torch.randn(_SMOKE_N_STEPS)
    returns = torch.randn(_SMOKE_N_STEPS)
    p_loss, v_loss = trainer.compute_loss(traj, advantages, returns)
    assert isinstance(p_loss, torch.Tensor) and p_loss.ndim == 0
    assert isinstance(v_loss, torch.Tensor) and v_loss.ndim == 0
    assert torch.isfinite(p_loss) and torch.isfinite(v_loss)


def test_update_returns_metrics_dict_with_required_keys(trainer: PPOTrainer) -> None:
    """``update`` must return ``{policy_loss, value_loss, clip_fraction, approx_kl}``."""
    traj = trainer.collect_rollout()
    metrics = trainer.update(traj)
    assert isinstance(metrics, dict)
    assert _REQUIRED_METRIC_KEYS.issubset(metrics.keys())
    for k in _REQUIRED_METRIC_KEYS:
        assert isinstance(metrics[k], float)
        # NaN check — common smoke test for PPO numerical health.
        assert metrics[k] == metrics[k]  # NaN != NaN


def test_short_train_run_smoke(trainer: PPOTrainer) -> None:
    """Smoke: a few episodes should run, no NaN, history length matches outer loops."""
    history = trainer.train(total_steps=_SMOKE_TOTAL_STEPS)
    assert isinstance(history, list)
    expected_iters = (_SMOKE_TOTAL_STEPS + _SMOKE_N_STEPS - 1) // _SMOKE_N_STEPS
    assert len(history) == expected_iters
    for record in history:
        for k in _REQUIRED_METRIC_KEYS:
            assert record[k] == record[k], f"NaN in metric {k}: {record}"
        assert record["steps"] >= _SMOKE_N_STEPS


def test_collect_rollout_stores_masks(trainer: PPOTrainer) -> None:
    """Trajectory.masks must be populated (not None) to avoid _eval recomputation."""
    traj = trainer.collect_rollout()
    assert traj.masks is not None
    assert len(traj.masks) == _SMOKE_N_STEPS
    assert all(m.dtype == torch.bool for m in traj.masks)
