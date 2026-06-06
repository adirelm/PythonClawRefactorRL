"""Integration smoke test — PPO trainer + scripts/train_ppo.py end-to-end.

This is the OVERALL Phase-3 acceptance test (i12 in the build plan). It
exercises the full custom-PPO stack against the sample skills fixture:

* ``test_ppo_smoke_in_process`` -- builds env/policy/trainer directly,
  runs the smallest possible training budget (``total_steps=128`` ==
  one rollout-update iteration), then verifies:

  - rollout produced finite rewards,
  - at least one update step ran with a finite ``value_loss``,
  - the canonical "betweenness exactly 2x/seed" budget is honoured
    after ``env.final_betweenness()`` (CLAUDE.md sealed value).

* ``test_train_ppo_script_smoke`` -- subprocess-runs the production CLI
  driver ``scripts/train_ppo.py`` against one seed with the smallest
  rollout-aligned budget (``--total-steps 64`` triggers exactly one
  ``n_steps=128`` rollout). It then asserts the script exited 0 and
  wrote ``results/training/seed_42/metrics.json`` so the persistence
  contract is honoured end-to-end.

The brief §2.2 NO-Gymnasium ban is exercised implicitly: importing
``SkillsGraphEnv``/``PPOTrainer`` here would fail loudly if either
re-introduced a ``gymnasium`` dependency.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.env.skills_graph_env import SkillsGraphEnv
from src.model.policy_net import PolicyNet
from src.services.ppo_trainer import PPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SKILLS = REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_ppo.py"
SMOKE_SEED = 42
SMOKE_TOTAL_STEPS = 128  # == PPOTrainer.n_steps default → exactly 1 iter
SCRIPT_TOTAL_STEPS = 64  # script triggers 1 iter (steps += n_steps=128)


def _assert_finite(name: str, value: float) -> None:
    """Fail with a useful message if ``value`` is NaN or ±inf."""
    assert math.isfinite(value), f"{name} must be finite, got {value!r}"


def test_ppo_smoke_in_process() -> None:
    """Build the trainer in-process, run 1 PPO iteration, assert health."""
    env = SkillsGraphEnv(SAMPLE_SKILLS, seed=SMOKE_SEED)
    policy = PolicyNet()
    trainer = PPOTrainer(env, policy)

    history = trainer.train(total_steps=SMOKE_TOTAL_STEPS)

    assert history, "trainer.train returned no iteration metrics"
    last = history[-1]
    for key in ("policy_loss", "value_loss", "clip_fraction", "approx_kl"):
        assert key in last, f"missing PPO metric: {key}"
        _assert_finite(key, float(last[key]))

    # Verify rollouts produced finite rewards (no NaN propagation).
    trajectory = trainer.collect_rollout()
    rewards = trajectory.rewards.tolist()
    assert len(rewards) > 0, "rollout produced zero rewards"
    for idx, reward in enumerate(rewards):
        _assert_finite(f"rewards[{idx}]", float(reward))

    # Brief §2.2 + ADR-006: betweenness must be called EXACTLY 2x/seed.
    # __init__ is call 1; final_betweenness() makes it 2. A 3rd call raises.
    env.final_betweenness()
    assert env.centrality._betweenness_calls == 2, (
        f"betweenness budget violated: {env.centrality._betweenness_calls} calls (canonical: 2)"
    )


@pytest.mark.slow
def test_train_ppo_script_smoke(tmp_path: Path) -> None:
    """Subprocess-run scripts/train_ppo.py and verify persistence contract.

    Uses ``--output-dir`` so the smoke test does not pollute the canonical
    ``results/training/`` tree on developer machines while still exercising
    the same serialization code path the production driver uses.
    """
    output_dir = tmp_path / "training"
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--seeds",
        str(SMOKE_SEED),
        "--total-steps",
        str(SCRIPT_TOTAL_STEPS),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, (
        f"train_ppo.py exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    seed_dir = output_dir / f"seed_{SMOKE_SEED}"
    metrics_path = seed_dir / "metrics.json"
    assert metrics_path.exists(), (
        f"expected metrics.json at {metrics_path}, got: "
        f"{list(seed_dir.glob('*')) if seed_dir.exists() else 'seed dir missing'}"
    )
    payload = json.loads(metrics_path.read_text())
    assert "iterations" in payload and payload["num_iterations"] >= 1, (
        f"metrics.json missing iterations: {payload!r}"
    )
