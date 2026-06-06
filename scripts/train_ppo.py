#!/usr/bin/env -S uv run python
"""5-seed PPO training driver (Phase 3, i11).

Builds :class:`SkillsGraphEnv` + :class:`PolicyNet` + :class:`PPOTrainer`
for every seed in the CLAUDE.md sealed list, runs PPO, and persists
``results/training/seed_{seed}/{checkpoint.pt, metrics.json, rewards.csv}``
plus a top-level ``aggregate.json`` with per-seed final-reward mean ± std.

``metrics.json`` records ``initial_betweenness`` (CALL 1/2 — captured at
env.__init__) and ``final_betweenness`` (CALL 2/2 — invoked here) so the
F10 betweenness chart can read BEFORE/AFTER bars per the brief §3 /
CLAUDE.md §CANONICAL VALUES "exactly 2 betweenness calls per seed" budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.skills_graph_env import SkillsGraphEnv  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.services.ppo_trainer import PPOTrainer  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

DEFAULT_SEEDS = [42, 7, 123, 314, 271]  # CLAUDE.md sealed seed list
SMOKE_STEPS = 5000
DEFAULT_SOURCE = REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "training"

logger = logging.getLogger("train_ppo")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-seed PPO training driver.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--total-steps", type=int, default=SMOKE_STEPS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _set_global_seeds(seed: int) -> None:
    """Seed python ``random``, ``numpy``, and torch (CPU + CUDA if present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_components(seed: int, source_dir: Path, cfg: dict) -> tuple:
    """Construct (env, policy, trainer) wired with canonical PPO constants."""
    env = SkillsGraphEnv(source_dir, seed=seed)
    policy = PolicyNet()
    trainer = PPOTrainer(
        env,
        policy,
        clip_eps=float(cfg["ppo"]["clip_eps"]),
        gae_lambda=float(cfg["ppo"]["gae_lambda"]),
    )
    return env, policy, trainer


def _evaluation_rewards(trainer: PPOTrainer) -> list[float]:
    """One post-training rollout under the trained policy → per-step rewards.

    The trainer's ``train()`` returns per-iter loss metrics but no per-step
    reward trace, so we replay one masked rollout here to populate the CSV.
    Uses ``torch.no_grad`` because no gradient is needed for logging.
    """
    with torch.no_grad():
        trajectory = trainer.collect_rollout()
    return [float(r) for r in trajectory.rewards.tolist()]


def _save_seed_outputs(seed_dir: Path, policy: PolicyNet, payload: dict, rewards: list[float]) -> None:
    """Persist checkpoint.pt + metrics.json + rewards.csv for one seed.

    ``payload`` carries ``iterations`` + ``initial_betweenness`` +
    ``final_betweenness`` + ``betweenness_calls`` — the schema consumed
    by ``scripts/render_betweenness_chart.py`` (F10 deliverable).
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), seed_dir / "checkpoint.pt")
    (seed_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with (seed_dir / "rewards.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "reward"])
        for step, reward in enumerate(rewards):
            writer.writerow([step, float(reward)])


def _final_reward(rewards: list[float]) -> float:
    """Final-reward summary: sum of the eval-rollout reward sequence.

    Sum (not last-step) so a longer rollout that earns more positive ΔMod
    or ΔCohesion shows up as a higher score, which is the comparable
    quantity for cross-seed mean ± std.
    """
    if not rewards:
        return float("nan")
    return float(sum(rewards))


def _run_one_seed(seed: int, args: argparse.Namespace, cfg: dict) -> dict:
    """Train one seed end-to-end; return its summary row for the aggregate."""
    _set_global_seeds(seed)
    env, policy, trainer = _build_components(seed, args.source_dir, cfg)
    initial_btw = dict(env._initial_betweenness)  # captured by env.__init__ (CALL 1/2)
    history = trainer.train(args.total_steps)
    rewards = _evaluation_rewards(trainer)
    final_btw = env.final_betweenness()  # CALL 2/2 — completes the canonical budget
    payload = {
        "iterations": history,
        "num_iterations": len(history),
        "initial_betweenness": initial_btw,
        "final_betweenness": dict(final_btw),
        "betweenness_calls": int(env.centrality.betweenness_calls),
    }
    _save_seed_outputs(args.output_dir / f"seed_{seed}", policy, payload, rewards)
    final = _final_reward(rewards)
    logger.info(
        "seed=%d final_reward=%.6f steps=%d btw_calls=%d",
        seed,
        final,
        args.total_steps,
        env.centrality.betweenness_calls,
    )
    return {"seed": seed, "final_reward": final, "num_reward_points": len(rewards)}


def _is_nan(value: float) -> bool:
    """NaN-safe predicate (avoids polluting the aggregate with NaN seeds)."""
    return math.isnan(value)


def _aggregate(per_seed: list[dict], output_dir: Path, total_steps: int) -> dict:
    """Compute mean ± std final reward and dump ``aggregate.json``."""
    finals = [row["final_reward"] for row in per_seed if not _is_nan(row["final_reward"])]
    aggregate = {
        "seeds": [row["seed"] for row in per_seed],
        "per_seed_final_reward": {str(row["seed"]): row["final_reward"] for row in per_seed},
        "mean_final_reward": float(mean(finals)) if finals else float("nan"),
        "std_final_reward": float(pstdev(finals)) if len(finals) > 1 else 0.0,
        "total_steps_per_seed": int(total_steps),
        "num_seeds": len(per_seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    cfg = load_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = [_run_one_seed(seed, args, cfg) for seed in args.seeds]
    aggregate = _aggregate(per_seed, args.output_dir, args.total_steps)
    print(f"seeds={aggregate['seeds']}")
    print(f"mean_final_reward={aggregate['mean_final_reward']:.6f}")
    print(f"std_final_reward={aggregate['std_final_reward']:.6f}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
