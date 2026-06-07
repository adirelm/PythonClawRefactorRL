#!/usr/bin/env -S uv run python
# ruff: noqa: RUF001
"""Render the D6 learning-curve artifact: mean ± 95% CI reward vs. episode step.

Reads per-seed rewards.csv files from results/training/seed_*/rewards.csv,
aligns them by step index, computes mean ± 95% CI (Student-t, dof=n-1),
and saves results/learning_curves/reward_vs_episode.png.

Usage::

    uv run python scripts/render_learning_curve.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "results" / "training"
OUTPUT_DIR = REPO_ROOT / "results" / "learning_curves"
OUTPUT_PNG = OUTPUT_DIR / "reward_vs_episode.png"
AGGREGATE_JSON = TRAINING_DIR / "aggregate.json"

_T_95_BY_DOF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def _t95(dof: int) -> float:
    return _T_95_BY_DOF.get(dof, 1.96)  # fallback to z for large n


def _read_rewards(seed_dir: Path) -> list[float]:
    csv_path = seed_dir / "rewards.csv"
    if not csv_path.exists():
        return []
    rows: list[float] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(float(row["reward"]))
            except (KeyError, ValueError):
                continue
    return rows


def main() -> None:
    seed_dirs = sorted(TRAINING_DIR.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {TRAINING_DIR}")

    all_rewards: dict[str, list[float]] = {}
    for sd in seed_dirs:
        seed_name = sd.name
        r = _read_rewards(sd)
        if r:
            all_rewards[seed_name] = r

    if not all_rewards:
        raise ValueError("No rewards.csv data found in any seed directory.")

    # Align: pad shorter sequences to the maximum length with the last value.
    max_len = max(len(r) for r in all_rewards.values())
    matrix = np.full((len(all_rewards), max_len), np.nan)
    for i, rewards in enumerate(all_rewards.values()):
        n = len(rewards)
        matrix[i, :n] = rewards
        if n < max_len:
            matrix[i, n:] = rewards[-1]

    # Smooth with a rolling mean (window=10) for visual clarity.
    def _smooth(arr: np.ndarray, w: int = 10) -> np.ndarray:
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    smoothed = np.array([_smooth(row) for row in matrix])
    mean_r = np.nanmean(smoothed, axis=0)
    std_r = np.nanstd(smoothed, axis=0, ddof=1)
    n_seeds = (~np.isnan(smoothed[:, 0])).sum()
    sem_r = std_r / math.sqrt(n_seeds)
    ci95 = _t95(n_seeds - 1) * sem_r
    steps = np.arange(max_len)

    # Load seed list from aggregate.json if available.
    seed_label = f"n={n_seeds} seeds"
    if AGGREGATE_JSON.exists():
        with AGGREGATE_JSON.open(encoding="utf-8") as fh:
            agg = json.load(fh)
        seed_list = agg.get("seeds", [])
        seed_label = f"seeds={{{', '.join(str(s) for s in sorted(seed_list))}}}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(steps, mean_r, color="#2563eb", linewidth=1.5, label=f"mean reward ({seed_label})")
    ax.fill_between(steps, mean_r - ci95, mean_r + ci95, alpha=0.25, color="#2563eb", label="95% CI")
    ax.axhline(0.0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode reward (per-step sum)")
    ax.set_title(
        f"PPO+GAE reward vs. step — PythonClaw skills graph\n"
        f"ε=0.2, λ=0.95, γ=0.99, α=1.0, β=1.0, γ_r=0.5, P_skills=−5.0\n"
        f"{seed_label}, total_steps=256, rolling window=10"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
