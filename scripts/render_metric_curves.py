#!/usr/bin/env -S uv run python
# ruff: noqa: RUF001
"""Render the brief §3 per-metric improvement curves (D9b).

Replays each trained policy (``results/training/seed_*/checkpoint.pt``) on the
PythonClaw source graph, records modularity / cohesion / coupling at every step
(``src.services._metric_trace``), aggregates mean ± 95% CI across seeds, writes
``results/data/metric_curves.csv`` and a 3-panel figure
``results/figures/metric_improvement_curves.png`` (modularity ↑, cohesion ↑,
coupling ↓ = the architecture getting better under the trained policy).

Usage::

    uv run python scripts/render_metric_curves.py \
        --source vendor/pythonclaw/pythonclaw --n-steps 128
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.skills_graph_env import SkillsGraphEnv  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.services._metric_trace import policy_metric_rollout  # noqa: E402

TRAINING_DIR = REPO_ROOT / "results" / "training"
DEFAULT_SOURCE = REPO_ROOT / "vendor" / "pythonclaw" / "pythonclaw"
OUTPUT_PNG = REPO_ROOT / "results" / "figures" / "metric_improvement_curves.png"
OUTPUT_CSV = REPO_ROOT / "results" / "data" / "metric_curves.csv"

# (key, panel title, colour, "higher is better"?)
METRICS = [
    ("modularity", "Modularity (Newman–Girvan Q)\n↑ better", "#2563eb", True),
    ("cohesion", "Cohesion\n↑ better", "#059669", True),
    ("coupling", "Coupling penalty\n↓ better", "#dc2626", False),
]
_T95_BY_DOF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def _t95(dof: int) -> float:
    return _T95_BY_DOF.get(dof, 1.96)


def _aggregate(per_seed: dict[str, list[dict[str, float]]]) -> dict[str, dict]:
    """Align per-seed rows by step → mean ± t-CI95 per metric per step."""
    series_lists = list(per_seed.values())
    n_seeds = len(series_lists)
    n_steps = min(len(rows) for rows in series_lists)
    out: dict[str, dict] = {}
    for key, _title, _colour, _hib in METRICS:
        steps = [float(series_lists[0][i]["step"]) for i in range(n_steps)]
        means, cis = [], []
        for i in range(n_steps):
            vals = np.array([rows[i][key] for rows in series_lists], dtype=float)
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1)) / math.sqrt(n_seeds) if n_seeds > 1 else 0.0
            means.append(mean)
            cis.append(_t95(n_seeds - 1) * sem)
        out[key] = {"step": steps, "mean": means, "ci": cis, "n": n_seeds}
    return out


def _write_csv(agg: dict[str, dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "step", "mean", "ci95", "n"])
        for key, _title, _colour, _hib in METRICS:
            s = agg[key]
            for step, mean, ci in zip(s["step"], s["mean"], s["ci"], strict=True):
                writer.writerow([key, step, mean, ci, s["n"]])


def render(agg: dict[str, dict], out_png: Path) -> None:
    """Draw the 3-panel improvement figure and save it to ``out_png``."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = agg[METRICS[0][0]]["n"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=130)
    for ax, (key, title, colour, _hib) in zip(axes, METRICS, strict=True):
        s = agg[key]
        steps = np.array(s["step"])
        mean = np.array(s["mean"])
        ci = np.array(s["ci"])
        ax.plot(steps, mean, color=colour, linewidth=1.6, label=f"mean (n={n})")
        ax.fill_between(steps, mean - ci, mean + ci, color=colour, alpha=0.22, label="95% CI")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Eval-rollout step")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Per-metric improvement under the trained PPO policy — real PythonClaw (1,190-node Skills graph)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _collect(training_dir: Path, source: Path, n_steps: int) -> dict[str, list[dict[str, float]]]:
    """Replay every seed checkpoint → per-step metric rows (heavy path)."""
    rows: dict[str, list[dict[str, float]]] = {}
    for seed_dir in sorted(training_dir.glob("seed_*")):
        ckpt = seed_dir / "checkpoint.pt"
        if not ckpt.exists():
            continue
        seed = int(seed_dir.name.split("_")[1])
        env = SkillsGraphEnv(source, seed=seed)
        policy = PolicyNet()
        policy.load_state_dict(torch.load(ckpt, weights_only=True))
        policy.eval()
        rows[seed_dir.name] = policy_metric_rollout(env, policy, n_steps=n_steps)
        print(f"  replayed {seed_dir.name} ({len(rows[seed_dir.name])} steps)")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render brief §3 per-metric improvement curves.")
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--out", type=Path, default=OUTPUT_PNG)
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args(argv)
    per_seed = _collect(args.training_dir, args.source, args.n_steps)
    if not per_seed:
        raise FileNotFoundError(f"no seed_*/checkpoint.pt under {args.training_dir}")
    agg = _aggregate(per_seed)
    _write_csv(agg, args.out_csv)
    render(agg, args.out)
    print(f"Saved: {args.out}\nSaved: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
