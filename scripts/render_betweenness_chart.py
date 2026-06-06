#!/usr/bin/env -S uv run python
"""Render the F10 deliverable: betweenness 95% CI bar chart + CSV table.

Phase 3, F10 (CLAUDE.md §CANONICAL VALUES — 5 seeds, betweenness 2x/seed).

Reads ``results/training/seed_*/metrics.json`` for each canonical seed,
expects top-level ``initial_betweenness`` and ``final_betweenness`` dicts
(node_id -> centrality in [0, 1] — the shape returned by
``SkillsGraphEnv.final_betweenness()`` / ``CentralityScheduler``).

For every node id that appears in ALL seeds, the script computes:
  * mean ± std of the BEFORE betweenness across the 5 seeds
  * mean ± std of the AFTER  betweenness across the 5 seeds
  * 95% CI half-width (t-distribution, dof = n-1)
  * delta_mean = mean_after - mean_before

Outputs:
  * results/figures/betweenness_ci.png : top-10 nodes by mean_before,
    grouped bars (BEFORE vs AFTER) with 95% CI error bars.
  * results/data/betweenness_table.csv : full per-node table, sorted
    by mean_before descending so the chart's top-10 are the first 10 rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAINING_DIR = REPO_ROOT / "results" / "training"
DEFAULT_OUTPUT_PNG = REPO_ROOT / "results" / "figures" / "betweenness_ci.png"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results" / "data" / "betweenness_table.csv"
TOP_N = 10
MIN_N_FOR_CI = 2  # Student's t needs at least dof=1 -> n>=2
CSV_HEADER = [
    "node_id",
    "mean_before",
    "std_before",
    "ci95_before",
    "mean_after",
    "std_after",
    "ci95_after",
    "delta_mean",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render betweenness 95% CI chart + table.")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args(argv)


def _load_seed_metrics(training_dir: Path) -> list[dict]:
    """Load every ``seed_*/metrics.json`` under ``training_dir`` (sorted by seed)."""
    seed_dirs = sorted(training_dir.glob("seed_*"))
    seeds: list[dict] = []
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        before = data.get("initial_betweenness") or {}
        after = data.get("final_betweenness") or {}
        if isinstance(before, dict) and isinstance(after, dict):
            seeds.append({"seed_dir": seed_dir.name, "before": before, "after": after})
    return seeds


def _common_nodes(seeds: list[dict], key: str) -> set[str]:
    """Intersection of node ids present (for ``key`` in {'before','after'}) in all seeds."""
    if not seeds:
        return set()
    common = set(seeds[0][key].keys())
    for entry in seeds[1:]:
        common &= set(entry[key].keys())
    return common


def _ci95_halfwidth(values: list[float]) -> float:
    """Half-width of the 95% confidence interval via Student's t (dof = n-1)."""
    n = len(values)
    if n < MIN_N_FOR_CI:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std == 0.0 or math.isnan(std):
        return 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return t_crit * std / math.sqrt(n)


def _per_node_stats(seeds: list[dict], node_id: str) -> dict[str, float]:
    """Mean/std/95% CI for one node across the 5 seeds (BEFORE + AFTER)."""
    before_vals = [float(entry["before"].get(node_id, 0.0)) for entry in seeds]
    after_vals = [float(entry["after"].get(node_id, 0.0)) for entry in seeds]
    mean_b, mean_a = float(np.mean(before_vals)), float(np.mean(after_vals))
    return {
        "node_id": node_id,
        "mean_before": mean_b,
        "std_before": float(np.std(before_vals, ddof=1)) if len(before_vals) > 1 else 0.0,
        "ci95_before": _ci95_halfwidth(before_vals),
        "mean_after": mean_a,
        "std_after": float(np.std(after_vals, ddof=1)) if len(after_vals) > 1 else 0.0,
        "ci95_after": _ci95_halfwidth(after_vals),
        "delta_mean": mean_a - mean_b,
    }


def _build_table(seeds: list[dict]) -> list[dict[str, float]]:
    """Per-node aggregated table, sorted by ``mean_before`` descending."""
    nodes = _common_nodes(seeds, "before") & _common_nodes(seeds, "after")
    rows = [_per_node_stats(seeds, nid) for nid in nodes]
    rows.sort(key=lambda r: (-r["mean_before"], str(r["node_id"])))
    return rows


def _write_csv(rows: list[dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row[col] for col in CSV_HEADER])


def _render_png(rows: list[dict[str, float]], output_png: Path, num_seeds: int) -> None:
    """Grouped bar chart: top-10 by mean_before, BEFORE vs AFTER with 95% CI."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    top = rows[:TOP_N]
    labels = [str(r["node_id"]) for r in top]
    before = [r["mean_before"] for r in top]
    after = [r["mean_after"] for r in top]
    err_before = [r["ci95_before"] for r in top]
    err_after = [r["ci95_after"] for r in top]
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2, before, width, yerr=err_before, capsize=4, label="BEFORE (initial)", color="#4C78A8"
    )
    ax.bar(x + width / 2, after, width, yerr=err_after, capsize=4, label="AFTER (final)", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Betweenness centrality")
    ax.set_xlabel("Node id (top-10 by initial betweenness)")
    ax.set_title(f"Betweenness Centrality 95% CI across {num_seeds} seeds, before vs after PPO")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    seeds = _load_seed_metrics(args.training_dir)
    if not seeds:
        print(f"no seed_*/metrics.json found under {args.training_dir}", file=sys.stderr)
        return 1
    rows = _build_table(seeds)
    _write_csv(rows, args.output_csv)
    _render_png(rows, args.output_png, num_seeds=len(seeds))
    print(f"seeds={len(seeds)} nodes={len(rows)} png={args.output_png} csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
