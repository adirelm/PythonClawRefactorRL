#!/usr/bin/env -S uv run python
# ruff: noqa: RUF001
"""Render the 2x2 per-knob marginal sensitivity plot (Wave 4c Stream C).

For each reward-knob (alpha, beta, gamma, P_skills -- Greek glyphs are the
canonical notation per CLAUDE.md CANONICAL VALUES) the script collapses the
81-cell x 3-seed ablation table into a 1-D marginal: y = mean ``final_reward``
across every (cell, seed) row that pins the knob to a given value while the
other three knobs vary freely. The error band is the 95% half-width via
Student's t (dof = n-1). Each point is annotated with ``n=<rows>``.

Inputs : ``results/ablations/seed_table.csv``
Output : ``results/figures/ablation_marginals.png`` (300 dpi)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "ablations" / "seed_table.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "figures" / "ablation_marginals.png"
KNOBS = (
    ("alpha", "α marginal", "α"),
    ("beta", "β marginal", "β"),
    ("gamma", "γ marginal", "γ"),
    ("p_skills", "P_skills marginal", "P_skills"),
)
MIN_N_FOR_CI = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 2x2 per-knob marginals.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if raw.get("status") != "ok":
                continue
            try:
                rows.append(
                    {
                        "alpha": float(raw["alpha"]),
                        "beta": float(raw["beta"]),
                        "gamma": float(raw["gamma"]),
                        "p_skills": float(raw["p_skills"]),
                        "final_reward": float(raw["final_reward"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _ci95_halfwidth(values: list[float]) -> float:
    n = len(values)
    if n < MIN_N_FOR_CI:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std == 0.0 or math.isnan(std):
        return 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return t_crit * std / math.sqrt(n)


def _marginal(rows: list[dict[str, float]], knob: str) -> list[tuple[float, float, float, int]]:
    """Return ``[(knob_value, mean, ci95, n), ...]`` sorted by knob_value."""
    buckets: dict[float, list[float]] = {}
    for row in rows:
        buckets.setdefault(row[knob], []).append(row["final_reward"])
    out = []
    for value in sorted(buckets):
        samples = buckets[value]
        out.append((value, float(np.mean(samples)), _ci95_halfwidth(samples), len(samples)))
    return out


def _monotonicity(points: list[tuple[float, float, float, int]]) -> str:
    means = [m for _, m, _, _ in points]
    if all(b >= a - 1e-9 for a, b in pairwise(means)):
        return "increasing"
    if all(b <= a + 1e-9 for a, b in pairwise(means)):
        return "decreasing"
    return "non_monotonic"


def _plot_panel(ax, title: str, xlabel: str, points: list[tuple[float, float, float, int]]) -> None:
    xs = [p[0] for p in points]
    means = [p[1] for p in points]
    cis = [p[2] for p in points]
    lower = [m - c for m, c in zip(means, cis, strict=True)]
    upper = [m + c for m, c in zip(means, cis, strict=True)]
    ax.errorbar(
        xs, means, yerr=cis, fmt="o-", color="#4C78A8", ecolor="#888", capsize=5, linewidth=2, markersize=7
    )
    ax.fill_between(xs, lower, upper, color="#4C78A8", alpha=0.15)
    for x, mean, _ci, n in points:
        ax.annotate(f"n={n}", (x, mean), textcoords="offset points", xytext=(6, 8), fontsize=8, color="#444")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean final_reward")
    ax.set_xticks(xs)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def render(rows: list[dict[str, float]], out: Path) -> dict[str, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    directions: dict[str, str] = {}
    for ax, (knob, title, xlabel) in zip(axes.flat, KNOBS, strict=True):
        points = _marginal(rows, knob)
        _plot_panel(ax, title, xlabel, points)
        directions[knob] = _monotonicity(points)
    fig.suptitle("Per-knob marginal sensitivity (other knobs varying)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return directions


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _load_rows(args.input)
    if not rows:
        print(f"no ok rows in {args.input}", file=sys.stderr)
        return 1
    directions = render(rows, args.out)
    summary = " ".join(f"{k}={v}" for k, v in directions.items())
    print(f"rows={len(rows)} out={args.out} {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
