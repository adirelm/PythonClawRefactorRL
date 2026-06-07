#!/usr/bin/env -S uv run python
"""Render the Phase-4 ablation heatmap (D3) + essay D2 marginal summary.

Reads ``results/ablations/seed_table.csv`` (81 cells x 3 seeds = 243 rows
when full; canonical baseline alpha=1.0, beta=1.0, gamma=0.5, p_skills=-5.0).

Sobol-lite (first-order conditional-mean variance ratio) picks the top-2
most-sensitive knobs; the other two are pinned at baseline values, yielding
a 3x3 heatmap of mean_final_reward annotated with mean +- ci95
(Student-t, dof=n-1). Cells with n_ok<3 are hatched. A companion D2 figure
shows each knob's 1-D marginal trend.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "ablations" / "seed_table.csv"
DEFAULT_HEATMAP = REPO_ROOT / "results" / "figures" / "ablation_heatmap.png"
DEFAULT_D2 = REPO_ROOT / "results" / "figures" / "essay_d2_ablation_summary.png"
KNOBS = ("alpha", "beta", "gamma", "p_skills")
BASELINE = {"alpha": 1.0, "beta": 1.0, "gamma": 0.5, "p_skills": -5.0}
MIN_SEEDS_CI = 2
FULL_N_OK = 3
HEATMAP_TITLE = "Ablation heatmap: mean_final_reward across the top-2 most-sensitive knobs"
D2_TITLE = "D2: ablation marginals - mean_final_reward per knob value (95% CI)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation heatmap (D3) + essay D2.")
    p.add_argument("--input", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-heatmap", type=Path, default=DEFAULT_HEATMAP)
    p.add_argument("--out-d2", type=Path, default=DEFAULT_D2)
    return p.parse_args(argv)


def _read_cells(csv_path: Path) -> dict[tuple[float, ...], list[float]]:
    """Return dict keyed by (alpha,beta,gamma,p_skills) -> list of ok rewards."""
    cells: dict[tuple[float, ...], list[float]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            key = tuple(float(row[k]) for k in KNOBS)
            cells[key].append(float(row["final_reward"]))
    return cells


def _sobol_lite(cells: dict[tuple[float, ...], list[float]]) -> dict[str, float]:
    """First-order Sobol-lite index per knob (variance of conditional means / total)."""
    cell_means = {k: float(np.mean(v)) for k, v in cells.items()}
    total = float(np.var(list(cell_means.values())))
    scores: dict[str, float] = {}
    for i, knob in enumerate(KNOBS):
        groups: dict[float, list[float]] = defaultdict(list)
        for k, m in cell_means.items():
            groups[k[i]].append(m)
        cond = [float(np.mean(vs)) for vs in groups.values()]
        scores[knob] = float(np.var(cond)) / total if total > 0 else 0.0
    return scores


def _top2(scores: dict[str, float]) -> tuple[str, str]:
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], KNOBS.index(kv[0])))
    return ranked[0][0], ranked[1][0]


def _ci95(values: list[float]) -> float:
    n = len(values)
    if n < MIN_SEEDS_CI:
        return 0.0
    sd = float(np.std(values, ddof=1))
    if sd == 0.0 or math.isnan(sd):
        return 0.0
    return float(stats.t.ppf(0.975, df=n - 1)) * sd / math.sqrt(n)


def _slice_for_heatmap(cells: dict[tuple[float, ...], list[float]], knob_x: str, knob_y: str) -> dict:
    """Build x/y axis values + mean/ci95/n_ok matrices at baseline for off-axis knobs."""
    ix, iy = KNOBS.index(knob_x), KNOBS.index(knob_y)
    xs = sorted({k[ix] for k in cells})
    ys = sorted({k[iy] for k in cells})
    mean = np.full((len(ys), len(xs)), np.nan)
    ci = np.zeros((len(ys), len(xs)))
    nok = np.zeros((len(ys), len(xs)), dtype=int)
    off = [(i, BASELINE[k]) for i, k in enumerate(KNOBS) if k not in (knob_x, knob_y)]
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            for key, vals in cells.items():
                if key[ix] == xv and key[iy] == yv and all(key[oi] == ov for oi, ov in off):
                    mean[j, i], ci[j, i], nok[j, i] = float(np.mean(vals)), _ci95(vals), len(vals)
    return {"xs": xs, "ys": ys, "mean": mean, "ci": ci, "nok": nok}


def _draw_heatmap(ax: plt.Axes, grid: dict, axes_labels: tuple[str, str]) -> None:
    xs, ys = grid["xs"], grid["ys"]
    mean, ci, nok = grid["mean"], grid["ci"], grid["nok"]
    im = ax.imshow(mean, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs])
    ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    plt.colorbar(im, ax=ax, label="mean_final_reward")
    for j in range(mean.shape[0]):
        for i in range(mean.shape[1]):
            txt = f"{mean[j, i]:+.3f}\n+-{ci[j, i]:.3f}"
            ax.text(i, j, txt, ha="center", va="center", fontsize=9, color="white")
            if nok[j, i] < FULL_N_OK:
                rect = plt.Rectangle(
                    (i - 0.5, j - 0.5), 1, 1, fill=False, hatch="///", edgecolor="grey", lw=0.0
                )
                ax.add_patch(rect)


def render_heatmap(cells: dict[tuple[float, ...], list[float]], knob_x: str, knob_y: str, out: Path) -> None:
    grid = _slice_for_heatmap(cells, knob_x, knob_y)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    _draw_heatmap(ax, grid, (knob_x, knob_y))
    fig.suptitle(HEATMAP_TITLE, fontsize=12, fontweight="bold")
    fixed = ", ".join(f"{k}={BASELINE[k]:g}" for k in KNOBS if k not in (knob_x, knob_y))
    ax.set_title(f"Other knobs fixed at baseline ({fixed})", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_d2(cells: dict[tuple[float, ...], list[float]], out: Path) -> None:
    """1x4 marginal panel: each knob's mean +- ci95 across its 3 values."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=200, sharey=True)
    for ax, knob in zip(axes, KNOBS, strict=True):
        idx = KNOBS.index(knob)
        groups: dict[float, list[float]] = defaultdict(list)
        for key, vals in cells.items():
            groups[key[idx]].extend(vals)
        xs = sorted(groups)
        means = [float(np.mean(groups[x])) for x in xs]
        errs = [_ci95(groups[x]) for x in xs]
        ax.errorbar(xs, means, yerr=errs, fmt="o-", capsize=4, color="#2C3E50", lw=1.6)
        ax.set_title(knob, fontsize=11, fontweight="bold")
        ax.set_xlabel(knob)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("mean_final_reward (95% CI)")
    fig.suptitle(D2_TITLE, fontsize=12, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cells = _read_cells(args.input)
    if not cells:
        print(f"no ok rows in {args.input}", file=sys.stderr)
        return 1
    scores = _sobol_lite(cells)
    knob_x, knob_y = _top2(scores)
    render_heatmap(cells, knob_x, knob_y, args.out_heatmap)
    render_d2(cells, args.out_d2)
    sobol = {k: round(v, 4) for k, v in scores.items()}
    summary = f"cells={len(cells)} top2=({knob_x},{knob_y}) sobol={sobol}"
    print(f"{summary} heatmap={args.out_heatmap} d2={args.out_d2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
