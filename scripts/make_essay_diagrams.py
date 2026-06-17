#!/usr/bin/env -S uv run python
"""Render the Phase-4 essay architecture diagram (D1).

The Phase-4 brief mandates ≥2 figures in ``docs/ESSAY.md``. D1 is the
deterministic-priors → RL-policy → reward closed loop that justifies the
COMPLEMENTARITY thesis (LocalGraphify priors x LLM/PPO semantic judgment,
A_max = 45057 boundary). D2 (learning curves / ablation summary) is
rendered by a sibling script once AB-EXEC lands.

Pure matplotlib so the figure is reproducible from the repo without any
external service. Style stays clean-academic: white background, one
subplot, no clutter, dpi=200.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_FILENAME = "essay_d1_architecture.png"
TITLE = "A4 architecture: deterministic priors → RL policy → reward feedback"
CAPTION = "D1: GRAPHIFY → state → action → reward closed loop"

# (label, sub-label) for each box, ordered left-to-right / top-to-bottom.
NODES: list[tuple[str, str]] = [
    ("PythonClaw source", "Python repo under refactor"),
    ("LocalGraphify", "deterministic AST + import priors"),
    ("SkillsGraphEnv state", "V≤512, edges, node features"),
    ("PolicyNet", "actor + critic, A_max=45057"),
    ("Action", "SPLIT / MERGE / REWIRE / NOOP"),
    ("_apply_action", "slot-correct resolver → refactor_ops"),
    ("compute_reward", "α·ΔMod + β·ΔCoh − γ·Coup + P_skills"),  # noqa: RUF001
    ("GAE buffer", "λ=0.95, γ=0.99"),  # noqa: RUF001
    ("PPO update", "clipped surrogate, ε=0.2"),
]
# 3 rows x 3 cols layout in axes coordinates (0..1).
COLS = [0.16, 0.50, 0.84]
ROWS = [0.82, 0.52, 0.22]
BOX_W, BOX_H = 0.26, 0.16
BOX_KW = {"boxstyle": "round,pad=0.02", "ec": "#2C3E50", "lw": 1.4, "fc": "#EAF2F8"}
ARROW_KW = {
    "arrowstyle": "-|>",
    "lw": 1.3,
    "color": "#2C3E50",
    "mutation_scale": 14,
    "shrinkA": 6,
    "shrinkB": 6,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render essay architecture diagram (D1).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory to write essay_d1_architecture.png into",
    )
    return parser.parse_args(argv)


def _node_positions() -> list[tuple[float, float]]:
    """Snake order: row0 L→R, row1 R→L, row2 L→R (matches NODES list)."""
    order = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
    return [(COLS[c], ROWS[r]) for r, c in order]


def _draw_box(ax: plt.Axes, x: float, y: float, label: str, sub: str) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2),
            BOX_W,
            BOX_H,
            **BOX_KW,
        )
    )
    ax.text(x, y + 0.025, label, ha="center", va="center", fontsize=10, fontweight="bold", color="#2C3E50")
    ax.text(x, y - 0.030, sub, ha="center", va="center", fontsize=8, color="#34495E", style="italic")


def _draw_arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float]) -> None:
    ax.annotate("", xy=p1, xytext=p0, arrowprops=ARROW_KW)


def _draw_feedback(ax: plt.Axes, positions: list[tuple[float, float]]) -> None:
    """Long curved arrow PPO update -> PolicyNet (closes the RL loop)."""
    x_ppo, y_ppo = positions[8]
    x_policy, y_policy = positions[3]
    ax.annotate(
        "",
        xy=(x_policy, y_policy - BOX_H / 2 - 0.005),
        xytext=(x_ppo, y_ppo + BOX_H / 2 + 0.005),
        arrowprops={**ARROW_KW, "color": "#C0392B", "connectionstyle": "arc3,rad=0.25", "lw": 1.6},
    )
    ax.text(
        0.92, 0.36, "gradient\nupdate", ha="center", va="center", fontsize=8, color="#C0392B", style="italic"
    )


def render(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(TITLE, fontsize=13, fontweight="bold", color="#2C3E50", pad=14)

    positions = _node_positions()
    for (x, y), (label, sub) in zip(positions, NODES, strict=True):
        _draw_box(ax, x, y, label, sub)
    for i in range(len(positions) - 1):
        _draw_arrow(ax, positions[i], positions[i + 1])
    _draw_feedback(ax, positions)

    ax.text(0.5, 0.04, CAPTION, ha="center", va="center", fontsize=10, color="#2C3E50", style="italic")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = args.output_dir / OUTPUT_FILENAME
    render(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
