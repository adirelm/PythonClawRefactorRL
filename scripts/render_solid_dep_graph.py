#!/usr/bin/env -S uv run python
"""Render docs/diagrams/solid_dep_graph.png — SOLID layer dependency diagram.

Shows the C4 container layers (cli/notebook → sdk → services → env/model → graphify)
as a directed acyclic graph to visualise the dependency rule: arrows flow downward only.

Usage::

    uv run python scripts/render_solid_dep_graph.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "diagrams"
OUTPUT_PNG = OUTPUT_DIR / "solid_dep_graph.png"

LAYERS = [
    ("CLI / Notebook", "#dbeafe", 5.0),
    ("SDK (RefactorSDK)", "#bfdbfe", 4.0),
    ("Services (PPO, GAE, Metrics, Vault)", "#93c5fd", 3.0),
    ("Env + Model (SkillsGraphEnv, PolicyNet)", "#60a5fa", 2.0),
    ("Graphify + PythonClaw shim", "#3b82f6", 1.0),
    ("Config (config.yaml)", "#1d4ed8", 0.0),
]

EDGES = [
    (5.0, 4.0, "uses only SDK"),
    (4.0, 3.0, "orchestrates"),
    (4.0, 2.0, "builds env"),
    (3.0, 2.0, "calls step/reset"),
    (3.0, 1.0, "reads graph"),
    (2.0, 1.0, "state from graph"),
    (5.0, 0.0, "reads config"),
    (4.0, 0.0, "reads config"),
    (3.0, 0.0, "reads config"),
    (2.0, 0.0, "reads config"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 8), dpi=120)
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(-0.5, 5.8)
    ax.axis("off")

    # Draw layer boxes.
    y_to_label: dict[float, str] = {}
    for label, colour, y in LAYERS:
        rect = mpatches.FancyBboxPatch(
            (0.1, y - 0.35),
            2.6,
            0.7,
            boxstyle="round,pad=0.05",
            facecolor=colour,
            edgecolor="#1e40af",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(1.4, y, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1e3a8a")
        y_to_label[y] = label

    # Draw directed edges (downward arrows).
    _max_span = 1.5  # skip long config lines to reduce clutter; config is global
    for y_from, y_to, _edge_label in EDGES:
        if abs(y_from - y_to) > _max_span:
            continue
        ax.annotate(
            "",
            xy=(1.4, y_to + 0.36),
            xytext=(1.4, y_from - 0.36),
            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.1},
        )

    ax.set_title(
        "SOLID Layer Dependency Rule — arrows flow downward only\n"
        "(ADR-002; SDK is single entry point; no env bypass from CLI)",
        fontsize=10,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
