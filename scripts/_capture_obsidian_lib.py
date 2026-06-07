"""Render + verify helpers for :mod:`scripts.capture_obsidian_after`.

Split out so the CLI entry stays under the 150-LOC cap (CLAUDE.md §1).
These are pure styling / IO helpers — no policy logic, no RL state.
The "before" and "after" PNGs must stay diff-able by eye, so the
colour map / size formula / legend layout live here as constants
that both scripts can import (the "before" stub keeps its own copy
today; convergence is intentional, not mandatory).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend; no display required.
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_LAYER_COLORS = {1: "lightblue", 2: "lightyellow", 3: "lightcoral", 0: "lightgrey"}
_LAYER_LABELS = {
    1: "L1 metadata",
    2: "L2 instructions",
    3: "L3 resources",
    0: "code (module/class/fn)",
}
_REL_COLORS = {"call": "#1f77b4", "import": "#ff7f0e", "inheritance": "#2ca02c"}
_FALLBACK_COLOR = "#999999"
_LOC_BASE_SIZE = 80
_LOC_SCALE = 8
_LOC_CAP = 500
_PNG_MIN_BYTES = 1024  # anti-blank guard — a real labelled chart is well above this
_SPRING_SEED = 42


def _layer_key(raw: object) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        token = raw.upper().lstrip("L")
        if token.isdigit():
            return int(token)
    return 0


def _styles(graph: nx.DiGraph) -> tuple[list[str], list[float], list[str]]:
    node_c = [
        _LAYER_COLORS.get(_layer_key(graph.nodes[n].get("layer")), _FALLBACK_COLOR)
        for n in graph.nodes
    ]
    node_s = [
        min(_LOC_BASE_SIZE + _LOC_SCALE * float(graph.nodes[n].get("LOC", 0) or 0), _LOC_CAP)
        for n in graph.nodes
    ]
    edge_c = [
        _REL_COLORS.get(str(d.get("rel_type", "")), _FALLBACK_COLOR)
        for _, _, d in graph.edges(data=True)
    ]
    return node_c, node_s, edge_c


def _legend_handles(graph: nx.DiGraph) -> list:
    layers = sorted({_layer_key(graph.nodes[n].get("layer")) for n in graph.nodes})
    rels = sorted(
        {str(d.get("rel_type", "")) for _, _, d in graph.edges(data=True) if d.get("rel_type")}
    )
    handles: list = []
    for lk in layers:
        handles.append(
            Patch(
                facecolor=_LAYER_COLORS.get(lk, _FALLBACK_COLOR),
                edgecolor="black",
                label=_LAYER_LABELS.get(lk, f"layer={lk}"),
            )
        )
    for rel in rels:
        handles.append(
            Line2D([0], [0], color=_REL_COLORS.get(rel, _FALLBACK_COLOR), lw=2, label=f"edge: {rel}")
        )
    return handles


def render(graph: nx.DiGraph, out_path: Path, title: str) -> None:
    """Draw refactored graph with spring_layout(seed=42) and write PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=_SPRING_SEED)
    node_c, node_s, edge_c = _styles(graph)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=140)
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_c, node_size=node_s, ax=ax, edgecolors="black", linewidths=0.5
    )
    nx.draw_networkx_edges(graph, pos, edge_color=edge_c, arrows=True, ax=ax, alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    ax.legend(handles=_legend_handles(graph), loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def verify_png(out_path: Path) -> None:
    """Anti-screenshot-debacle: confirm PNG exists and is non-trivial in size."""
    if not out_path.exists():
        raise RuntimeError(f"PNG not written: {out_path}")
    size = out_path.stat().st_size
    if size < _PNG_MIN_BYTES:
        raise RuntimeError(f"PNG suspiciously small ({size} B) — likely blank: {out_path}")
    with out_path.open("rb") as fh:
        header = fh.read(8)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Not a PNG header at {out_path}: {header!r}")
