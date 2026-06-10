"""CLI: programmatic NetworkX visualization (ADR-009 dual-track stand-in).

Renders a matplotlib spring-layout figure of the Skills dependency graph
so the brief's "Obsidian Graph View hero shot" requirement has a
reproducible counterpart that does not depend on the Obsidian desktop
app. Nodes are coloured by layer (L1=Metadata / L2=Instructions /
L3=Resources) and sized by LOC; edges are coloured by ``rel_type``
(call=blue, import=orange, inheritance=green). The output PNG lands at
``results/figures/obsidian_before.png``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend; no display required.
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PICKLE = _REPO_ROOT / "results" / "graphify_output.gpickle"
_DEFAULT_PNG = _REPO_ROOT / "results" / "figures" / "obsidian_before.png"
_SEED = 42
# Layer is stored as int on graph nodes (1/2/3); 0 = code module/class/etc.
_LAYER_COLORS = {1: "lightblue", 2: "lightyellow", 3: "lightcoral", 0: "lightgrey"}
_LAYER_LABELS = {1: "L1 metadata", 2: "L2 instructions", 3: "L3 resources", 0: "code (module/class/fn)"}
_REL_COLORS = {"call": "#1f77b4", "import": "#ff7f0e", "inheritance": "#2ca02c"}
_FALLBACK_COLOR = "#999999"
_LOC_BASE_SIZE = 80
_LOC_SCALE = 8
_LOC_CAP = 500
_TITLE = "PythonClaw — real package dependency graph (BEFORE refactor; 1,190 nodes)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Programmatic NetworkX graph PNG (Obsidian stand-in).")
    parser.add_argument("--pickle", type=Path, default=_DEFAULT_PICKLE, help="Graph pickle to load.")
    parser.add_argument("--out", type=Path, default=_DEFAULT_PNG, help="Output PNG path.")
    return parser.parse_args(argv)


def _load_graph(pickle_path: Path) -> nx.DiGraph:
    # SAFETY: pickle is loaded only from the local artefact written by
    # ``scripts/build_vault.py`` — never user-supplied input.
    with pickle_path.open("rb") as fh:
        return pickle.load(fh)


def _layer_key(raw: object) -> int:
    """Coerce a node's ``layer`` attr to the int keys used by _LAYER_COLORS."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        token = raw.upper().lstrip("L")
        if token.isdigit():
            return int(token)
    return 0


def _styles(graph: nx.DiGraph) -> tuple[list[str], list[float], list[str]]:
    node_c = [
        _LAYER_COLORS.get(_layer_key(graph.nodes[n].get("layer")), _FALLBACK_COLOR) for n in graph.nodes
    ]
    node_s = [
        min(_LOC_BASE_SIZE + _LOC_SCALE * float(graph.nodes[n].get("LOC", 0) or 0), _LOC_CAP)
        for n in graph.nodes
    ]
    edge_c = [
        _REL_COLORS.get(str(d.get("rel_type", "")), _FALLBACK_COLOR) for _, _, d in graph.edges(data=True)
    ]
    return node_c, node_s, edge_c


def _legend_handles(graph: nx.DiGraph) -> list:
    """Build proxy artists for layer (patch) + rel_type (line) legend entries."""
    layers_present = sorted({_layer_key(graph.nodes[n].get("layer")) for n in graph.nodes})
    rels_present = sorted(
        {str(d.get("rel_type", "")) for _, _, d in graph.edges(data=True) if d.get("rel_type")}
    )
    handles: list = []
    for lk in layers_present:
        handles.append(
            Patch(
                facecolor=_LAYER_COLORS.get(lk, _FALLBACK_COLOR),
                edgecolor="black",
                label=_LAYER_LABELS.get(lk, f"layer={lk}"),
            )
        )
    for rel in rels_present:
        handles.append(
            Line2D([0], [0], color=_REL_COLORS.get(rel, _FALLBACK_COLOR), lw=2, label=f"edge: {rel}")
        )
    return handles


def render(graph: nx.DiGraph, out_path: Path) -> None:
    """Draw the graph with spring_layout(seed=42) and write a PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=_SEED)
    node_c, node_s, edge_c = _styles(graph)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=140)
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_c, node_size=node_s, ax=ax, edgecolors="black", linewidths=0.5
    )
    nx.draw_networkx_edges(graph, pos, edge_color=edge_c, arrows=True, ax=ax, alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    ax.legend(handles=_legend_handles(graph), loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(_TITLE)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    graph = _load_graph(args.pickle)
    render(graph, args.out)
    print(f"Figure written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
