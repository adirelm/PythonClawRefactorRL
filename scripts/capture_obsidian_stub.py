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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PICKLE = _REPO_ROOT / "results" / "graphify_output.gpickle"
_DEFAULT_PNG = _REPO_ROOT / "results" / "figures" / "obsidian_before.png"
_SEED = 42
_LAYER_COLORS = {"L1": "#4C9AFF", "L2": "#FFAB00", "L3": "#36B37E"}
_REL_COLORS = {"call": "#1f77b4", "import": "#ff7f0e", "inheritance": "#2ca02c"}
_FALLBACK_COLOR = "#999999"
_LOC_BASE_SIZE = 80
_LOC_SCALE = 12
_TITLE = "PythonClaw Skills shim — initial dependency graph (BEFORE refactor)"


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


def _styles(graph: nx.DiGraph) -> tuple[list[str], list[float], list[str]]:
    node_c = [_LAYER_COLORS.get(str(graph.nodes[n].get("layer", "")), _FALLBACK_COLOR) for n in graph.nodes]
    node_s = [_LOC_BASE_SIZE + _LOC_SCALE * float(graph.nodes[n].get("LOC", 0) or 0) for n in graph.nodes]
    edge_c = [
        _REL_COLORS.get(str(d.get("rel_type", "")), _FALLBACK_COLOR) for _, _, d in graph.edges(data=True)
    ]
    return node_c, node_s, edge_c


def render(graph: nx.DiGraph, out_path: Path) -> None:
    """Draw the graph with spring_layout(seed=42) and write a PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=_SEED)
    node_c, node_s, edge_c = _styles(graph)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=140)
    nx.draw_networkx_nodes(graph, pos, node_color=node_c, node_size=node_s, ax=ax)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_c, arrows=True, ax=ax, alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
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
