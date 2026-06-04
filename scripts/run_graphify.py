#!/usr/bin/env -S uv run python
"""CLI driver: run GRAPHIFY end-to-end over a Skills source tree.

Builds the dependency DiGraph via LocalGraphify, prints a summary
(|V|, |E|, edge-type counts), persists it to disk as a gpickle, and
prints a small text chart of the top-10 nodes by total degree.

Covered by integration test i14 (no dedicated unit test per spec).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graphify.local_impl import LocalGraphify  # noqa: E402

DEFAULT_SRC = REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "graphify_output.gpickle"
DEFAULT_SEED = 42
TOP_N = 10


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LocalGraphify on a Skills source tree and persist the resulting DiGraph.",
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Skills source root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output gpickle path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Determinism seed.")
    return parser.parse_args(argv)


def _edge_type_counts(graph: nx.DiGraph) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _u, _v, data in graph.edges(data=True):
        counts[data.get("rel_type", "unknown")] += 1
    return counts


def _print_summary(graph: nx.DiGraph) -> None:
    counts = _edge_type_counts(graph)
    print(f"|V|={graph.number_of_nodes()}  |E|={graph.number_of_edges()}")
    print("edge-type counts:")
    for rel in ("call", "import", "inheritance"):
        print(f"  {rel:<12} {counts.get(rel, 0)}")
    other = {k: v for k, v in counts.items() if k not in {"call", "import", "inheritance"}}
    for rel, n in sorted(other.items()):
        print(f"  {rel:<12} {n}")


def _print_top_degree_chart(graph: nx.DiGraph, top_n: int = TOP_N) -> None:
    if graph.number_of_nodes() == 0:
        print("(no nodes — top-degree chart skipped)")
        return
    ranked = sorted(graph.degree(), key=lambda kv: (-kv[1], str(kv[0])))[:top_n]
    max_deg = ranked[0][1] if ranked and ranked[0][1] > 0 else 1
    print(f"top-{top_n} nodes by degree:")
    for node, deg in ranked:
        bar = "█" * max(1, int(40 * deg / max_deg)) if deg > 0 else ""
        print(f"  {node!s:<40} {deg:>4}  {bar}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph = LocalGraphify().build(args.src, seed=args.seed)
    _print_summary(graph)
    with args.output.open("wb") as fh:
        pickle.dump(graph, fh)
    print(f"Saved to {args.output}")
    _print_top_degree_chart(graph)
    return 0


if __name__ == "__main__":
    sys.exit(main())
