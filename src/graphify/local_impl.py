"""Local NetworkX-backed GraphifyAdapter (ADR-002). Walks .py + Skills JSON →
``nx.DiGraph`` with {kind, LOC, cyclomatic, layer, lazy_load_flag} node and
{rel_type, weight} edge contracts. Methods + nested funcs via ``ast.walk``;
unresolved targets → synthetic ``kind="external"`` nodes (LOC=cyclomatic=layer=0).
Walker helpers split into ``_walkers`` to keep this file ≤150 LOC (CLAUDE.md §1).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx

from src.graphify._walkers import EXT_ATTRS, walk_py, walk_skill_json

logger = logging.getLogger(__name__)


class LocalGraphify:
    """NetworkX-backed concrete GraphifyAdapter (ADR-002, Phase 1)."""

    def build(self, src_root: Path, *, seed: int = 0) -> nx.DiGraph:  # walk .py + .json → DiGraph
        logger.info("LocalGraphify.build(src_root=%s, seed=%d)", src_root, seed)
        root = Path(src_root)
        g: nx.DiGraph = nx.DiGraph()
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" not in py.parts:
                self._absorb(g, *walk_py(py, root))
        for js in sorted(root.rglob("*.json")):
            self._absorb(g, *walk_skill_json(js))
        for nid in list(g.nodes()):
            if "kind" not in g.nodes[nid]:
                g.add_node(nid, **EXT_ATTRS)
        return g

    @staticmethod
    def _absorb(g: nx.DiGraph, nodes: list, edges: list) -> None:
        for nid, attrs in nodes:
            if nid in g and g.nodes[nid].get("kind") not in (None, "external"):
                continue
            g.add_node(nid, **attrs)
        for s, d, attrs in edges:
            g.add_edge(s, d, **attrs)

    def load(self, pickle_path: Path) -> nx.DiGraph:  # read pickled DiGraph; raise FileNotFoundError
        path = Path(pickle_path)
        try:
            with path.open("rb") as fh:
                g = pickle.load(fh)  # internal-only cache (ADR-002)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"LocalGraphify.load: no pickle at {path!s}") from exc
        if not isinstance(g, nx.DiGraph):
            raise TypeError(f"LocalGraphify.load: pickle at {path!s} is not nx.DiGraph")
        return g
