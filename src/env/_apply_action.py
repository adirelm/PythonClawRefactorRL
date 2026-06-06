"""Phase-3 Action → graph mutation primitives extracted from skills_graph_env.

Lives next to ``SkillsGraphEnv`` so the env stays ≤150 LOC (CLAUDE.md §1).
Each ``_split_module`` / ``_merge_modules`` / ``_rewire_edge`` mutates ``graph``
in place and raises ``RefactorOpError`` when the indices land on a no-op
configuration (empty graph, identical endpoints, missing out-edges).
The dispatcher ``apply`` is a pure dict lookup over :class:`ActionKind` so
NOOP / unknown kinds return the graph untouched.
"""

from __future__ import annotations

import logging

from src.env.actions import Action, ActionKind

logger = logging.getLogger(__name__)


class RefactorOpError(RuntimeError):
    """Raised when a refactor op cannot be applied (bad indices, empty graph)."""


def _nid_at(graph, idx: int) -> str:
    nodes = sorted(graph.nodes())
    if not nodes:
        raise RefactorOpError("empty graph")
    return nodes[idx % len(nodes)]


def _split_module(graph, node_idx: int, split_point: int):
    """Shadow-clone half of ``node``'s out-edges onto ``<node>#split{k}``."""
    nid = _nid_at(graph, node_idx)
    out_edges = list(graph.out_edges(nid, data=True))
    if not out_edges:
        raise RefactorOpError(f"split: {nid} has no out-edges")
    pivot = max(1, (len(out_edges) * (split_point + 1)) // 8)
    new_id = f"{nid}#split{split_point}"
    graph.add_node(new_id, **dict(graph.nodes[nid]))
    for _, dst, edata in out_edges[:pivot]:
        graph.add_edge(new_id, dst, **edata)
        graph.remove_edge(nid, dst)
    return graph


def _merge_modules(graph, a_idx: int, b_idx: int):
    """Contract node b into node a — redirect b's edges, then delete b."""
    a, b = _nid_at(graph, a_idx), _nid_at(graph, b_idx)
    if a == b:
        raise RefactorOpError("merge: identical endpoints")
    for _, dst, edata in list(graph.out_edges(b, data=True)):
        if dst != a:
            graph.add_edge(a, dst, **edata)
    for src, _, edata in list(graph.in_edges(b, data=True)):
        if src != a:
            graph.add_edge(src, a, **edata)
    graph.remove_node(b)
    return graph


def _rewire_edge(graph, edge_idx: int, new_target_idx: int):
    """Redirect ``edge[edge_idx]`` onto node at ``new_target_idx``."""
    edges = sorted(graph.edges(data=True))
    if not edges:
        raise RefactorOpError("rewire: no edges")
    src, dst, edata = edges[edge_idx % len(edges)]
    new_dst = _nid_at(graph, new_target_idx)
    if new_dst in (src, dst):
        raise RefactorOpError("rewire: noop target")
    graph.remove_edge(src, dst)
    graph.add_edge(src, new_dst, **edata)
    return graph


def apply(graph, action: Action):
    """Dispatch Action → graph mutation. NOOP unchanged; failed ops log+pass."""
    if action.kind is ActionKind.NOOP:
        return graph
    try:
        if action.kind is ActionKind.SPLIT:
            return _split_module(graph, action.primary, action.secondary)
        if action.kind is ActionKind.MERGE:
            return _merge_modules(graph, action.primary, action.secondary)
        if action.kind is ActionKind.REWIRE:
            return _rewire_edge(graph, action.primary, action.secondary)
    except RefactorOpError as exc:
        logger.warning("refactor op failed (kind=%s): %s", action.kind.name, exc)
    return graph
