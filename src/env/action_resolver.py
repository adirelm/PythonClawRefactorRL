"""Resolve flat ``Action``s to concrete (node_id, args) tuples for refactor_ops.

The policy emits ``Action`` instances over the flat ACTION_DESIGN encoding
(``src.env.actions``); ``refactor_ops`` (Phase 3, metrics service) needs
real ``node_id`` / ``edge`` strings against the live ``nx.DiGraph``. This
module bridges those two worlds.

Ordering contract (CRITICAL for determinism — must match
``State.from_digraph`` and ``compute_mask``):

- Nodes are indexed by ``tuple(sorted(graph.nodes()))``.
- Edges are indexed by ``sorted(graph.edges())`` (lexicographic on ``(u,v)``).
- Merge top-M similarity is **cosine over X features** built via
  ``State.from_digraph(graph)`` — same X the mask consumes.
- Rewire top-R candidates are the **lowest combined-degree** nodes (matches
  ``_rewire_mask`` in ``action_mask.py``).

Every resolver returns ``None`` if the action references an index outside
the current graph (defence in depth — the mask should already have blocked
these, but resolvers must not crash mid-rollout).
"""

from __future__ import annotations

import networkx as nx
import torch

from src.env.actions import K_SPLIT, M_MERGE, R_REWIRE, Action
from src.env.state import State

_COL_DIN, _COL_DOUT = 2, 3  # STATE_DESIGN §3 — degree-in / degree-out columns
_COSINE_EPS = 1e-8
_MIN_NODES_FOR_MERGE = 2  # MERGE requires at least one possible partner


def _sorted_nodes(graph: nx.DiGraph) -> list[str]:
    """Canonical node ordering — matches ``State.from_digraph``."""
    return sorted(graph.nodes())


def _sorted_edges(graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Canonical edge ordering — lexicographic ``(u, v)`` for determinism."""
    return sorted(graph.edges())


def resolve_split(action: Action, graph: nx.DiGraph) -> tuple[str, int] | None:
    """Resolve SPLIT to ``(node_id, split_point)``.

    ``action.primary`` ∈ ``[0, V_max)`` maps to the n-th sorted node.
    ``action.secondary`` ∈ ``[0, K_SPLIT)`` is the partition template.
    Returns ``None`` when ``primary`` exceeds the current node count.
    """
    nodes = _sorted_nodes(graph)
    if action.primary >= len(nodes):
        return None
    if not (0 <= action.secondary < K_SPLIT):
        return None
    return nodes[action.primary], int(action.secondary)


def _topm_similar(graph: nx.DiGraph, node_idx: int, m: int) -> list[int]:
    """Top-``m`` cosine-similar node indices to ``node_idx`` (self excluded)."""
    state = State.from_digraph(graph)
    if state.num_nodes < _MIN_NODES_FOR_MERGE:
        return []
    x = state.X[: state.num_nodes].float()
    xn = x / x.norm(dim=1, keepdim=True).clamp_min(_COSINE_EPS)
    sims = xn @ xn[node_idx]
    sims[node_idx] = float("-inf")
    k = min(m, state.num_nodes - 1)
    return torch.topk(sims, k).indices.tolist()


def resolve_merge(action: Action, graph: nx.DiGraph) -> tuple[str, str] | None:
    """Resolve MERGE to ``(node_a, node_b)``.

    ``action.primary`` indexes node A in the sorted node list.
    ``action.secondary`` indexes into the top-``M_MERGE`` cosine-similar
    candidates to A (self excluded). Returns ``None`` if either index is
    out of range for the current graph.
    """
    nodes = _sorted_nodes(graph)
    if action.primary >= len(nodes) or len(nodes) < _MIN_NODES_FOR_MERGE:
        return None
    if not (0 <= action.secondary < M_MERGE):
        return None
    candidates = _topm_similar(graph, action.primary, M_MERGE)
    if action.secondary >= len(candidates):
        return None
    return nodes[action.primary], nodes[candidates[action.secondary]]


def _topr_lowest_degree(graph: nx.DiGraph, r: int) -> list[int]:
    """Top-``r`` lowest combined-degree node indices (matches ``_rewire_mask``)."""
    state = State.from_digraph(graph)
    if state.num_nodes <= 0:
        return []
    degree = state.X[: state.num_nodes, _COL_DIN] + state.X[: state.num_nodes, _COL_DOUT]
    k = min(r, state.num_nodes)
    return torch.topk(degree, k, largest=False).indices.tolist()


def resolve_rewire(action: Action, graph: nx.DiGraph) -> tuple[str, str, str] | None:
    """Resolve REWIRE to ``(src, old_dst, new_dst)``.

    ``action.primary`` indexes into the sorted edge list (``src → old_dst``).
    ``action.secondary`` indexes into the top-``R_REWIRE`` lowest-degree
    candidates for ``new_dst``. Returns ``None`` if either index is invalid.
    """
    edges = _sorted_edges(graph)
    if action.primary >= len(edges):
        return None
    if not (0 <= action.secondary < R_REWIRE):
        return None
    src, old_dst = edges[action.primary]
    candidates = _topr_lowest_degree(graph, R_REWIRE)
    if action.secondary >= len(candidates):
        return None
    nodes = _sorted_nodes(graph)
    new_dst = nodes[candidates[action.secondary]]
    return src, old_dst, new_dst
