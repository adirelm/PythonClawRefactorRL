"""Phase-3 Action → graph mutation dispatcher (extracted from skills_graph_env).

Lives next to ``SkillsGraphEnv`` so the env stays ≤150 LOC (CLAUDE.md §1).
``apply`` decodes the policy's flat ``Action`` into a concrete graph edit by
delegating to the **slot-correct** resolvers in :mod:`src.env.action_resolver`
(which interpret ``Action.secondary`` exactly as ``compute_mask`` does — a rank
into the top-M cosine-similar / top-R lowest-degree candidate list, NOT a raw
node index) and then to the attribute-preserving, pure ops in
:mod:`src.env.refactor_ops`. This keeps the action the agent *executes*
identical to the action its legality mask *promised* (ADR-005 + ACTION_DESIGN
§2): the mask's similarity / lazy-load-break reasoning now applies to the very
partner that gets merged or rewired.

``refactor_ops`` operations are pure (they return a NEW DiGraph and never mutate
their input). Resolvers return ``None`` for an out-of-range index and the ops
raise ``KeyError`` for a missing node/edge; both are treated as a safe no-op
(the graph is returned untouched and the miss is logged) so a rollout never
crashes — the mask should already have blocked these, this is defence in depth.
"""

from __future__ import annotations

import logging

import networkx as nx

from src.env.action_resolver import resolve_merge, resolve_rewire, resolve_split
from src.env.actions import Action, ActionKind
from src.env.refactor_ops import merge_modules, rewire_edge, split_module

logger = logging.getLogger(__name__)


class RefactorOpError(RuntimeError):
    """Raised when a refactor op cannot be applied (bad indices, empty graph)."""


def _split(graph: nx.DiGraph, action: Action) -> nx.DiGraph:
    resolved = resolve_split(action, graph)
    if resolved is None:
        raise RefactorOpError(f"split: index out of range ({action.primary}, {action.secondary})")
    node_id, split_point = resolved
    return split_module(graph, node_id, split_point)


def _merge(graph: nx.DiGraph, action: Action) -> nx.DiGraph:
    resolved = resolve_merge(action, graph)
    if resolved is None:
        raise RefactorOpError(f"merge: index out of range ({action.primary}, {action.secondary})")
    node_a, node_b = resolved
    return merge_modules(graph, node_a, node_b)


def _rewire(graph: nx.DiGraph, action: Action) -> nx.DiGraph:
    resolved = resolve_rewire(action, graph)
    if resolved is None:
        raise RefactorOpError(f"rewire: index out of range ({action.primary}, {action.secondary})")
    src, old_dst, new_dst = resolved
    return rewire_edge(graph, src, old_dst, new_dst)


_DISPATCH = {
    ActionKind.SPLIT: _split,
    ActionKind.MERGE: _merge,
    ActionKind.REWIRE: _rewire,
}


def apply(graph: nx.DiGraph, action: Action) -> nx.DiGraph:
    """Dispatch ``action`` to a slot-correct, attr-preserving graph edit.

    NOOP / unknown kinds return ``graph`` unchanged. A refactor that cannot be
    realised on the current graph (out-of-range index or missing node/edge) is
    logged and treated as a no-op so the agent learns to avoid it via the
    reward signal rather than crashing the rollout.
    """
    handler = _DISPATCH.get(action.kind)
    if handler is None:  # NOOP or unrecognised kind
        return graph
    try:
        return handler(graph, action)
    except (RefactorOpError, KeyError) as exc:
        logger.warning("refactor op failed (kind=%s): %s", action.kind.name, exc)
        return graph
