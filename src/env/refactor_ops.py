"""Phase-3 refactor operations on ``nx.DiGraph`` (closes the Phase-2 NOOP-only
stub in ``SkillsGraphEnv._apply_action``). Three pure ops — ``split_module``,
``merge_modules``, ``rewire_edge`` — each returns a NEW DiGraph and never
mutates its input. Node-attr contract (kind, LOC, cyclomatic, layer,
lazy_load_flag) is preserved on every output node so downstream metrics
(modularity, cohesion, coupling) keep their invariants (CLAUDE.md
§CANONICAL VALUES). Split/merge naming is canonical: ``"_A" / "_B"`` suffix
on split outputs; merge keeps ``node_a`` as the survivor id.
"""

from __future__ import annotations

import networkx as nx

# Node-attr contract from src/graphify/_walkers.py (the only place that mints
# nodes today). Must stay in sync with ``EXT_ATTRS`` defaults there.
_REQUIRED_ATTRS: tuple[str, ...] = ("kind", "LOC", "cyclomatic", "layer", "lazy_load_flag")
_DEFAULTS: dict = {"kind": "module", "LOC": 0, "cyclomatic": 1, "layer": 0, "lazy_load_flag": False}


def _attrs_with_defaults(src: dict) -> dict:
    """Return a copy of ``src`` with every required attr present (defaults filled)."""
    out = dict(src)
    for key in _REQUIRED_ATTRS:
        if key not in out:
            out[key] = _DEFAULTS[key]
    return out


def split_module(graph: nx.DiGraph, node_id: str, split_point: int) -> nx.DiGraph:
    """Split ``node_id`` into two children ``<node_id>_A`` / ``<node_id>_B``.

    ``split_point`` ∈ [0..7] partitions the original node's *outgoing* edges:
    children whose sorted-position index ``< k`` go to ``_A``, the rest go
    to ``_B`` where ``k = round(split_point/7 * |children|)``. Incoming edges
    are duplicated onto both halves (every caller still reaches one of the
    new pieces). LOC is split proportionally to child count; cyclomatic is
    halved (floor, ≥1); other attrs are copied verbatim.
    Returns a NEW DiGraph; ``graph`` is untouched.
    """
    if node_id not in graph:
        raise KeyError(f"split_module: {node_id!r} not in graph")
    new_graph = graph.copy()
    base = _attrs_with_defaults(new_graph.nodes[node_id])
    children = sorted(new_graph.successors(node_id))
    k = round((split_point / 7.0) * len(children)) if children else 0
    k = max(0, min(k, len(children)))
    id_a, id_b = f"{node_id}_A", f"{node_id}_B"
    loc = int(base.get("LOC", 0))
    loc_a = (loc * k) // max(len(children), 1) if children else loc // 2
    loc_b = loc - loc_a
    cyc = max(1, int(base.get("cyclomatic") or 1) // 2)
    new_attrs = {**base, "LOC": loc_a, "cyclomatic": cyc}
    new_graph.add_node(id_a, **new_attrs)
    new_graph.add_node(id_b, **{**base, "LOC": loc_b, "cyclomatic": cyc})
    for pred, _, edge_attrs in list(new_graph.in_edges(node_id, data=True)):
        new_graph.add_edge(pred, id_a, **dict(edge_attrs))
        new_graph.add_edge(pred, id_b, **dict(edge_attrs))
    for i, child in enumerate(children):
        edge_attrs = dict(new_graph[node_id][child])
        target = id_a if i < k else id_b
        new_graph.add_edge(target, child, **edge_attrs)
    new_graph.remove_node(node_id)
    return new_graph


def merge_modules(graph: nx.DiGraph, node_a: str, node_b: str) -> nx.DiGraph:
    """Combine ``node_b`` into ``node_a`` (``node_a`` keeps its id as survivor).

    Attribute aggregation:
      * ``LOC``        — summed
      * ``cyclomatic`` — max
      * ``lazy_load_flag`` — boolean OR
      * ``kind`` / ``layer`` — taken from ``node_a`` (precedence)
    Edge union: every (pred, node_b) becomes (pred, node_a); every
    (node_b, succ) becomes (node_a, succ). Self-loops (node_b ↔ node_a)
    are dropped. Edge attrs from ``node_b`` win on collision iff ``node_a``
    had no such edge; otherwise ``node_a``'s edge attrs are preserved.
    Returns a NEW DiGraph.
    """
    if node_a not in graph or node_b not in graph:
        raise KeyError(f"merge_modules: missing node ({node_a!r} or {node_b!r})")
    new_graph = graph.copy()
    if node_a == node_b:
        return new_graph
    attrs_a = _attrs_with_defaults(new_graph.nodes[node_a])
    attrs_b = _attrs_with_defaults(new_graph.nodes[node_b])
    merged = {
        **attrs_a,
        "LOC": int(attrs_a.get("LOC", 0)) + int(attrs_b.get("LOC", 0)),
        "cyclomatic": max(int(attrs_a.get("cyclomatic") or 1), int(attrs_b.get("cyclomatic") or 1)),
        "lazy_load_flag": bool(attrs_a.get("lazy_load_flag")) or bool(attrs_b.get("lazy_load_flag")),
    }
    new_graph.add_node(node_a, **merged)
    for pred, _, edge_attrs in list(new_graph.in_edges(node_b, data=True)):
        if pred == node_a:
            continue
        if not new_graph.has_edge(pred, node_a):
            new_graph.add_edge(pred, node_a, **dict(edge_attrs))
    for _, succ, edge_attrs in list(new_graph.out_edges(node_b, data=True)):
        if succ == node_a:
            continue
        if not new_graph.has_edge(node_a, succ):
            new_graph.add_edge(node_a, succ, **dict(edge_attrs))
    new_graph.remove_node(node_b)
    return new_graph


def rewire_edge(graph: nx.DiGraph, edge_src: str, edge_dst: str, new_target: str) -> nx.DiGraph:
    """Redirect ``(edge_src → edge_dst)`` to ``(edge_src → new_target)``.

    Edge attrs are carried over verbatim. Self-loops are refused (no
    rewire-to-self) — the original edge is left in place. Returns a NEW
    DiGraph; ``graph`` is untouched.
    """
    if not graph.has_edge(edge_src, edge_dst):
        raise KeyError(f"rewire_edge: edge ({edge_src!r}, {edge_dst!r}) not in graph")
    if new_target not in graph:
        raise KeyError(f"rewire_edge: new_target {new_target!r} not in graph")
    new_graph = graph.copy()
    if new_target == edge_src:
        return new_graph
    edge_attrs = dict(new_graph[edge_src][edge_dst])
    new_graph.remove_edge(edge_src, edge_dst)
    new_graph.add_edge(edge_src, new_target, **edge_attrs)
    return new_graph
