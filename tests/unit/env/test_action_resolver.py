"""Unit tests for ``src.env.action_resolver`` (Phase 3 — ACTION_DESIGN §2).

Pins the contract that flat ``Action`` indices map deterministically to
concrete ``(node_id, args)`` tuples consumable by ``refactor_ops``. The
ordering contract (sorted node names + sorted edge tuples) is the
load-bearing invariant — it must match ``State.from_digraph`` exactly so
the policy's mask and the env's resolver agree.
"""

from __future__ import annotations

import networkx as nx

from src.env.action_resolver import (
    resolve_merge,
    resolve_rewire,
    resolve_split,
)
from src.env.actions import K_SPLIT, M_MERGE, Action, ActionKind


def _mk_graph(num_nodes: int, edges: list[tuple[int, int]] | None = None) -> nx.DiGraph:
    """Build a deterministic ``nx.DiGraph`` with string node ids ``n0..n{N-1}``.

    Node attributes mirror what ``GraphifyAdapter`` emits so ``State.from_digraph``
    succeeds: every node gets ``LOC``, ``cyclomatic``, ``kind``, ``layer``,
    ``lazy_load_flag``. Edges default to ``rel_type='call'`` weight 1.0.
    """
    g = nx.DiGraph()
    for i in range(num_nodes):
        g.add_node(
            f"n{i}",
            LOC=10 + i,
            cyclomatic=1.0 + i * 0.1,
            kind="module",
            layer=None,
            lazy_load_flag=False,
        )
    for u, v in edges or []:
        g.add_edge(f"n{u}", f"n{v}", rel_type="call", weight=1.0)
    return g


def test_resolve_split_returns_node_and_point() -> None:
    """SPLIT(primary=k, secondary=p) → (sorted_nodes[k], p) for in-range k, p."""
    graph = _mk_graph(num_nodes=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])
    action = Action(kind=ActionKind.SPLIT, primary=0, secondary=3)
    resolved = resolve_split(action, graph)
    assert resolved is not None
    node_id, split_point = resolved
    assert node_id == "n0"
    assert split_point == 3
    assert 0 <= split_point < K_SPLIT


def test_resolve_split_returns_none_for_out_of_range() -> None:
    """SPLIT with primary ≥ |V| (the mask-leak guard) returns None, not raise."""
    graph = _mk_graph(num_nodes=3)
    action = Action(kind=ActionKind.SPLIT, primary=99, secondary=0)
    assert resolve_split(action, graph) is None


def test_resolve_merge_returns_two_distinct_nodes() -> None:
    """MERGE returns (a, b) with a ≠ b — self-merge is always blocked."""
    graph = _mk_graph(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    action = Action(kind=ActionKind.MERGE, primary=0, secondary=0)
    resolved = resolve_merge(action, graph)
    assert resolved is not None
    node_a, node_b = resolved
    assert node_a == "n0"
    assert node_a != node_b
    assert node_b in {"n1", "n2", "n3"}


def test_resolve_merge_returns_none_for_out_of_range() -> None:
    """MERGE with primary ≥ |V| → None; also None when |V| < 2."""
    graph = _mk_graph(num_nodes=2, edges=[(0, 1)])
    bad = Action(kind=ActionKind.MERGE, primary=99, secondary=0)
    assert resolve_merge(bad, graph) is None
    singleton = _mk_graph(num_nodes=1)
    only = Action(kind=ActionKind.MERGE, primary=0, secondary=0)
    assert resolve_merge(only, singleton) is None
    # secondary past number of available candidates also returns None
    too_far = Action(kind=ActionKind.MERGE, primary=0, secondary=M_MERGE - 1)
    assert resolve_merge(too_far, graph) is None


def test_resolve_rewire_returns_three_nodes() -> None:
    """REWIRE returns (src, old_dst, new_dst) drawn from the live graph."""
    graph = _mk_graph(num_nodes=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)])
    action = Action(kind=ActionKind.REWIRE, primary=0, secondary=0)
    resolved = resolve_rewire(action, graph)
    assert resolved is not None
    src, old_dst, new_dst = resolved
    assert src == "n0" and old_dst == "n1"  # first edge in sorted order
    assert new_dst in {f"n{i}" for i in range(5)}


def test_resolve_rewire_returns_none_for_nonexistent_edge() -> None:
    """REWIRE with primary ≥ |E| returns None — the mask should have caught it."""
    graph = _mk_graph(num_nodes=3, edges=[(0, 1)])
    action = Action(kind=ActionKind.REWIRE, primary=99, secondary=0)
    assert resolve_rewire(action, graph) is None


def test_resolve_returns_none_for_negative_secondary() -> None:
    """All resolvers reject ``secondary < 0`` (defence — mask should pre-block)."""
    graph = _mk_graph(num_nodes=3, edges=[(0, 1), (1, 2)])
    split_bad = Action(kind=ActionKind.SPLIT, primary=0, secondary=-1)
    merge_bad = Action(kind=ActionKind.MERGE, primary=0, secondary=-1)
    rewire_bad = Action(kind=ActionKind.REWIRE, primary=0, secondary=-1)
    assert resolve_split(split_bad, graph) is None
    assert resolve_merge(merge_bad, graph) is None
    assert resolve_rewire(rewire_bad, graph) is None


def test_resolver_uses_canonical_node_ordering() -> None:
    """Two graphs with the same node names but different insertion order MUST
    resolve identically — the resolver sorts node names before indexing.
    """
    # Build the same logical graph two different ways and assert equivalence.
    forward = _mk_graph(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    reverse = nx.DiGraph()
    for i in (3, 2, 1, 0):  # reversed insertion order
        reverse.add_node(
            f"n{i}",
            LOC=10 + i,
            cyclomatic=1.0 + i * 0.1,
            kind="module",
            layer=None,
            lazy_load_flag=False,
        )
    for u, v in [(2, 3), (1, 2), (0, 1)]:  # reversed edge order too
        reverse.add_edge(f"n{u}", f"n{v}", rel_type="call", weight=1.0)
    action = Action(kind=ActionKind.SPLIT, primary=2, secondary=4)
    assert resolve_split(action, forward) == resolve_split(action, reverse)
    rewire = Action(kind=ActionKind.REWIRE, primary=0, secondary=0)
    assert resolve_rewire(rewire, forward) == resolve_rewire(rewire, reverse)
