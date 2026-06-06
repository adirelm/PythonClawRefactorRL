"""Unit tests for ``src.env.refactor_ops`` — the three Phase-3 pure ops that
close the Phase-2 NOOP-only stub. Every op must (a) return a NEW DiGraph,
(b) leave its input untouched, and (c) preserve the node-attr contract
(``kind``, ``LOC``, ``cyclomatic``, ``layer``, ``lazy_load_flag``) on every
output node — that contract is what downstream metrics (modularity, cohesion,
coupling) depend on per CLAUDE.md §CANONICAL VALUES.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.env.refactor_ops import merge_modules, rewire_edge, split_module

_REQUIRED_ATTRS = ("kind", "LOC", "cyclomatic", "layer", "lazy_load_flag")


def _node_attrs(kind: str = "module", loc: int = 10, cyc: int = 3,
                layer: int = 0, lazy: bool = False) -> dict:  # fmt: skip
    return {"kind": kind, "LOC": loc, "cyclomatic": cyc, "layer": layer, "lazy_load_flag": lazy}


@pytest.fixture()
def parent_with_4_children() -> nx.DiGraph:
    """Parent ``P`` with 4 sorted children C0..C3 and one inbound caller R."""
    g = nx.DiGraph()
    g.add_node("P", **_node_attrs(kind="module", loc=40, cyc=4))
    g.add_node("R", **_node_attrs(kind="module", loc=5, cyc=1))
    for i in range(4):
        g.add_node(f"C{i}", **_node_attrs(kind="function", loc=2, cyc=1))
        g.add_edge("P", f"C{i}", rel_type="call", weight=1.0)
    g.add_edge("R", "P", rel_type="import", weight=1.0)
    return g


def _assert_required_attrs(g: nx.DiGraph) -> None:
    for nid, attrs in g.nodes(data=True):
        missing = [a for a in _REQUIRED_ATTRS if a not in attrs]
        assert not missing, f"node {nid!r} missing attrs {missing}; has {dict(attrs)}"


def test_split_creates_two_new_nodes(parent_with_4_children: nx.DiGraph) -> None:
    out = split_module(parent_with_4_children, "P", split_point=4)
    # mid-point split: 4 children round-split → ~2 children each side
    assert "P_A" in out and "P_B" in out, f"split must mint _A/_B; got {sorted(out.nodes())}"
    assert "P" not in out, "original parent must be removed after split"
    # All 4 children still reachable from exactly one of the two halves.
    reachable = set()
    for child in (f"C{i}" for i in range(4)):
        for src in ("P_A", "P_B"):
            if out.has_edge(src, child):
                reachable.add(child)
    assert reachable == {"C0", "C1", "C2", "C3"}, f"all children must be wired; got {reachable}"


def test_split_preserves_total_loc(parent_with_4_children: nx.DiGraph) -> None:
    """LOC is split proportionally — sum of the two halves equals the parent's LOC.

    The other nodes' LOC are untouched, so the *total* over the new graph
    equals the original total (R: 5, C0..C3: 2 each → 5 + 8 + (P_A + P_B) ==
    5 + 8 + 40 == 53).
    """
    before = sum(int(parent_with_4_children.nodes[n].get("LOC", 0))
                 for n in parent_with_4_children.nodes())  # fmt: skip
    out = split_module(parent_with_4_children, "P", split_point=4)
    after = sum(int(out.nodes[n].get("LOC", 0)) for n in out.nodes())
    assert after == before, f"total LOC must be conserved on split; {after} != {before}"
    assert int(out.nodes["P_A"]["LOC"]) + int(out.nodes["P_B"]["LOC"]) == 40


def test_merge_unions_edges() -> None:
    """Two nodes with disjoint children → merged node has all children."""
    g = nx.DiGraph()
    g.add_node("A", **_node_attrs(loc=10, cyc=2))
    g.add_node("B", **_node_attrs(loc=20, cyc=5))
    for child in ("ca1", "ca2"):
        g.add_node(child, **_node_attrs(kind="function", loc=1))
        g.add_edge("A", child, rel_type="call", weight=1.0)
    for child in ("cb1", "cb2"):
        g.add_node(child, **_node_attrs(kind="function", loc=1))
        g.add_edge("B", child, rel_type="call", weight=1.0)
    out = merge_modules(g, "A", "B")
    assert "B" not in out, "merged-away node must be removed"
    assert "A" in out, "survivor node id must remain"
    succs = set(out.successors("A"))
    assert succs == {"ca1", "ca2", "cb1", "cb2"}, f"edge union failed; got {succs}"


def test_merge_preserves_loc_sum() -> None:
    g = nx.DiGraph()
    g.add_node("A", **_node_attrs(loc=10, cyc=2, lazy=False))
    g.add_node("B", **_node_attrs(loc=25, cyc=7, lazy=True))
    out = merge_modules(g, "A", "B")
    assert int(out.nodes["A"]["LOC"]) == 35, "LOC must be summed on merge"
    assert int(out.nodes["A"]["cyclomatic"]) == 7, "cyclomatic must be max on merge"
    assert bool(out.nodes["A"]["lazy_load_flag"]) is True, "lazy_load_flag must be OR'd"


def test_rewire_changes_target() -> None:
    g = nx.DiGraph()
    for nid in ("A", "B", "C"):
        g.add_node(nid, **_node_attrs())
    g.add_edge("A", "B", rel_type="call", weight=1.0)
    out = rewire_edge(g, "A", "B", "C")
    assert not out.has_edge("A", "B"), "old edge (A,B) must be gone after rewire"
    assert out.has_edge("A", "C"), "new edge (A,C) must exist after rewire"
    assert out["A"]["C"]["rel_type"] == "call", "edge attrs must carry over verbatim"
    assert out["A"]["C"]["weight"] == 1.0


def test_ops_return_new_graph_not_mutate(parent_with_4_children: nx.DiGraph) -> None:
    """All three ops must leave the input graph byte-identical."""
    snapshot_nodes = sorted(parent_with_4_children.nodes(data=False))
    snapshot_edges = sorted(parent_with_4_children.edges(data=False))
    _ = split_module(parent_with_4_children, "P", split_point=4)
    _ = merge_modules(parent_with_4_children, "C0", "C1")
    _ = rewire_edge(parent_with_4_children, "P", "C0", "C1")
    assert sorted(parent_with_4_children.nodes()) == snapshot_nodes
    assert sorted(parent_with_4_children.edges()) == snapshot_edges
    assert parent_with_4_children.nodes["P"]["LOC"] == 40, "parent LOC must be untouched"


def test_ops_preserve_required_attrs(parent_with_4_children: nx.DiGraph) -> None:
    """``kind/LOC/cyclomatic/layer/lazy_load_flag`` must survive every op."""
    after_split = split_module(parent_with_4_children, "P", split_point=4)
    _assert_required_attrs(after_split)
    after_merge = merge_modules(after_split, "P_A", "P_B")
    _assert_required_attrs(after_merge)
    after_rewire = rewire_edge(after_merge, "R", "P_A", "C0")
    _assert_required_attrs(after_rewire)
