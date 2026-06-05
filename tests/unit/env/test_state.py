"""Unit tests for src.env.state.State (Phase 2 — STATE_DESIGN.md contract)."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.env.state import State

# Column indices mirror src/env/state.py (kept local — duplication is intentional
# so any silent reorder of the X contract is caught by red tests).
COL_LOC, COL_CYC = 0, 1
COL_DIN, COL_DOUT = 2, 3
COL_BTW = 4
COL_L1, COL_L2, COL_L3 = 5, 6, 7
COL_LAZY = 8
COL_KCLS, COL_KMOD, COL_KFN = 9, 10, 11
COL_INSK = 12
COL_RES = 15


def _mk_graph() -> nx.DiGraph:
    """Tri-graph: module → class → method, plus an external import target."""
    g: nx.DiGraph = nx.DiGraph()
    g.add_node("m", kind="module", LOC=42, cyclomatic=None, layer=None, lazy_load_flag=False)
    g.add_node("m.Foo", kind="class", LOC=10, cyclomatic=1, layer=None, lazy_load_flag=False)
    g.add_node("m.Foo.bar", kind="method", LOC=5, cyclomatic=2, layer=None, lazy_load_flag=False)
    g.add_node("skill.demo.L1", kind="skill_layer", LOC=8, cyclomatic=None, layer=1, lazy_load_flag=False)
    g.add_node("skill.demo.L2", kind="skill_layer", LOC=6, cyclomatic=None, layer=2, lazy_load_flag=True)
    g.add_edge("m", "m.Foo", rel_type="call", weight=1.0)
    g.add_edge("m.Foo", "m.Foo.bar", rel_type="call", weight=1.0)
    g.add_edge("m.Foo.bar", "m.Foo", rel_type="inheritance", weight=1.0)
    g.add_edge("m", "skill.demo.L1", rel_type="import", weight=1.0)
    return g


def test_from_digraph_returns_state() -> None:
    state = State.from_digraph(_mk_graph())
    assert isinstance(state, State)
    assert isinstance(state.A, sp.csr_matrix)
    assert isinstance(state.X, torch.Tensor)
    assert state.num_nodes == 5
    assert state.num_edges >= 1


def test_x_shape_is_n_v_by_16() -> None:
    state = State.from_digraph(_mk_graph())
    assert state.X.shape == (state.num_nodes, 16)
    assert state.X.dtype == torch.float32


def test_kind_columns_one_hot() -> None:
    """Class / module / function (incl. method) ∈ exactly one of cols 9-11."""
    state = State.from_digraph(_mk_graph())
    by_id = {nid: state.X[i] for i, nid in enumerate(state.node_ids)}
    kind_cols = (COL_KCLS, COL_KMOD, COL_KFN)
    assert [by_id["m"][c] for c in kind_cols] == [0.0, 1.0, 0.0]
    assert [by_id["m.Foo"][c] for c in kind_cols] == [1.0, 0.0, 0.0]
    # method routes to kind_function (methods collapse into function bucket).
    assert [by_id["m.Foo.bar"][c] for c in kind_cols] == [0.0, 0.0, 1.0]
    # skill_layer nodes have ALL kind one-hot cols zero (legal sentinel per STATE_DESIGN §3).
    assert [by_id["skill.demo.L1"][c] for c in kind_cols] == [0.0, 0.0, 0.0]


def test_layer_columns_one_hot_for_skill_layer() -> None:
    state = State.from_digraph(_mk_graph())
    by_id = {nid: state.X[i] for i, nid in enumerate(state.node_ids)}
    layer_cols = (COL_L1, COL_L2, COL_L3)
    assert [by_id["skill.demo.L1"][c] for c in layer_cols] == [1.0, 0.0, 0.0]
    assert [by_id["skill.demo.L2"][c] for c in layer_cols] == [0.0, 1.0, 0.0]
    # Code nodes carry layer=None → all layer one-hot cols are zero.
    assert [by_id["m"][c] for c in layer_cols] == [0.0, 0.0, 0.0]


def test_lazy_load_flag_correct() -> None:
    state = State.from_digraph(_mk_graph())
    by_id = {nid: state.X[i] for i, nid in enumerate(state.node_ids)}
    assert by_id["skill.demo.L2"][COL_LAZY] == 1.0
    assert by_id["skill.demo.L1"][COL_LAZY] == 0.0
    assert by_id["m"][COL_LAZY] == 0.0


def test_loc_normalized_via_log1p() -> None:
    """LOC column is log1p-normalized then divided by the per-graph max."""
    state = State.from_digraph(_mk_graph())
    by_id = {nid: state.X[i] for i, nid in enumerate(state.node_ids)}
    locs = {"m": 42, "m.Foo": 10, "m.Foo.bar": 5, "skill.demo.L1": 8, "skill.demo.L2": 6}
    log_locs = {nid: math.log1p(v) for nid, v in locs.items()}
    max_log = max(log_locs.values())
    for nid, lv in log_locs.items():
        assert float(by_id[nid][COL_LOC]) == pytest.approx(lv / max_log, rel=1e-5)


def test_degree_in_out_computed_correctly() -> None:
    """A 3-node triangle a→b→c→a produces degree_in == degree_out == 1 for each node."""
    g: nx.DiGraph = nx.DiGraph()
    for nid in ("a", "b", "c"):
        g.add_node(nid, kind="function", LOC=1, cyclomatic=1, layer=None, lazy_load_flag=False)
    g.add_edge("a", "b", rel_type="call", weight=1.0)
    g.add_edge("b", "c", rel_type="call", weight=1.0)
    g.add_edge("c", "a", rel_type="call", weight=1.0)
    state = State.from_digraph(g)
    # Each node has out-deg sum 1.0 and in-deg sum 1.0 → after /max_deg=1.0 they all map to 1.0.
    assert state.X.shape == (3, 16)
    assert torch.allclose(state.X[:, COL_DIN], torch.ones(3))
    assert torch.allclose(state.X[:, COL_DOUT], torch.ones(3))


def test_edge_type_counts_match() -> None:
    state = State.from_digraph(_mk_graph())
    # _mk_graph emits: 2 call edges, 1 inheritance edge, 1 import edge.
    assert state.edge_type_counts == {"call": 2, "inheritance": 1, "import": 1}


def test_to_pyg_data_emits_edge_index_and_x() -> None:
    state = State.from_digraph(_mk_graph())
    data = state.to_pyg_data()
    assert data["x"] is state.X
    assert data["num_nodes"] == state.num_nodes
    assert data["edge_index"].dtype == torch.long
    assert data["edge_index"].shape[0] == 2
    assert data["edge_index"].shape[1] == state.A.nnz


def test_betweenness_reserved_zero_and_self_loops_dropped() -> None:
    """Cols 4/15 start at 0 (STATE_DESIGN §3); self-loops dropped (§2 step 2)."""
    state = State.from_digraph(_mk_graph())
    assert torch.all(state.X[:, COL_BTW] == 0.0)
    assert torch.all(state.X[:, COL_RES] == 0.0)
    g: nx.DiGraph = nx.DiGraph()
    g.add_node("x", kind="function", LOC=1, cyclomatic=1, layer=None, lazy_load_flag=False)
    g.add_edge("x", "x", rel_type="call", weight=1.0)
    assert State.from_digraph(g).A.nnz == 0
    assert np.sum(State.from_digraph(g).A.toarray()) == 0.0
