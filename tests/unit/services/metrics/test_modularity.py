"""Tests for src/services/metrics/modularity.py — Newman-Girvan Q.

Coverage target: ≥85% on this module. Each test asserts a property
of Q that the canonical reward equation
(R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling_Penalty + P_skills)
relies on — so a regression here is felt directly by the PPO trainer.
"""

from __future__ import annotations

import networkx as nx

from src.services.metrics.modularity import compute_modularity, delta_modularity

# Two disjoint K_3 should yield Q ≳ 0.5 with Louvain (clear community split).
_DISJOINT_TRIANGLES_Q_FLOOR = 0.4


def _make_disjoint_triangles() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    return g


def _make_k5() -> nx.DiGraph:
    return nx.complete_graph(5).to_directed()


def test_modularity_returns_float() -> None:
    """Even on a trivial graph, the return type must be a plain float."""
    q = compute_modularity(_make_disjoint_triangles())
    assert isinstance(q, float)


def test_empty_graph_modularity_is_zero() -> None:
    """No nodes ⇒ no partition ⇒ Q is defined as 0.0 by our contract."""
    assert compute_modularity(nx.DiGraph()) == 0.0


def test_single_node_modularity_is_zero() -> None:
    """Single isolated node has no edges, hence Q = 0.0."""
    g = nx.DiGraph()
    g.add_node("only")
    assert compute_modularity(g) == 0.0


def test_disjoint_triangles_have_positive_modularity() -> None:
    """Two disjoint K_3 ⇒ Louvain finds the split ⇒ Q clearly > 0.4."""
    q = compute_modularity(_make_disjoint_triangles())
    assert q > _DISJOINT_TRIANGLES_Q_FLOOR


def test_complete_graph_modularity_low() -> None:
    """K_5 is one community ⇒ Q ≈ 0 (no meaningful sub-structure)."""
    q = compute_modularity(_make_k5())
    assert abs(q) < 1e-6


def test_delta_modularity_returns_difference() -> None:
    """ΔQ = Q(after) - Q(before), as a plain float."""
    before = _make_k5()
    after = _make_disjoint_triangles()
    delta = delta_modularity(before, after)
    assert isinstance(delta, float)
    expected = compute_modularity(after) - compute_modularity(before)
    assert abs(delta - expected) < 1e-9


def test_delta_positive_when_split_creates_communities() -> None:
    """Going K_5 → disjoint K_3 ∪ K_3 must yield ΔQ > 0 (refactor helped)."""
    before = _make_k5()
    after = _make_disjoint_triangles()
    assert delta_modularity(before, after) > 0.0


def test_delta_zero_for_identical_snapshots() -> None:
    """Idempotent step (e.g. NOOP) ⇒ ΔQ exactly 0.0."""
    g = _make_disjoint_triangles()
    assert delta_modularity(g, g) == 0.0
