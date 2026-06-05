"""Unit tests for src/services/metrics/coupling.py.

Coupling penalty := fraction of edges that cross Louvain community
boundaries on the undirected projection of the graph. Louvain is seeded
with seed=42 (sealed in CLAUDE.md §CANONICAL VALUES) so partitions —
and therefore coupling values — are reproducible across runs.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.services.metrics.coupling import (
    compute_coupling,
    compute_coupling_penalty,
    delta_coupling,
)


def test_coupling_returns_float_in_0_to_1() -> None:
    """Return type is float and value sits inside the closed interval [0, 1]."""
    graph: nx.DiGraph = nx.DiGraph()
    # A small directed motif with at least one edge so we hit the non-empty branch.
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")])

    value = compute_coupling_penalty(graph)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_empty_graph_coupling_is_zero() -> None:
    """No edges → coupling is defined as 0.0 (no cross-community edges exist)."""
    empty_directed: nx.DiGraph = nx.DiGraph()
    assert compute_coupling_penalty(empty_directed) == 0.0

    # Nodes but no edges → still 0.0 (early return guards against /0).
    nodes_only: nx.DiGraph = nx.DiGraph()
    nodes_only.add_nodes_from(range(5))
    assert compute_coupling_penalty(nodes_only) == 0.0


def test_complete_graph_low_coupling() -> None:
    """K_5: Louvain collapses to a single community → coupling ≈ 0.0."""
    # nx.complete_graph yields an undirected K_5; the metric to_undirected()'s
    # input so we wrap it in a DiGraph to honour the type contract.
    k5_undirected = nx.complete_graph(5)
    k5: nx.DiGraph = nx.DiGraph(k5_undirected)

    coupling = compute_coupling_penalty(k5)

    # On K_5 every node sits in one Louvain community → zero cross edges.
    assert coupling == 0.0


def test_two_disjoint_components_zero_coupling() -> None:
    """Two disconnected triangles → Louvain finds them as separate communities,
    and *no* edge crosses between them, so coupling is exactly 0.0.
    """
    graph: nx.DiGraph = nx.DiGraph()
    # Component A: triangle on {0, 1, 2}
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Component B: triangle on {10, 11, 12}
    graph.add_edges_from([(10, 11), (11, 12), (12, 10)])

    assert compute_coupling_penalty(graph) == 0.0


def test_bridge_between_clusters_high_coupling() -> None:
    """Two K_3 triangles linked by a single bridge edge.

    Undirected edge set: 3 (triangle A) + 3 (triangle B) + 1 (bridge) = 7.
    Louvain (seed=42) splits the two triangles into separate communities,
    so exactly the bridge edge crosses → coupling = 1/7 ≈ 0.142857.
    """
    graph: nx.DiGraph = nx.DiGraph()
    # Triangle A on {0, 1, 2}
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Triangle B on {3, 4, 5}
    graph.add_edges_from([(3, 4), (4, 5), (5, 3)])
    # Single bridge edge between the two clusters
    graph.add_edge(2, 3)

    coupling = compute_coupling_penalty(graph)

    assert coupling == pytest.approx(1.0 / 7.0)
    assert coupling > 0.0  # ie. genuinely non-zero / "high" relative to the disjoint case


def test_delta_coupling_returns_difference() -> None:
    """delta_coupling = coupling(after) − coupling(before).

    Uses the bridge-vs-no-bridge construction so the delta is exactly the
    bridge edge's contribution: 1/7 − 0 = 1/7. Sign convention: positive
    delta means coupling **increased**; the reward formula subtracts
    γ·delta so this hurts R_t (expected — adding a cross-community edge
    is a regression).
    """
    before: nx.DiGraph = nx.DiGraph()
    before.add_edges_from([(0, 1), (1, 2), (2, 0)])
    before.add_edges_from([(3, 4), (4, 5), (5, 3)])

    after: nx.DiGraph = nx.DiGraph()
    after.add_edges_from([(0, 1), (1, 2), (2, 0)])
    after.add_edges_from([(3, 4), (4, 5), (5, 3)])
    after.add_edge(2, 3)  # the regression: now the clusters are bridged

    assert compute_coupling_penalty(before) == 0.0
    assert compute_coupling_penalty(after) == pytest.approx(1.0 / 7.0)
    assert delta_coupling(before, after) == pytest.approx(1.0 / 7.0)

    # And the reverse direction (refactor that removes the bridge) is negative,
    # i.e. a coupling *improvement* — this is what the agent should learn.
    assert delta_coupling(after, before) == pytest.approx(-1.0 / 7.0)


def test_compute_coupling_alias_matches_canonical() -> None:
    """``compute_coupling`` is the reward-module-facing alias of the canonical
    ``compute_coupling_penalty``. Same object → both names always agree.
    """
    assert compute_coupling is compute_coupling_penalty

    graph: nx.DiGraph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    assert compute_coupling(graph) == compute_coupling_penalty(graph)
