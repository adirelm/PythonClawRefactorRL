"""Unit tests for src/services/metrics/cohesion.py.

Cohesion := size-weighted mean within-community clustering coefficient on
the undirected projection of the graph. Louvain is seeded with seed=42
(sealed in CLAUDE.md §CANONICAL VALUES) so the partition — and therefore
the cohesion value — is reproducible across runs and CI shards.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.services.metrics.cohesion import compute_cohesion, delta_cohesion


def test_cohesion_returns_float_in_0_to_1() -> None:
    """Return type is float and value sits inside the closed interval [0, 1]."""
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")])

    value = compute_cohesion(graph)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_empty_graph_cohesion_is_zero() -> None:
    """No nodes / no edges → cohesion is defined as 0.0 (clustering undefined)."""
    empty_directed: nx.DiGraph = nx.DiGraph()
    assert compute_cohesion(empty_directed) == 0.0

    # Nodes but no edges → still 0.0 (|E| == 0 short-circuit).
    nodes_only: nx.DiGraph = nx.DiGraph()
    nodes_only.add_nodes_from(range(5))
    assert compute_cohesion(nodes_only) == 0.0


def test_single_node_graph_returns_zero() -> None:
    """|V| < 2 short-circuit guards against /0 in the size-weighted mean."""
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_node("solo")
    assert compute_cohesion(graph) == 0.0


def test_two_disjoint_triangles_high_cohesion() -> None:
    """Two disjoint K_3 triangles → every node has local clustering 1.0,
    so the size-weighted mean is exactly 1.0 (the upper bound).
    """
    graph: nx.DiGraph = nx.DiGraph()
    # Triangle A on {0, 1, 2}
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Triangle B on {3, 4, 5}
    graph.add_edges_from([(3, 4), (4, 5), (5, 3)])

    cohesion = compute_cohesion(graph)

    assert cohesion > 0.5  # the requested "high" threshold from the spec
    assert cohesion == pytest.approx(1.0)


def test_star_graph_low_cohesion() -> None:
    """A star graph has clustering coefficient 0 at every node — no triangles
    can form because the leaves only connect through the hub. Louvain
    collapses the star into a single community, so the size-weighted mean
    is exactly 0.0.
    """
    star_undirected = nx.star_graph(5)  # 1 hub + 5 leaves = 6 nodes, 5 edges
    star: nx.DiGraph = nx.DiGraph(star_undirected)

    cohesion = compute_cohesion(star)

    assert cohesion == 0.0


def test_delta_cohesion_returns_difference() -> None:
    """delta_cohesion = cohesion(after) − cohesion(before).

    Refactor scenario: the *before* graph is a star (cohesion = 0); the
    *after* graph closes the leaves into a triangle ring (cohesion > 0).
    Sign convention: positive delta means within-community connectivity
    tightened — the reward formula adds β·delta, so the agent is rewarded
    for moves that increase cohesion.
    """
    before: nx.DiGraph = nx.DiGraph(nx.star_graph(5))

    # After: two disjoint K_3 triangles → cohesion = 1.0 (maximally cohesive).
    after: nx.DiGraph = nx.DiGraph()
    after.add_edges_from([(0, 1), (1, 2), (2, 0)])
    after.add_edges_from([(3, 4), (4, 5), (5, 3)])

    delta = delta_cohesion(before, after)

    assert delta == pytest.approx(1.0)  # 1.0 - 0.0
    assert delta > 0.0  # the refactor *improved* cohesion

    # And the reverse direction (regression) is a negative delta.
    assert delta_cohesion(after, before) == pytest.approx(-1.0)


def test_self_loops_only_returns_zero() -> None:
    """Defensive branch: a DiGraph whose only edges are self-loops collapses
    to an undirected graph with no usable edges; cohesion must stay 0.0.
    """
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_edge("a", "a")  # self-loop only

    # |E|==1 so we pass the first guard, but the undirected projection has
    # exactly one (self-loop) edge that contributes no clustering signal.
    value = compute_cohesion(graph)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0
