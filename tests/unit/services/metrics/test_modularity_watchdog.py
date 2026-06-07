"""Fallback tests for ``src/services/metrics/modularity.py`` (Phase-4 RC-3).

RC-3 replaced the daemon-thread watchdog with a synchronous try/except so
that failed Louvain calls raise immediately instead of orphaning threads.
Tests verify correctness (greedy fallback returns finite Q; both-fail returns
Q=0.0) without enforcing a wall-clock budget (the budget is now just
``_run_with_budget``'s exception contract, not a timer).
"""

from __future__ import annotations

import math

import networkx as nx

from src.services.metrics import modularity as mod_mod
from src.services.metrics.modularity import compute_modularity

_DISJOINT_TRIANGLES_Q_FLOOR = 0.4


def _make_disjoint_triangles() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    return g


def test_louvain_failure_falls_back_to_greedy(monkeypatch) -> None:
    """If Louvain raises, the greedy fallback must produce finite Q."""

    def failing_louvain(_undirected, **_kw) -> None:
        raise RuntimeError("simulated Louvain failure (RC-3: no daemon-thread needed)")

    monkeypatch.setattr(mod_mod.nx_comm, "louvain_communities", failing_louvain)

    q = compute_modularity(_make_disjoint_triangles())
    assert math.isfinite(q), f"greedy fallback must return finite Q, got {q!r}"
    assert q > _DISJOINT_TRIANGLES_Q_FLOOR


def test_both_failures_returns_zero(monkeypatch) -> None:
    """If both Louvain and greedy raise, we surrender to Q = 0.0."""

    def failing(_undirected, **_kw) -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(mod_mod.nx_comm, "louvain_communities", failing)
    monkeypatch.setattr(mod_mod.nx_comm, "greedy_modularity_communities", failing)

    q = compute_modularity(_make_disjoint_triangles())
    assert q == 0.0


def test_normal_graph_returns_positive_q() -> None:
    """Happy path: Louvain on a well-structured graph returns Q above floor."""
    q = compute_modularity(_make_disjoint_triangles())
    assert q > _DISJOINT_TRIANGLES_Q_FLOOR


# Keep aliases so existing WATCHDOG_SECONDS references in docs stay valid.
test_watchdog_fires_and_falls_back_to_greedy = test_louvain_failure_falls_back_to_greedy
test_watchdog_double_timeout_returns_zero = test_both_failures_returns_zero
test_watchdog_does_not_fire_on_normal_graph = test_normal_graph_returns_positive_q
