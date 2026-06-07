"""RC-1 watchdog + fallback tests for ``src/services/metrics/modularity.py``.

Split from ``test_modularity.py`` to stay under the 150-LOC per-file cap
(CLAUDE.md §1). These tests prove the wall-clock budget enforced by
``_run_with_budget`` actually cancels a slow Louvain and either steps to
the greedy fallback or surrenders to ``Q = 0.0``.
"""

from __future__ import annotations

import math
import time

import networkx as nx

from src.services.metrics import modularity as mod_mod
from src.services.metrics.modularity import compute_modularity

_DISJOINT_TRIANGLES_Q_FLOOR = 0.4


def _make_disjoint_triangles() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    return g


def test_watchdog_fires_and_falls_back_to_greedy(monkeypatch) -> None:
    """If Louvain blows the budget, the greedy fallback must produce finite Q."""

    def slow_louvain(_undirected, **_kw):
        time.sleep(mod_mod.WATCHDOG_SECONDS + 1.0)
        raise AssertionError("watchdog should have cancelled this")

    monkeypatch.setattr(mod_mod.nx_comm, "louvain_communities", slow_louvain)

    t0 = time.perf_counter()
    q = compute_modularity(_make_disjoint_triangles())
    elapsed = time.perf_counter() - t0

    assert math.isfinite(q), f"fallback must return finite Q, got {q!r}"
    # Two-budget envelope (Louvain budget + greedy budget) + ~0.5s slack.
    assert elapsed < 2 * mod_mod.WATCHDOG_SECONDS + 0.5, f"watchdog should bound runtime; got {elapsed:.2f}s"


def test_watchdog_double_timeout_returns_zero(monkeypatch) -> None:
    """If *both* Louvain and greedy blow budget, we surrender to Q = 0.0."""

    def slow(_undirected, **_kw):
        time.sleep(mod_mod.WATCHDOG_SECONDS + 1.0)
        raise AssertionError("watchdog should have cancelled this")

    monkeypatch.setattr(mod_mod.nx_comm, "louvain_communities", slow)
    monkeypatch.setattr(mod_mod.nx_comm, "greedy_modularity_communities", slow)

    q = compute_modularity(_make_disjoint_triangles())
    assert q == 0.0


def test_watchdog_does_not_fire_on_normal_graph() -> None:
    """Happy path: Louvain finishes well inside the budget, Q matches contract."""
    t0 = time.perf_counter()
    q = compute_modularity(_make_disjoint_triangles())
    elapsed = time.perf_counter() - t0
    assert q > _DISJOINT_TRIANGLES_Q_FLOOR
    assert elapsed < mod_mod.WATCHDOG_SECONDS
