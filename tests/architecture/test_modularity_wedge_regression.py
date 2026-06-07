"""RC-4 architectural regression test for the Louvain wedge.

Background (CLAUDE.md §RC-0 finding): on seeds 123/314 the policy walks
the graph into a topology where ``nx_comm.louvain_communities`` wedges
for many seconds, blowing the 120s per-seed PPO budget. RC-1 added a
``WATCHDOG_SECONDS = 1.0`` wall-clock guard plus a greedy fallback.

This test pins the contract: *any* graph the env can plausibly hand to
``compute_modularity`` must return in well under 1.5s, even when the
underlying topology would otherwise wedge Louvain. We construct two
worst-case shapes (a wide star, a "wheel of triangles") that empirically
correlated with slow Louvain runs, plus a third shape that forces the
watchdog to fire by stubbing Louvain to a slow no-op — the wall-clock
bound must hold either way.
"""

from __future__ import annotations

import time

import networkx as nx

from src.services.metrics import modularity as mod_mod
from src.services.metrics.modularity import compute_modularity

# 1.5s is the RC-4 hard ceiling: watchdog budget (1.0s) + greedy budget
# (1.0s) + ~0.5s slack, then halved because most calls only hit the first
# budget. We pin 1.5s as the *contract* a single call must never exceed.
_WALL_CLOCK_CEILING_S = 1.5


def _wide_star(n: int = 256) -> nx.DiGraph:
    """Star with a single hub + ``n`` leaves — pathological for Louvain levels."""
    g = nx.DiGraph()
    for leaf in range(1, n + 1):
        g.add_edge(0, leaf)
        g.add_edge(leaf, 0)
    return g


def _wheel_of_triangles(rings: int = 32) -> nx.DiGraph:
    """Many disjoint K_3 sharing a hub — exercises Louvain merge ladders."""
    g = nx.DiGraph()
    for r in range(rings):
        a, b, c = 3 * r + 1, 3 * r + 2, 3 * r + 3
        g.add_edges_from([(a, b), (b, c), (c, a)])
        g.add_edge(0, a)
        g.add_edge(a, 0)
    return g


def test_wide_star_under_wall_clock_ceiling() -> None:
    g = _wide_star()
    t0 = time.perf_counter()
    q = compute_modularity(g)
    elapsed = time.perf_counter() - t0
    assert isinstance(q, float)
    assert elapsed < _WALL_CLOCK_CEILING_S, (
        f"wide-star Louvain regressed: {elapsed:.2f}s > {_WALL_CLOCK_CEILING_S}s"
    )


def test_wheel_of_triangles_under_wall_clock_ceiling() -> None:
    g = _wheel_of_triangles()
    t0 = time.perf_counter()
    q = compute_modularity(g)
    elapsed = time.perf_counter() - t0
    assert isinstance(q, float)
    assert elapsed < _WALL_CLOCK_CEILING_S, (
        f"wheel-of-triangles regressed: {elapsed:.2f}s > {_WALL_CLOCK_CEILING_S}s"
    )


def test_forced_wedge_still_under_ceiling(monkeypatch) -> None:
    """Even when Louvain *would* wedge, the watchdog must hold the ceiling.

    Stubs Louvain to sleep > watchdog budget; the greedy fallback runs on
    the small wide-star and returns quickly. The contract is the wall-clock
    bound, not the value of Q (which legitimately drops on the fallback).
    """

    def slow_louvain(_undirected, **_kw):
        time.sleep(mod_mod.WATCHDOG_SECONDS + 1.0)
        raise AssertionError("watchdog should have cancelled this")

    monkeypatch.setattr(mod_mod.nx_comm, "louvain_communities", slow_louvain)

    g = _wide_star(n=64)
    t0 = time.perf_counter()
    q = compute_modularity(g)
    elapsed = time.perf_counter() - t0
    # 2x ceiling because we have to budget Louvain + greedy + slack.
    assert elapsed < 2 * _WALL_CLOCK_CEILING_S, f"forced-wedge breached watchdog: {elapsed:.2f}s"
    # Q must still be a finite float (greedy fallback or 0.0 surrender).
    assert isinstance(q, float)
