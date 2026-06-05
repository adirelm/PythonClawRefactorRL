"""Centrality service — Degree per step (cheap), Betweenness exactly twice/seed.

Brief §2.2 + ADR-006 + CLAUDE.md §CANONICAL VALUES:

    * Degree Centrality may be computed every MDP step (it is O(|E|)).
    * Betweenness Centrality MUST be computed exactly **twice per seed**:
      once at training start (initial graph) and once at training end
      (final graph). Any extra call leaks compute budget; any missing
      call breaks the Δ-Betweenness comparison in ANALYSIS.md.

This module owns the discipline. ``CentralityScheduler`` keeps a per-seed
counter and surfaces the module-level ``compute_betweenness`` symbol so
``tests/architecture/test_betweenness_call_count.py`` can ``patch`` it
and count invocations.

A scheduler instance carries ``_betweenness_calls`` (int) — the in-process
call counter for the seed. Tests assert it equals 1 after env ``__init__``
(start) and 2 after ``env.final_betweenness()`` (end). The cap is enforced
by ``RuntimeError`` if a third call is attempted within the same seed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)

_MAX_CALLS_PER_SEED = 2  # CLAUDE.md §CANONICAL: start + end ONLY


def compute_betweenness(graph: nx.DiGraph) -> dict[str, float]:
    """Return per-node betweenness centrality.

    Thin wrapper around ``networkx.betweenness_centrality`` so the
    architectural contract test can ``patch`` exactly one symbol and
    spy on call counts (see ``tests/architecture/test_betweenness_call_count.py``).

    Args:
        graph: Live ``nx.DiGraph`` snapshot.

    Returns:
        Dict mapping node id to its (already normalized) betweenness in [0, 1].
    """
    if graph.number_of_nodes() == 0:
        return {}
    # networkx normalizes by default (k=None, normalized=True) — keeps values in
    # [0, 1] which is the contract STATE_DESIGN §3 col-4 expects.
    return dict(nx.betweenness_centrality(graph, normalized=True))


def compute_degree(graph: nx.DiGraph) -> dict[str, int]:
    """Return per-node total degree (in + out for DiGraphs).

    Cheap (O(|E|)); safe to call every MDP step per ADR-006.
    """
    return {nid: int(graph.in_degree(nid) + graph.out_degree(nid)) for nid in graph.nodes()}


@dataclass
class CentralityScheduler:
    """Per-seed centrality scheduler with the 2-betweenness-calls budget.

    Attributes:
        seed: Seed value this scheduler is bound to (informational; used by
            log lines so cross-seed leakage is easy to spot in CI output).
        _betweenness_calls: In-process counter, **must** end the seed at 2.
        _last_betweenness: Last betweenness dict (cached for re-use without
            recomputation; cheaper than re-running the O(VE) algorithm).
    """

    seed: int
    _betweenness_calls: int = 0
    _last_betweenness: dict[str, float] = field(default_factory=dict)

    def compute_betweenness(self, graph: nx.DiGraph) -> dict[str, float]:
        """Compute betweenness; raise if the per-seed budget is exhausted.

        Re-uses the module-level ``compute_betweenness`` symbol so the
        spy in ``tests/architecture/test_betweenness_call_count.py`` sees
        every call. Each invocation bumps ``_betweenness_calls`` and the
        result is cached as ``_last_betweenness``.
        """
        if self._betweenness_calls >= _MAX_CALLS_PER_SEED:
            raise RuntimeError(
                f"CentralityScheduler(seed={self.seed}) exceeded the canonical "
                f"budget of {_MAX_CALLS_PER_SEED} betweenness calls per seed "
                f"(brief §2.2 + ADR-006 + CLAUDE.md). Already at "
                f"{self._betweenness_calls}; refuse 3rd call."
            )
        result = compute_betweenness(graph)
        self._betweenness_calls += 1
        self._last_betweenness = result
        logger.debug(
            "CentralityScheduler(seed=%d): betweenness call %d/%d (|V|=%d)",
            self.seed,
            self._betweenness_calls,
            _MAX_CALLS_PER_SEED,
            graph.number_of_nodes(),
        )
        return result

    def compute_degree(self, graph: nx.DiGraph) -> dict[str, int]:
        """Per-step degree centrality (cheap; unmetered)."""
        return compute_degree(graph)

    @property
    def betweenness_calls(self) -> int:
        """Public read-only view of the in-process call counter."""
        return self._betweenness_calls
