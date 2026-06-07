"""Coupling penalty metric - fraction of cross-community edges (brief sec 2.2).

The reward equation R_t = alpha*dModularity + beta*dCohesion - gamma*Coupling
penalises *increases* in coupling, so:

* ``compute_coupling_penalty(g)`` returns a value in ``[0, 1]`` - the
  share of edges that cross Louvain community boundaries on the
  undirected projection of ``g``.
* ``delta_coupling(before, after)`` returns ``coupling(after) - coupling(before)``.
  A **negative** delta means coupling *decreased* (improvement); the
  reward formula subtracts ``gamma * delta`` so a negative delta yields a
  positive contribution to R_t - refactors that disconnect communities
  are rewarded.

Louvain is seeded with ``seed=42`` so the partition (and hence the
coupling value) is reproducible across runs and across the paired-seed
ablations in ANALYSIS.md.

``compute_coupling`` is exposed as an alias of ``compute_coupling_penalty``
because ``src/env/reward.py`` late-imports the name ``compute_coupling``
from ``src.services.metrics``; keeping both names in this file means the
reward wiring works the moment ``src/services/metrics/__init__.py``
re-exports either symbol.
"""

from __future__ import annotations

import logging

import networkx as nx

from src.services.metrics.modularity import safe_louvain

_LOUVAIN_SEED = 42  # sealed in CLAUDE.md §CANONICAL VALUES for reproducibility

_log = logging.getLogger(__name__)


def compute_coupling_penalty(graph: nx.DiGraph, *, _partition: list[set] | None = None) -> float:
    """Return the fraction of edges that cross Louvain communities.

    The directed graph is projected to its undirected counterpart before
    Louvain runs (Louvain is defined for undirected graphs). An empty
    edge set returns ``0.0`` - there are no cross-community edges by
    definition when there are no edges at all.

    Phase 4 RC-1: the Louvain call here shares the same wedge surface as
    ``modularity.compute_modularity`` — on mid-rollout snapshots from
    seeds 123/314 it can block for many seconds. We delegate to the
    shared ``_run_with_budget`` watchdog. On Louvain timeout we try
    ``greedy_modularity_communities``; on second timeout we surrender
    and return ``0.0`` so the env.step caller never blocks.

    Args:
        graph: Snapshot of the live PythonClaw Skills graph.

    Returns:
        Float in ``[0.0, 1.0]``. ``0.0`` means every edge is intra-community
        (perfectly modular); ``1.0`` means every edge crosses a community
        boundary (maximally coupled).
    """
    if graph.number_of_edges() == 0:
        return 0.0

    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        # Defensive: a DiGraph with reciprocal-only edges collapses to
        # an undirected graph with the same edge set, but if upstream code
        # ever hands us a graph where to_undirected() drops every edge
        # (e.g. all self-loops on a MultiDiGraph), treat it as no-edges.
        return 0.0

    communities = _partition if _partition is not None else safe_louvain(undirected)
    if communities is None:
        _log.warning(
            "coupling: Louvain+greedy both exceeded watchdog (V=%d E=%d); coupling=0.0",
            undirected.number_of_nodes(),
            undirected.number_of_edges(),
        )
        return 0.0

    node_to_comm: dict[object, int] = {
        node: idx for idx, community in enumerate(communities) for node in community
    }

    cross = sum(1 for u, v in undirected.edges() if node_to_comm[u] != node_to_comm[v])
    return cross / undirected.number_of_edges()


def delta_coupling(before: nx.DiGraph, after: nx.DiGraph) -> float:
    """Return ``coupling(after) - coupling(before)``.

    NOTE on sign convention: a *negative* return value means coupling has
    **decreased** (improvement). The canonical reward equation subtracts
    ``gamma * delta_coupling`` from R_t, so a negative delta contributes a
    positive term to the reward - i.e. refactors that lower cross-community
    coupling are rewarded.
    """
    return compute_coupling_penalty(after) - compute_coupling_penalty(before)


# Reward-wiring alias: src/env/reward.py late-imports the name ``compute_coupling``
# from ``src.services.metrics``. Re-exporting under both names lets the metrics
# package __init__ surface either symbol without changing the reward module.
compute_coupling = compute_coupling_penalty
