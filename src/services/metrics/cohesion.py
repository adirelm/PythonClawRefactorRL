"""Cohesion metric — size-weighted mean within-community clustering (brief §2.2).

The canonical reward equation
``R_t = a*dModularity + b*dCohesion - g*Coupling_Penalty + P_skills``
expects ``compute_cohesion`` to return a scalar in ``[0, 1]`` that grows
when nodes inside the same Louvain community become more tightly
interconnected — i.e. when each community becomes more "clique-like".

The implementation:

1. Project the directed PythonClaw graph onto its undirected counterpart
   (Louvain + clustering coefficient are defined for undirected graphs).
2. Partition the undirected graph with ``louvain_communities(seed=42)``
   — the same seed used by ``modularity.py`` and ``coupling.py`` so the
   three metrics see *one* community structure per snapshot.
3. For every community ``c`` with ``|c| ≥ 2``, compute
   ``nx.average_clustering(g.subgraph(c))``. Single-node communities
   contribute ``0`` because the clustering coefficient is undefined for
   isolated nodes; including them as ``0`` keeps the metric well-defined.
4. Aggregate as a **size-weighted** mean
   ``sum(avg_clust(c) · |c|) / sum(|c|)`` so a tiny perfectly-clique-y
   community can't dominate a large messy one. The denominator equals
   ``|V|`` whenever every node lands in exactly one community (Louvain
   guarantees this), so the formula reduces to a per-node average of
   each node's community-local clustering signal.

Edge cases (``|V| < 2`` or ``|E| == 0``) short-circuit to ``0.0`` so the
metric is total over every DiGraph the env hands us (RC-2 early-return,
matches the contract in ``modularity.py``).

Phase 4 RC-1: the Louvain call here shares the same wedge surface as
``modularity.compute_modularity`` — on mid-rollout snapshots from
seeds 123/314 it can block for many seconds. We delegate to the
shared ``_run_with_budget`` watchdog (re-exported from ``modularity``)
and on timeout fall back to a single-community partition (cohesion = 0).
"""

from __future__ import annotations

import logging

import networkx as nx
import networkx.algorithms.community as nx_comm

from src.services.metrics.modularity import _run_with_budget

_LOUVAIN_SEED = 42  # sealed in CLAUDE.md §CANONICAL VALUES for reproducibility
_MIN_NODES_FOR_CLUSTERING = 2  # clustering coefficient undefined for |V| < 2
_MIN_COMMUNITY_SIZE = 2  # single-node communities contribute 0 (no neighbours)

_log = logging.getLogger(__name__)


def _louvain_partition(undirected: nx.Graph) -> list[set]:
    return nx_comm.louvain_communities(undirected, seed=_LOUVAIN_SEED)


def compute_cohesion(graph: nx.DiGraph) -> float:
    """Return size-weighted mean within-community clustering in ``[0, 1]``.

    Args:
        graph: Snapshot of the live PythonClaw Skills DiGraph.

    Returns:
        Float in ``[0.0, 1.0]``. ``0.0`` for empty / edgeless / singleton
        graphs **or** when every Louvain community is a tree (e.g. a star).
        ``1.0`` when every Louvain community is a clique (e.g. disjoint K_n).
    """
    if graph.number_of_nodes() < _MIN_NODES_FOR_CLUSTERING or graph.number_of_edges() == 0:
        return 0.0

    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        # Defensive: a DiGraph whose edges all collapse on to_undirected()
        # (e.g. self-loops only) yields no usable edge set for clustering.
        return 0.0

    # RC-1: Louvain watchdog — on wedge, fall back to a single community,
    # which yields cohesion = 0.0 below (no within-community structure to score).
    communities = _run_with_budget(_louvain_partition, undirected)
    if communities is None:
        _log.warning(
            "cohesion: Louvain exceeded watchdog budget (V=%d E=%d); returning 0.0",
            undirected.number_of_nodes(),
            undirected.number_of_edges(),
        )
        return 0.0

    total_weighted = 0.0
    total_size = 0
    for community in communities:
        size = len(community)
        if size < _MIN_COMMUNITY_SIZE:
            # Single-node community: clustering coefficient is undefined,
            # contribute 0 but still count the node toward the denominator
            # so single-node communities suppress the overall score rather
            # than silently inflating it.
            total_size += size
            continue
        subgraph = undirected.subgraph(community)
        avg_clustering = nx.average_clustering(subgraph)
        total_weighted += avg_clustering * size
        total_size += size

    return total_weighted / total_size if total_size > 0 else 0.0


def delta_cohesion(before: nx.DiGraph, after: nx.DiGraph) -> float:
    """Return ``compute_cohesion(after) - compute_cohesion(before)``.

    A **positive** delta means within-community connectivity tightened
    (refactor improved local clustering). The reward formula adds
    ``beta * delta_cohesion`` to R_t, so positive deltas are rewarded —
    encouraging the agent to keep tightly-coupled functions inside the
    same module rather than scattering them.
    """
    return compute_cohesion(after) - compute_cohesion(before)
