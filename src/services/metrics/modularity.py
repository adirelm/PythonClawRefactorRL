"""Newman-Girvan modularity (Q) for the skills-graph MDP.

Wires alpha*dModularity in the canonical reward (Phase 2). Phase 4 RC-1/RC-2:
some mid-rollout topologies wedge Louvain for >>1s on seeds 123/314; we wrap
Louvain in a ``WATCHDOG_SECONDS`` cap, fall back to greedy CNM, then to Q=0.
``safe_louvain`` memoizes by structural graph key so reward.compute_reward
costs 1 Louvain per snapshot (2 per env.step) instead of 6 (3 metrics x 2).
"""

from __future__ import annotations

import logging

import networkx as nx
import networkx.algorithms.community as nx_comm

_LOUVAIN_SEED = 42
WATCHDOG_SECONDS = 0.05  # Aggressive cap; OK seeds run Louvain in microseconds.
_MIN_NODES_FOR_MODULARITY = 2  # Q undefined for V<2 (RC-2 topology guard)
_PARTITION_CACHE_MAX = 64

_log = logging.getLogger(__name__)


def _louvain_partition(undirected: nx.Graph) -> list[set]:
    return nx_comm.louvain_communities(undirected, seed=_LOUVAIN_SEED)


def _greedy_partition(undirected: nx.Graph) -> list[set]:
    return list(nx_comm.greedy_modularity_communities(undirected))


def _run_with_budget(fn, undirected: nx.Graph) -> list[set] | None:
    """Run ``fn(undirected)`` under ``WATCHDOG_SECONDS``. Daemon thread; on
    timeout return None and let the worker finish in background (can't kill
    a Python thread cooperatively). ThreadPoolExecutor was rejected because
    __exit__ blocks on pending workers, defeating the wall-clock guarantee.
    """
    import threading  # noqa: PLC0415 — local to keep import budget low

    result: list[list[set] | None] = [None]
    done = threading.Event()

    def runner() -> None:
        try:
            result[0] = fn(undirected)
        except Exception:
            result[0] = None
        finally:
            done.set()

    threading.Thread(target=runner, daemon=True, name="modularity-watchdog").start()
    if done.wait(timeout=WATCHDOG_SECONDS):
        return result[0]
    return None


_partition_cache: dict[tuple[frozenset, frozenset], list[set] | None] = {}


def _graph_key(undirected: nx.Graph) -> tuple[frozenset, frozenset]:
    """Cache key by (nodes-frozenset, edges-frozenset). Both required: graphs
    with the same edges but different isolated-node sets must NOT collide
    (a cached partition includes node labels and is only valid for the exact
    graph it was computed on)."""
    return (frozenset(undirected.nodes()), frozenset(undirected.edges()))


def safe_louvain(undirected: nx.Graph) -> list[set] | None:
    """Run Louvain → greedy fallback → None, each under ``WATCHDOG_SECONDS``.

    Shared by ``modularity`` / ``cohesion`` / ``coupling`` so the 6-Louvain-calls-
    per-step worst case collapses to 2 (one per graph snapshot) when callers
    precompute the partition via ``reward.compute_reward`` and pass through.

    Memoizes by structural graph key so NOOPs and failed refactor ops (graph
    unchanged step-to-step) only pay once per unique topology in a rollout.
    Cache evicts oldest when ``_PARTITION_CACHE_MAX`` is exceeded.
    """
    key = _graph_key(undirected)
    if key in _partition_cache:
        return _partition_cache[key]
    partition = _run_with_budget(_louvain_partition, undirected)
    if partition is None:
        partition = _run_with_budget(_greedy_partition, undirected)
    if len(_partition_cache) >= _PARTITION_CACHE_MAX:
        _partition_cache.pop(next(iter(_partition_cache)))
    _partition_cache[key] = partition
    return partition


def clear_partition_cache() -> None:
    """Reset the partition memo (called by ``env.reset`` to keep tests deterministic)."""
    _partition_cache.clear()


def compute_modularity(graph: nx.DiGraph, *, _partition: list[set] | None = None) -> float:
    """Return the Newman-Girvan modularity ``Q`` of ``graph``.

    The skills-graph is a ``nx.DiGraph``; modularity is defined on
    undirected graphs, so we project via ``to_undirected()`` before
    running Louvain. Empty / single-node graphs return ``0.0`` because
    Q is undefined when there are no edges to partition.

    RC-1 watchdog: Louvain is bounded at ``WATCHDOG_SECONDS``. On timeout
    we try ``greedy_modularity_communities`` (faster, deterministic,
    Clauset-Newman-Moore). If *that* also exceeds budget we surrender
    and return ``0.0`` (trivial partition) so the env.step caller never
    blocks. Each fallback hop is logged at WARN.

    Args:
        graph: Directed skills-graph snapshot (e.g. ``env._graph``).

    Returns:
        Modularity score as a ``float``. Higher = stronger community
        structure. ``0.0`` for the trivial / empty / wedged case.
    """
    # RC-2: topology early-return — Q undefined on V<2 or E=0.
    if graph.number_of_nodes() < _MIN_NODES_FOR_MODULARITY or graph.number_of_edges() == 0:
        return 0.0
    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        return 0.0

    communities = _partition if _partition is not None else safe_louvain(undirected)
    if communities is None:
        _log.warning(
            "modularity: Louvain+greedy both exceeded %.2fs (V=%d E=%d); Q=0.0",
            WATCHDOG_SECONDS,
            undirected.number_of_nodes(),
            undirected.number_of_edges(),
        )
        return 0.0

    return float(nx_comm.modularity(undirected, communities))


def delta_modularity(before: nx.DiGraph, after: nx.DiGraph) -> float:
    """Signed change in modularity across one refactor step.

    ``dQ = Q(after) - Q(before)``. Used by ``src.env.reward.compute_reward``
    as the ``alpha*dModularity`` term of the canonical reward equation.

    Args:
        before: Graph snapshot before the action.
        after: Graph snapshot after the action.

    Returns:
        Float difference; positive means modularity improved.
    """
    return compute_modularity(after) - compute_modularity(before)
