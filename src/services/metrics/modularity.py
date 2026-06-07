"""Newman-Girvan modularity (Q) for the skills-graph MDP.

RC-4: uses SIGALRM (POSIX; no-op on Windows) for a 1-second hard cut that
raises in the *calling* thread — no daemon threads, no GIL accumulation.
``safe_louvain`` memoizes by structural graph key so reward.compute_reward
costs 1 Louvain per snapshot (2 per env.step) instead of 6 (3 metrics x 2).
"""

from __future__ import annotations

import logging
import signal as _signal
import threading

import networkx as nx
import networkx.algorithms.community as nx_comm

_LOUVAIN_SEED = 42
WATCHDOG_SECONDS = 1.0  # SIGALRM resolution is 1 s (POSIX constraint)
_MIN_NODES_FOR_MODULARITY = 2  # Q undefined for V<2 (RC-2 topology guard)
_PARTITION_CACHE_MAX = 64
_HAS_SIGALRM = hasattr(_signal, "SIGALRM")

_log = logging.getLogger(__name__)


def _louvain_partition(undirected: nx.Graph) -> list[set]:
    return nx_comm.louvain_communities(undirected, seed=_LOUVAIN_SEED)


def _greedy_partition(undirected: nx.Graph) -> list[set]:
    return list(nx_comm.greedy_modularity_communities(undirected))


class _AlarmTimeoutError(Exception):
    pass


def _run_with_budget(fn, undirected: nx.Graph) -> list[set] | None:
    """1-second SIGALRM hard cut; no daemon threads (RC-4).

    SIGALRM is only available on the main thread; off-main-thread callers
    (and Windows) fall back to a bare try/except with no wall-clock bound.
    """
    on_main = threading.current_thread() is threading.main_thread()
    if not _HAS_SIGALRM or not on_main:
        try:
            return fn(undirected)
        except Exception:
            return None

    def _h(s, f) -> None:
        raise _AlarmTimeoutError()

    old = _signal.signal(_signal.SIGALRM, _h)
    _signal.alarm(1)
    try:
        r = fn(undirected)
        _signal.alarm(0)
        return r
    except (_AlarmTimeoutError, Exception):
        return None
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old)


_partition_cache: dict[tuple[frozenset, frozenset], list[set] | None] = {}


def _graph_key(undirected: nx.Graph) -> tuple[frozenset, frozenset]:
    """Cache key: (nodes-frozenset, edges-frozenset). Node-set required to avoid
    collisions between graphs with the same edges but different isolated nodes."""
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

    Projects the ``DiGraph`` to undirected (Q is defined there); empty /
    single-node graphs return ``0.0``. Louvain runs under a 1 s SIGALRM cut
    (RC-4); on failure it falls back to greedy CNM, then to ``0.0`` so the
    env.step caller never blocks. Returns Q as a float (higher = stronger
    community structure; ``0.0`` for the trivial / empty / wedged case).
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
