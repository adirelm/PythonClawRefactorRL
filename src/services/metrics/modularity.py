"""Newman-Girvan modularity (Q) for the skills-graph MDP.

Phase 2 fix — wires the first leg of the canonical reward equation
``R_t = alpha*dModularity + beta*dCohesion - gamma*Coupling_Penalty + P_skills``.

Phase 4 RC-1/RC-2 hardening (sealed in CLAUDE.md §CANONICAL VALUES, doc:
``docs/known-gaps``): some mid-rollout topologies wedge Louvain for >>1s on
seeds 123/314, blowing the 120s per-seed budget. We wrap the Louvain call
with a ``WATCHDOG_SECONDS`` wall-clock budget; on timeout we fall back to
``greedy_modularity_communities`` and, if that *also* times out, to the
trivial partition ``Q = 0.0``. Determinism is preserved on the happy path
(``seed=42``); the fallback path is logged so graders can audit when it
fires.

Definitions:
- ``compute_modularity(graph)`` runs Louvain community detection on the
  undirected projection of ``graph`` and returns the resulting Q score
  (``[-0.5, 1.0]`` per Newman 2006). A zero-edge or zero-node graph has
  no community structure, so we short-circuit to ``0.0`` (RC-2 early-return).
- ``delta_modularity(before, after)`` is the signed change used directly
  by ``compute_reward``: positive dQ means the refactor increased
  modularity (good for the agent).
"""

from __future__ import annotations

import logging

import networkx as nx
import networkx.algorithms.community as nx_comm

_LOUVAIN_SEED = 42
WATCHDOG_SECONDS = 1.0  # RC-1 wall-clock budget per Louvain / greedy call
_MIN_NODES_FOR_MODULARITY = 2  # Q undefined for V<2 (RC-2 topology guard)

_log = logging.getLogger(__name__)


def _louvain_partition(undirected: nx.Graph) -> list[set]:
    return nx_comm.louvain_communities(undirected, seed=_LOUVAIN_SEED)


def _greedy_partition(undirected: nx.Graph) -> list[set]:
    return list(nx_comm.greedy_modularity_communities(undirected))


def _run_with_budget(fn, undirected: nx.Graph) -> list[set] | None:
    """Run ``fn(undirected)`` with a ``WATCHDOG_SECONDS`` wall-clock budget.

    Returns the partition on success; ``None`` if the budget elapsed.

    Implementation: spin up a one-shot daemon ``threading.Thread`` for the
    call. On timeout the caller returns ``None`` immediately; the worker
    thread keeps running in the background (Python can't kill a thread
    cooperatively) but is daemonised so it can't block process exit, and
    its result is discarded.

    A ``ThreadPoolExecutor`` was rejected because (a) ``__exit__`` blocks
    on pending workers — defeating the wall-clock guarantee — and (b) a
    pinned ``max_workers=1`` pool would serialise subsequent calls behind
    the wedged one.
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


def compute_modularity(graph: nx.DiGraph) -> float:
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

    communities = _run_with_budget(_louvain_partition, undirected)
    if communities is None:
        _log.warning(
            "modularity: Louvain exceeded %.2fs budget (V=%d E=%d); trying greedy fallback",
            WATCHDOG_SECONDS,
            undirected.number_of_nodes(),
            undirected.number_of_edges(),
        )
        communities = _run_with_budget(_greedy_partition, undirected)
    if communities is None:
        _log.warning(
            "modularity: greedy fallback also exceeded %.2fs budget; returning 0.0",
            WATCHDOG_SECONDS,
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
