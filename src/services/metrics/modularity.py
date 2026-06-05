"""Newman-Girvan modularity (Q) for the skills-graph MDP.

Phase 2 fix - wires the first leg of the canonical reward equation
``R_t = alpha*dModularity + beta*dCohesion - gamma*Coupling_Penalty + P_skills``.

Definitions:
- ``compute_modularity(graph)`` runs Louvain community detection on the
  undirected projection of ``graph`` and returns the resulting Q score
  (``[-0.5, 1.0]`` per Newman 2006). A zero-edge or zero-node graph has
  no community structure, so we short-circuit to ``0.0``.
- ``delta_modularity(before, after)`` is the signed change used directly
  by ``compute_reward``: positive dQ means the refactor increased
  modularity (good for the agent).

Determinism: Louvain is a randomised algorithm, so we pass ``seed=42``
to make training-time rewards reproducible. The seed is intentionally
the same constant used elsewhere in the repo so cross-module determinism
holds end-to-end.
"""

from __future__ import annotations

import networkx as nx
import networkx.algorithms.community as nx_comm

_LOUVAIN_SEED = 42


def compute_modularity(graph: nx.DiGraph) -> float:
    """Return the Newman-Girvan modularity ``Q`` of ``graph``.

    The skills-graph is a ``nx.DiGraph``; modularity is defined on
    undirected graphs, so we project via ``to_undirected()`` before
    running Louvain. Empty / single-node graphs return ``0.0`` because
    Q is undefined when there are no edges to partition.

    Args:
        graph: Directed skills-graph snapshot (e.g. ``env._graph``).

    Returns:
        Modularity score as a ``float``. Higher = stronger community
        structure. ``0.0`` for the trivial / empty case.
    """
    if graph.number_of_edges() == 0 or graph.number_of_nodes() == 0:
        return 0.0
    undirected = graph.to_undirected()
    communities = nx_comm.louvain_communities(undirected, seed=_LOUVAIN_SEED)
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
