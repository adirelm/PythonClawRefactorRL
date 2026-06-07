"""Canonical reward function for the PythonClaw refactor MDP (brief sec 2.2).

R_t = alpha*dModularity + beta*dCohesion - gamma*Coupling_Penalty + P_skills_t

Defaults from ``config/config.yaml#reward`` (sealed in Phase 0):
alpha=1.0, beta=1.0, gamma=0.5, P_skills=-5.0 (NEGATIVE penalty).

``P_skills_t`` is applied only when the lazy-load monitor flags a break;
otherwise ``p_skills_term`` is 0.0. Module-level constants ``alpha``,
``beta``, ``gamma``, ``p_skills`` are exposed for the architectural
contract test in ``tests/architecture/test_reward_formula.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.config_loader import get_canonical_reward_coeffs

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    import networkx as nx

# Module-level canonical constants (loaded from config at import time so that
# tests/architecture/test_reward_formula.py can introspect them as floats).
alpha, beta, gamma, p_skills = get_canonical_reward_coeffs()
assert p_skills < 0.0, f"P_skills must be negative, got {p_skills}"


@dataclass(frozen=True)
class RewardComponents:
    """Decomposition of a single reward step.

    Attributes:
        delta_modularity: Modularity(after) - Modularity(before).
        delta_cohesion:   Cohesion(after) - Cohesion(before).
        coupling_penalty: Coupling(after) - Coupling(before).  Positive value
            means coupling **increased**; the reward formula subtracts
            ``gamma * coupling_penalty`` so increases hurt the agent.
        p_skills_term: ``p_skills`` (a NEGATIVE float, default ``-5.0``)
            iff the lazy-load monitor flagged the step; otherwise ``0.0``.
        total: ``alpha*dMod + beta*dCoh - gamma*coupling_penalty + p_skills_term``.
    """

    delta_modularity: float
    delta_cohesion: float
    coupling_penalty: float
    p_skills_term: float
    total: float


def _resolve_coeffs(
    alpha_arg: float | None,
    beta_arg: float | None,
    gamma_arg: float | None,
    p_skills_arg: float | None,
) -> tuple[float, float, float, float]:
    """Return (alpha, beta, gamma, p_skills); fall back to canonical config."""
    cfg_alpha, cfg_beta, cfg_gamma, cfg_p_skills = get_canonical_reward_coeffs()
    a = float(cfg_alpha if alpha_arg is None else alpha_arg)
    b = float(cfg_beta if beta_arg is None else beta_arg)
    g = float(cfg_gamma if gamma_arg is None else gamma_arg)
    ps = float(cfg_p_skills if p_skills_arg is None else p_skills_arg)
    assert ps < 0.0, f"P_skills must be negative, got {ps}"
    return a, b, g, ps


def compute_reward(  # noqa: PLR0913 - signature dictated by canonical reward equation
    graph_before: nx.DiGraph,
    graph_after: nx.DiGraph,
    *,
    lazy_load_broken: bool = False,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    p_skills: float | None = None,
) -> RewardComponents:
    """Compute canonical R_t between two consecutive graph snapshots.

    Metric primitives (``compute_modularity/cohesion/coupling``) are
    imported lazily from ``src.services.metrics`` so this module loads
    cleanly before the metrics phase lands. Tests inject deterministic
    values by stubbing ``sys.modules["src.services.metrics"]``.
    Optional ``alpha/beta/gamma/p_skills`` override the canonical config.
    ``lazy_load_broken=True`` triggers the negative ``P_skills`` term.
    """
    a, b, g, ps = _resolve_coeffs(alpha, beta, gamma, p_skills)

    # Late import so an as-yet-unimplemented metrics module does not break
    # import of src.env.reward. Tests inject these via monkey-patch on
    # ``src.env.reward`` for deterministic before/after deltas.
    from src.services.metrics import (  # noqa: PLC0415
        compute_cohesion,
        compute_coupling,
        compute_modularity,
    )

    # RC-1 optimization: compute Louvain partition once per snapshot and pass
    # to all 3 metrics so a wedge-prone graph costs <=2 watchdog timeouts per
    # env.step (one for before, one for after) instead of 6 (3 metrics x 2).
    # Tests monkey-patch src.services.metrics with stub functions; in that
    # case the modularity submodule import fails and we fall through to the
    # legacy per-metric path (partitions stay None).
    try:
        from src.services.metrics.modularity import safe_louvain  # noqa: PLC0415

        part_before = safe_louvain(graph_before.to_undirected()) if graph_before.number_of_edges() else None
        part_after = safe_louvain(graph_after.to_undirected()) if graph_after.number_of_edges() else None
    except (ImportError, AttributeError):
        part_before = None
        part_after = None

    # Only pass _partition when computed (test stubs don't accept the kwarg).
    after_kw = {} if part_after is None else {"_partition": part_after}
    before_kw = {} if part_before is None else {"_partition": part_before}

    delta_modularity = float(
        compute_modularity(graph_after, **after_kw) - compute_modularity(graph_before, **before_kw)
    )
    delta_cohesion = float(
        compute_cohesion(graph_after, **after_kw) - compute_cohesion(graph_before, **before_kw)
    )
    coupling_penalty = float(
        compute_coupling(graph_after, **after_kw) - compute_coupling(graph_before, **before_kw)
    )
    p_skills_term = ps if lazy_load_broken else 0.0

    total = a * delta_modularity + b * delta_cohesion - g * coupling_penalty + p_skills_term
    return RewardComponents(
        delta_modularity=delta_modularity,
        delta_cohesion=delta_cohesion,
        coupling_penalty=coupling_penalty,
        p_skills_term=p_skills_term,
        total=total,
    )
