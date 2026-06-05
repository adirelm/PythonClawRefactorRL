"""Unit tests for src/env/reward.py — canonical R_t per brief §2.2.

R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling_Penalty + P_skills_t
α=1.0, β=1.0, γ=0.5, P_skills=-5.0 (NEGATIVE, applied only on lazy-load break).

These tests inject deterministic metric values by monkey-patching the
lazily-imported names on ``src.env.reward`` so we do not depend on the
yet-to-land ``src.services.metrics`` module.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import networkx as nx
import pytest

from src.env import reward as reward_mod
from src.env.reward import RewardComponents, compute_reward


def _install_metrics_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    modularity_map: dict[Any, float],
    cohesion_map: dict[Any, float],
    coupling_map: dict[Any, float],
) -> None:
    """Inject a fake ``src.services.metrics`` module the late import will hit.

    Each map is keyed by the graph's number_of_nodes(); the stubs return the
    matching value so distinct before/after graphs yield deterministic deltas.
    """
    fake = types.ModuleType("src.services.metrics")
    fake.compute_modularity = lambda g: modularity_map[g.number_of_nodes()]
    fake.compute_cohesion = lambda g: cohesion_map[g.number_of_nodes()]
    fake.compute_coupling = lambda g: coupling_map[g.number_of_nodes()]
    monkeypatch.setitem(sys.modules, "src.services.metrics", fake)


def _graph(n_nodes: int) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    return g


def test_reward_uses_canonical_coeffs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module-level constants come straight from config.reward (1.0, 1.0, 0.5, -5.0)."""
    assert reward_mod.alpha == 1.0
    assert reward_mod.beta == 1.0
    assert reward_mod.gamma == 0.5
    assert reward_mod.p_skills == -5.0

    # End-to-end: a no-op transition + lazy-load intact => total == 0.0
    _install_metrics_stub(
        monkeypatch,
        modularity_map={1: 0.5, 2: 0.5},
        cohesion_map={1: 0.3, 2: 0.3},
        coupling_map={1: 0.1, 2: 0.1},
    )
    out = compute_reward(_graph(1), _graph(2))
    assert isinstance(out, RewardComponents)
    assert out.delta_modularity == 0.0
    assert out.delta_cohesion == 0.0
    assert out.coupling_penalty == 0.0
    assert out.p_skills_term == 0.0
    assert out.total == 0.0


def test_p_skills_must_be_negative() -> None:
    """Sanity-check: explicit positive P_skills override is rejected at runtime."""
    with pytest.raises(AssertionError, match="P_skills must be negative"):
        compute_reward(_graph(1), _graph(2), p_skills=5.0)


def test_reward_increases_when_modularity_increases(monkeypatch: pytest.MonkeyPatch) -> None:
    """ΔModularity > 0 with everything else flat must yield total > 0."""
    _install_metrics_stub(
        monkeypatch,
        modularity_map={1: 0.2, 2: 0.7},  # +0.5
        cohesion_map={1: 0.4, 2: 0.4},
        coupling_map={1: 0.1, 2: 0.1},
    )
    out = compute_reward(_graph(1), _graph(2))
    assert out.delta_modularity == pytest.approx(0.5)
    assert out.delta_cohesion == 0.0
    assert out.coupling_penalty == 0.0
    assert out.p_skills_term == 0.0
    # alpha=1.0 -> total == 0.5
    assert out.total == pytest.approx(0.5)
    assert out.total > 0.0


def test_reward_decreases_when_coupling_increases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coupling rising with everything else flat must yield total < 0 (penalised)."""
    _install_metrics_stub(
        monkeypatch,
        modularity_map={1: 0.5, 2: 0.5},
        cohesion_map={1: 0.4, 2: 0.4},
        coupling_map={1: 0.1, 2: 0.9},  # +0.8
    )
    out = compute_reward(_graph(1), _graph(2))
    assert out.coupling_penalty == pytest.approx(0.8)
    # gamma=0.5 -> total == -0.5 * 0.8 = -0.4
    assert out.total == pytest.approx(-0.4)
    assert out.total < 0.0


def test_lazy_load_broken_applies_negative_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """lazy_load_broken=True folds in the negative P_skills = −5.0 term."""
    _install_metrics_stub(
        monkeypatch,
        modularity_map={1: 0.5, 2: 0.5},
        cohesion_map={1: 0.4, 2: 0.4},
        coupling_map={1: 0.1, 2: 0.1},
    )
    out = compute_reward(_graph(1), _graph(2), lazy_load_broken=True)
    assert out.p_skills_term == -5.0
    assert out.total == pytest.approx(-5.0)


def test_lazy_load_intact_applies_zero_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """lazy_load_broken=False (default) leaves the p_skills_term at 0.0."""
    _install_metrics_stub(
        monkeypatch,
        modularity_map={1: 0.5, 2: 0.5},
        cohesion_map={1: 0.4, 2: 0.4},
        coupling_map={1: 0.1, 2: 0.1},
    )
    out = compute_reward(_graph(1), _graph(2), lazy_load_broken=False)
    assert out.p_skills_term == 0.0
    assert out.total == 0.0
