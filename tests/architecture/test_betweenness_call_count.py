"""Architectural contract: betweenness centrality is called EXACTLY twice per seed.

Brief §2.2 + ADR-006 + CLAUDE.md canonical discipline:
    1. Training start (initial graph)
    2. Training end (final graph)

Any extra call leaks compute budget; any missing call breaks the Δ-Betweenness
comparison required in ANALYSIS.md.

Phase-2 surface area: ``SkillsGraphEnv.__init__`` makes CALL 1 (initial
graph snapshot) and ``SkillsGraphEnv.final_betweenness()`` makes CALL 2
(final graph snapshot). Phase-3 will wire the SDK ``train_one_seed`` loop
around this same env contract; the call-count invariant is identical.

A 3rd betweenness call within the same seed must raise ``RuntimeError``
from ``CentralityScheduler`` — the canonical budget is hard, not advisory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.env.skills_graph_env import SkillsGraphEnv
from src.services import centrality as centrality_mod

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"


def test_betweenness_called_exactly_twice_per_seed() -> None:
    """compute_betweenness must be invoked exactly 2 times in one seed's env lifecycle."""
    with patch(
        "src.services.centrality.compute_betweenness",
        wraps=centrality_mod.compute_betweenness,
    ) as spy:
        env = SkillsGraphEnv(SOURCE_DIR, seed=42)  # CALL 1 (init / start)
        env.reset()
        # A typical training loop would run env.step(...) here; betweenness
        # is NOT computed per-step (Degree only — ADR-006), so step calls
        # must NOT bump the spy count.
        env.final_betweenness()  # CALL 2 (end)

    assert spy.call_count == 2, (
        f"compute_betweenness called {spy.call_count} times; canonical spec "
        f"(CLAUDE.md + brief §2.2 + ADR-006) requires EXACTLY 2 calls per "
        f"seed: once at training start, once at training end."
    )


def test_betweenness_third_call_raises() -> None:
    """The scheduler's hard budget must block any 3rd call within the same seed."""
    env = SkillsGraphEnv(SOURCE_DIR, seed=42)  # CALL 1
    env.final_betweenness()  # CALL 2
    with pytest.raises(RuntimeError, match="budget"):
        env.final_betweenness()  # CALL 3 — must raise
