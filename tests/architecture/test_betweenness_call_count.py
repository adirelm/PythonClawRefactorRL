"""Architectural contract: betweenness centrality is called EXACTLY twice per seed.

Brief §2.2 + ADR-006 + CLAUDE.md canonical discipline:
    1. Training start (initial graph)
    2. Training end (final graph)

Any extra call leaks compute budget; any missing call breaks the Δ-Betweenness
comparison required in ANALYSIS.md.

NOTE: config/config.yaml currently shows `centrality.betweenness_calls_per_seed: 3`
(start + end + final-eval). The CANONICAL spec (CLAUDE.md) pins it at 2.
This test enforces the canonical 2; if the project genuinely needs 3, the
config and CLAUDE.md must be reconciled BEFORE relaxing this assertion.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


@pytest.mark.xfail(
    reason="Phase 3 will implement src/services/centrality.py and the training loop",
    strict=False,
)
def test_betweenness_called_exactly_twice_per_seed() -> None:
    """compute_betweenness must be invoked exactly 2 times in one seed's training."""
    try:
        centrality = importlib.import_module("src.services.centrality")
    except ModuleNotFoundError:
        pytest.xfail("src/services/centrality.py not yet implemented (Phase 3 pending)")
        return

    try:
        sdk = importlib.import_module("src.sdk")
        train_one_seed = sdk.train_one_seed
    except (ModuleNotFoundError, AttributeError):
        pytest.xfail("src.sdk.train_one_seed not yet implemented (Phase 3 pending)")
        return

    with patch(
        "src.services.centrality.compute_betweenness",
        wraps=centrality.compute_betweenness,
    ) as spy:
        train_one_seed(seed=42)

    assert spy.call_count == 2, (
        f"compute_betweenness called {spy.call_count} times; canonical spec "
        f"(CLAUDE.md + brief §2.2 + ADR-006) requires EXACTLY 2 calls per "
        f"seed: once at training start, once at training end."
    )
