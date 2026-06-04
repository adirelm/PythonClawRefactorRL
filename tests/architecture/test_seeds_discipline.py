"""Architectural test for ADR-006 multi-seed discipline.

Asserts every results/*_seeded.json sidecar (if present) has ≥5 seeds.
Phase 1+ will generate the sidecars; until then this test is no-op +
documents the convention.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SIDECARS = list((REPO / "results").glob("*_seeded.json"))

MIN_SEEDS = 5  # ADR-006 lock


def test_seeds_convention_documented():
    """Phase 0: assert ADR-006 file exists with ≥5 seeds claim."""
    adr = REPO / "docs" / "adr" / "ADR-006-multi-seed-eval-discipline.md"
    assert adr.exists()
    text = adr.read_text()
    assert "5 seeds" in text or "≥5" in text or ">= 5" in text


@pytest.mark.skipif(not SIDECARS, reason="Phase 1+ will produce sidecars")
def test_each_sidecar_has_at_least_5_seeds():
    for sidecar in SIDECARS:
        data = json.loads(sidecar.read_text())
        seeds = data.get("seeds", [])
        assert len(seeds) >= MIN_SEEDS, (
            f"{sidecar.name} has only {len(seeds)} seeds; ADR-006 requires ≥{MIN_SEEDS}"
        )
