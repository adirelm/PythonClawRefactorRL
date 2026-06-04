"""Architectural test for ADR-010 convergence schema.

Asserts:
- config/config.yaml has convergence block with the 4 required keys
- Non-overlapping windows (step ≥ window_length)
- check_convergence raises NotImplementedError until Phase 1+
"""

from pathlib import Path

import pytest
import yaml

from src.services.convergence import (
    PARTIAL_CONVERGENCE,
    ConvergenceVerdict,
    check_convergence,
)

REPO = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((REPO / "config" / "config.yaml").read_text())

# Re-exported so Phase 1+ tests can extend without re-importing.
__all__ = ["PARTIAL_CONVERGENCE", "ConvergenceVerdict", "check_convergence"]


def test_config_has_convergence_block():
    assert "convergence" in CFG
    for key in ["window_length", "step", "reward_rel_tol", "entropy_slope_threshold"]:
        assert key in CFG["convergence"], f"missing {key}"


def test_non_overlapping_windows():
    conv = CFG["convergence"]
    assert conv["step"] >= conv["window_length"], "windows overlap; ADR-010 requires non-overlap"


def test_check_convergence_stub_raises():
    with pytest.raises(NotImplementedError, match="Phase 1"):
        check_convergence([], [], config=CFG)
