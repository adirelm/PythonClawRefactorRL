"""Architectural contract: canonical reward equation constants.

Brief §2.2 verbatim:
    R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t

Defaults from config/config.yaml#reward:
    α = 1.0, β = 1.0, γ = 0.5, P_skills = -5.0  (NEGATIVE penalty)

Phase 2 populated ``src/env/reward.py`` with the canonical constants
sourced from ``config/config.yaml#reward``; these tests now actively
guard the contract (was xfail through Phase 1).
"""

from __future__ import annotations

import importlib

import yaml


def test_reward_constants_have_correct_types_and_signs() -> None:
    """alpha/beta/gamma are floats from config; p_skills is float AND NEGATIVE."""
    reward_mod = importlib.import_module("src.env.reward")

    for name in ("alpha", "beta", "gamma", "p_skills"):
        assert hasattr(reward_mod, name), f"src.env.reward missing constant: {name}"
        val = getattr(reward_mod, name)
        assert isinstance(val, float), f"{name} must be float, got {type(val).__name__}"

    assert reward_mod.p_skills < 0.0, (
        "P_skills MUST be negative — it is a PENALTY applied when the "
        "lazy-load monitor detects a break (CLAUDE.md canonical reward eqn)."
    )


def test_reward_constants_match_config_yaml(repo_root) -> None:
    """Imported constants must equal config/config.yaml#reward values."""
    reward_mod = importlib.import_module("src.env.reward")

    cfg = yaml.safe_load((repo_root / "config" / "config.yaml").read_text())
    rcfg = cfg["reward"]
    assert reward_mod.alpha == float(rcfg["alpha"])
    assert reward_mod.beta == float(rcfg["beta"])
    assert reward_mod.gamma == float(rcfg["gamma"])
    assert reward_mod.p_skills == float(rcfg["p_skills"])
