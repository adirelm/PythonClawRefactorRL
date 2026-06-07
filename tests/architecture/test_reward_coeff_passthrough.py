"""Architectural contract: AB-PLUMB reward-coefficient passthrough.

Wave-2 Stream A (AB-PLUMB) threads ``alpha/beta/gamma/p_skills`` from the
``scripts/train_ppo.py`` CLI down to ``compute_reward`` through the env.
Future ablation runners depend on this contract; we pin it here so a
silent regression in either layer breaks CI loudly.

Two seams checked:
    1. ``SkillsGraphEnv.__init__`` accepts the four ``reward_*`` kwargs.
    2. ``scripts/train_ppo.py`` argparse exposes ``--alpha --beta --gamma
       --p-skills`` (parsed by importing ``_parse_args``).
"""

from __future__ import annotations

import importlib
import inspect


def test_env_accepts_reward_coeff_kwargs() -> None:
    """SkillsGraphEnv.__init__ exposes the 4 AB-PLUMB override kwargs."""
    env_mod = importlib.import_module("src.env.skills_graph_env")
    params = inspect.signature(env_mod.SkillsGraphEnv.__init__).parameters
    for name in ("reward_alpha", "reward_beta", "reward_gamma", "reward_p_skills"):
        assert name in params, f"SkillsGraphEnv.__init__ missing kwarg: {name}"
        assert params[name].default is None, f"{name} must default to None (canonical fallback)"


def test_train_ppo_cli_exposes_reward_flags() -> None:
    """scripts/train_ppo.py argparse exposes the 4 AB-PLUMB override flags."""
    train_mod = importlib.import_module("scripts.train_ppo")
    ns = train_mod._parse_args(["--alpha", "1.0", "--beta", "1.0", "--gamma", "0.5", "--p-skills", "-5.0"])
    assert ns.alpha == 1.0
    assert ns.beta == 1.0
    assert ns.gamma == 0.5
    assert ns.p_skills == -5.0


def test_train_ppo_cli_defaults_to_none() -> None:
    """Without overrides the CLI yields None → env falls back to canonical config."""
    train_mod = importlib.import_module("scripts.train_ppo")
    ns = train_mod._parse_args([])
    assert ns.alpha is None
    assert ns.beta is None
    assert ns.gamma is None
    assert ns.p_skills is None
