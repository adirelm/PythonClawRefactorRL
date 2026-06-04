"""Cached YAML config loader for PythonClawRefactorRL.

Single source of truth for canonical constants used across the codebase:
reward coefficients, seeds, and PPO clip epsilon. All accessors assert
their invariants (per ADR / brief) so misconfiguration fails fast.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"

# Brief / ADR invariants — module-level constants so accessors stay magic-number-free.
MIN_SEEDS = 5  # ADR-006
PPO_CLIP_EPS_CANONICAL = 0.2  # brief §2.3 (FIXED, do not tune)


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict:
    """Load YAML config; cached so repeated calls return the same dict object."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. Expected config/config.yaml at repo root."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} did not parse to a dict.")
    return data


def get_canonical_reward_coeffs() -> tuple[float, float, float, float]:
    """Return (alpha, beta, gamma, P_skills) from config.reward.

    Asserts P_skills < 0 per the canonical reward design (Phase-0 seal).
    """
    cfg = load_config()
    reward = cfg["reward"]
    for key in ("alpha", "beta", "gamma", "p_skills"):
        assert key in reward, f"reward.{key} missing from config"
    alpha = float(reward["alpha"])
    beta = float(reward["beta"])
    gamma = float(reward["gamma"])
    p_skills = float(reward["p_skills"])
    assert p_skills < 0, f"P_skills must be negative, got {p_skills}"
    return alpha, beta, gamma, p_skills


def get_seeds() -> list[int]:
    """Return config.seeds; asserts len >= 5 per ADR-006."""
    cfg = load_config()
    seeds = list(cfg["seeds"])
    assert len(seeds) >= MIN_SEEDS, f"Need >=5 seeds (ADR-006), got {len(seeds)}"
    return seeds


def get_ppo_clip_eps() -> float:
    """Return config.ppo.clip_eps; asserts == 0.2 per brief §2.3."""
    cfg = load_config()
    clip_eps = float(cfg["ppo"]["clip_eps"])
    assert clip_eps == PPO_CLIP_EPS_CANONICAL, f"PPO clip_eps must be 0.2 (brief §2.3), got {clip_eps}"
    return clip_eps
