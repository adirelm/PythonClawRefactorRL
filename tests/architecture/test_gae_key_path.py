"""Architectural contract: GAE keys live under `ppo:` block, NOT top-level.

PRD-GAE.md FR-4/FR-5/§6 mandates that the GAE smoothing parameter `λ` and
the discount factor `γ` are exposed via `config/config.yaml` as
`ppo.gae_lambda` and `ppo.gamma` respectively. This matches the
stable-baselines3 idiom (GAE is a PPO-trainer concern, not an independent
component) and is the canonical key path that downstream code uses.

A stale top-level `gae:` block in `config/config.yaml` would silently
shadow the canonical keys at Phase 3 PPO construction time and surface as
a `KeyError`. This test is the architectural gate that prevents that drift.

Asserts:
    1. `config["ppo"]["gae_lambda"] == 0.95`  (PRD-GAE FR-1, FR-4)
    2. `config["ppo"]["gamma"]      == 0.99`  (PRD-GAE FR-2, FR-5)
    3. `"gae" not in config`                  (no stale top-level block)
    4. `config["ppo"]["clip_eps"]   == 0.20`  (sibling-invariant cross-check)
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_config(repo_root: Path) -> dict:
    """Load `config/config.yaml` as a plain dict."""
    config_path = repo_root / "config" / "config.yaml"
    assert config_path.is_file(), f"missing config file: {config_path}"
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict), "config.yaml must be a YAML mapping"
    return loaded


def test_ppo_gae_lambda_canonical_value(repo_root: Path) -> None:
    """PRD-GAE FR-1, FR-4: `ppo.gae_lambda == 0.95`."""
    config = _load_config(repo_root)
    assert "ppo" in config, "config.yaml missing required `ppo:` block"
    assert config["ppo"].get("gae_lambda") == 0.95, (
        f"PRD-GAE FR-4 violated: expected `ppo.gae_lambda == 0.95`, got {config['ppo'].get('gae_lambda')!r}"
    )


def test_ppo_gamma_canonical_value(repo_root: Path) -> None:
    """PRD-GAE FR-2, FR-5: `ppo.gamma == 0.99`."""
    config = _load_config(repo_root)
    assert config["ppo"].get("gamma") == 0.99, (
        f"PRD-GAE FR-5 violated: expected `ppo.gamma == 0.99`, got {config['ppo'].get('gamma')!r}"
    )


def test_no_top_level_gae_block(repo_root: Path) -> None:
    """No stale top-level `gae:` block (PRD-GAE §6 — GAE is a PPO concern)."""
    config = _load_config(repo_root)
    assert "gae" not in config, (
        "Stale top-level `gae:` block detected in config.yaml — "
        "GAE keys must live under `ppo.gae_lambda` / `ppo.gamma` "
        "per PRD-GAE FR-4/FR-5/§6 (SB3 idiom)."
    )


def test_ppo_clip_eps_sibling_invariant(repo_root: Path) -> None:
    """Cross-check sibling invariant: `ppo.clip_eps == 0.2` (ex04 §2.3)."""
    config = _load_config(repo_root)
    assert config["ppo"].get("clip_eps") == 0.2, (
        f"ex04 §2.3 violated: expected `ppo.clip_eps == 0.2`, got {config['ppo'].get('clip_eps')!r}"
    )
