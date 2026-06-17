"""Unit tests for src/utils/config_loader.py."""

from __future__ import annotations

import pytest

from src.utils import config_loader
from src.utils.config_loader import (
    get_canonical_reward_coeffs,
    get_ppo_clip_eps,
    get_ppo_config,
    get_seeds,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure each test starts with a clean lru_cache."""
    load_config.cache_clear()
    yield
    load_config.cache_clear()


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "reward" in cfg
    assert "ppo" in cfg
    assert "seeds" in cfg


def test_load_config_caches():
    """Second call must return the exact same object due to lru_cache."""
    first = load_config()
    second = load_config()
    assert id(first) == id(second)


def test_load_config_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)


def test_get_canonical_reward_coeffs_correct():
    alpha, beta, gamma, p_skills = get_canonical_reward_coeffs()
    assert alpha == 1.0
    assert beta == 1.0
    assert gamma == 0.5
    assert p_skills == -5.0


def test_p_skills_must_be_negative(monkeypatch):
    """If P_skills is positive, the canonical accessor must reject it."""
    bad_cfg = {
        "reward": {"alpha": 1.0, "beta": 1.0, "gamma": 0.5, "p_skills": 5.0},
        "seeds": [1, 2, 3, 4, 5],
        "ppo": {"clip_eps": 0.2},
    }
    monkeypatch.setattr(config_loader, "load_config", lambda: bad_cfg)
    with pytest.raises(AssertionError, match="P_skills must be negative"):
        get_canonical_reward_coeffs()


def test_get_seeds_has_at_least_5():
    seeds = get_seeds()
    assert len(seeds) >= 5
    assert all(isinstance(s, int) for s in seeds)


def test_get_seeds_too_few_raises(monkeypatch):
    bad_cfg = {"seeds": [1, 2, 3]}
    monkeypatch.setattr(config_loader, "load_config", lambda: bad_cfg)
    with pytest.raises(AssertionError, match=">=5 seeds"):
        get_seeds()


def test_clip_eps_is_exactly_0_2():
    assert get_ppo_clip_eps() == 0.2


def test_clip_eps_wrong_value_raises(monkeypatch):
    bad_cfg = {"ppo": {"clip_eps": 0.3}}
    monkeypatch.setattr(config_loader, "load_config", lambda: bad_cfg)
    with pytest.raises(AssertionError, match=r"clip_eps must be 0\.2"):
        get_ppo_clip_eps()


def test_get_ppo_config_sources_every_tunable():
    """All PPO hyperparameters resolve from config.ppo (CLAUDE.md §4 single source)."""
    ppo = get_ppo_config()
    assert ppo["clip_eps"] == 0.2
    assert ppo["gae_lambda"] == 0.95
    assert ppo["gamma"] == 0.99
    assert ppo["lr"] == pytest.approx(3.0e-4)
    assert ppo["n_steps"] == 128
    assert ppo["n_epochs"] == 4
    assert ppo["batch_size"] == 64
    assert ppo["vf_coef"] == 0.5
    assert set(ppo) == {
        "clip_eps",
        "gae_lambda",
        "gamma",
        "lr",
        "n_steps",
        "n_epochs",
        "batch_size",
        "vf_coef",
    }


def test_get_ppo_config_rejects_non_canonical_gae_lambda(monkeypatch):
    bad_cfg = {
        "ppo": {
            "clip_eps": 0.2,
            "gae_lambda": 0.9,
            "gamma": 0.99,
            "learning_rate": 3.0e-4,
            "n_steps": 1,
            "n_epochs": 1,
            "batch_size": 1,
            "vf_coef": 0.5,
        }
    }
    monkeypatch.setattr(config_loader, "load_config", lambda: bad_cfg)
    with pytest.raises(AssertionError, match=r"gae_lambda must be 0\.95"):
        get_ppo_config()
