"""Unit tests for ``src.env.skills_graph_env.SkillsGraphEnv``.

Pinning the brief §2.2 contract:
    * NO ``gymnasium`` inheritance.
    * ``step(action)`` returns the **4-tuple** ``(state, reward, done, info)``.
    * Betweenness centrality is called **exactly twice per seed** — once in
      ``__init__`` (training start) and once in ``final_betweenness()``
      (training end). The per-seed cap is enforced by
      ``CentralityScheduler`` and validated here at the env level.
    * Termination triggers on ``max_episode_steps`` or ``|V| < 2``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import src.env.skills_graph_env as env_mod
from src.env.actions import A_MAX_TOTAL, Action, ActionKind
from src.env.skills_graph_env import SkillsGraphEnv
from src.env.state import State


@pytest.fixture()
def tiny_source_tree(tmp_path: Path) -> Path:
    """Two-file Python tree → graph with enough nodes for non-trivial steps."""
    (tmp_path / "a.py").write_text(
        "def foo():\n    return 1\n\ndef bar():\n    return foo()\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "from a import foo\nclass C:\n    def m(self): foo()\n",
        encoding="utf-8",
    )
    return tmp_path


def _noop() -> Action:
    return Action(kind=ActionKind.NOOP, primary=0, secondary=0)


def test_env_is_not_gym_env(tiny_source_tree: Path) -> None:
    """Brief §2.2 ban: no gymnasium inheritance / API leakage."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    base_names = [cls.__module__ for cls in type(env).__mro__]
    assert not any("gymnasium" in m or m.startswith("gym.") for m in base_names), (
        f"SkillsGraphEnv MRO leaks gymnasium: {base_names}"
    )
    # And the module itself must not bind a `gymnasium` symbol.
    assert "gymnasium" not in {*getattr(env_mod, "__dict__", {}).keys()}


def test_reset_returns_state_and_info(tiny_source_tree: Path) -> None:
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    state, info = env.reset()
    assert isinstance(state, State)
    assert isinstance(info, dict)
    assert info["seed"] == 42
    assert info["num_nodes"] == env.graph.number_of_nodes()
    assert info["step"] == 0
    assert "edge_counts" in info


def test_step_returns_state_reward_done_info_quadruple(tiny_source_tree: Path) -> None:
    """Brief §2.2: 4-tuple (state, reward, done, info) — NOT the gym 5-tuple."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    env.reset()
    result = env.step(_noop())
    assert isinstance(result, tuple) and len(result) == 4, (
        f"step() must return a 4-tuple, got len={len(result)}"
    )
    state, reward, done, info = result
    assert isinstance(state, State)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert info["step"] == 1


def test_betweenness_called_at_init(tiny_source_tree: Path) -> None:
    """CALL 1 of 2 per seed lands inside ``__init__`` (training-start anchor)."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    assert env.centrality.betweenness_calls == 1, (
        "training-start betweenness must fire exactly once during __init__"
    )


def test_betweenness_called_at_final(tiny_source_tree: Path) -> None:
    """CALL 2 of 2 per seed lands inside ``final_betweenness()`` (end anchor)."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    _ = env.final_betweenness()
    assert env.centrality.betweenness_calls == 2, (
        "training-end betweenness must bump the counter to exactly 2"
    )
    # Third call must be refused — protects the canonical budget.
    with pytest.raises(RuntimeError, match="exceeded"):
        env.final_betweenness()


def test_done_when_max_steps_reached(tiny_source_tree: Path) -> None:
    env = SkillsGraphEnv(tiny_source_tree, seed=42, max_episode_steps=3)
    env.reset()
    flags = [env.step(_noop())[2] for _ in range(3)]
    assert flags == [False, False, True], (
        f"done flags must be False,False,True at max_episode_steps=3; got {flags}"
    )


def test_done_when_too_few_nodes(tiny_source_tree: Path, monkeypatch) -> None:
    """``done`` must trip when |V| < 2 even before ``max_episode_steps``."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42, max_episode_steps=100)
    env.reset()
    # Force the underlying graph to a degenerate state and verify the env reports done.
    nodes = list(env.graph.nodes())
    env.graph.remove_nodes_from(nodes[1:])
    _, _, done, _ = env.step(_noop())
    assert done is True, "env must terminate when |V| drops below 2"


def test_action_mask_has_canonical_shape(tiny_source_tree: Path) -> None:
    """get_action_mask shape must match the canonical A_max=45057 (CLAUDE.md)."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    mask = env.get_action_mask()
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (A_MAX_TOTAL,) == (45057,), (
        f"mask shape must equal canonical A_max=(45057,); got {tuple(mask.shape)}"
    )
    assert bool(mask[-1].item()) is True, "NOOP slot must always be legal"
