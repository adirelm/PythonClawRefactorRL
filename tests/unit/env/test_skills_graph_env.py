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

import sys
import types
from pathlib import Path

import networkx as nx
import pytest
import torch

import src.env.skills_graph_env as env_mod
from src.env import reward as reward_mod
from src.env.actions import A_MAX_TOTAL, Action, ActionKind
from src.env.reward import RewardComponents, compute_reward
from src.env.skills_graph_env import SkillsGraphEnv
from src.env.state import State

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SKILLS_DIR = REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"


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
    assert not any("gymnasium" in m or m.startswith("gym.") for m in base_names), base_names
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
    assert isinstance(result, tuple) and len(result) == 4, f"got len={len(result)}"
    state, reward, done, info = result
    assert isinstance(state, State) and isinstance(reward, float)
    assert isinstance(done, bool) and isinstance(info, dict)
    assert info["step"] == 1


def test_betweenness_called_at_init(tiny_source_tree: Path) -> None:
    """CALL 1 of 2 per seed lands inside ``__init__`` (training-start anchor)."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    assert env.centrality.betweenness_calls == 1, "training-start betweenness must fire once"


def test_betweenness_called_at_final(tiny_source_tree: Path) -> None:
    """CALL 2 of 2 per seed lands inside ``final_betweenness()`` (end anchor)."""
    env = SkillsGraphEnv(tiny_source_tree, seed=42)
    _ = env.final_betweenness()
    assert env.centrality.betweenness_calls == 2, "training-end betweenness must bump to 2"
    # Third call must be refused — protects the canonical budget.
    with pytest.raises(RuntimeError, match="exceeded"):
        env.final_betweenness()


def test_done_when_max_steps_reached(tiny_source_tree: Path) -> None:
    env = SkillsGraphEnv(tiny_source_tree, seed=42, max_episode_steps=3)
    env.reset()
    flags = [env.step(_noop())[2] for _ in range(3)]
    assert flags == [False, False, True], f"got {flags}"


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
    assert mask.shape == (A_MAX_TOTAL,) == (45057,), f"got {tuple(mask.shape)}"
    assert bool(mask[-1].item()) is True, "NOOP slot must always be legal"


def test_step_produces_nonzero_reward_when_action_changes_graph() -> None:
    """Phase-3 end-to-end: SPLIT actions mutate the graph (via real refactor
    ops) → metrics deltas → at least one non-zero reward across a short
    SPLIT sweep over the sample_skills graph.
    """
    env = SkillsGraphEnv(SAMPLE_SKILLS_DIR, seed=42)
    n0 = env.graph.number_of_nodes()
    assert n0 >= 10, f"sample_skills graph must have >=10 nodes; got {n0}"
    env.reset()
    rewards: list[float] = []
    for primary in range(8):  # 8 SPLITs over distinct nodes
        _, reward, _, info = env.step(Action(kind=ActionKind.SPLIT, primary=primary, secondary=0))
        rewards.append(reward)
    assert env.graph.number_of_nodes() > n0, "SPLIT must add shadow nodes"
    assert any(r != 0.0 for r in rewards), f"expected at least one non-zero reward; got {rewards}"
    assert info["history_len"] == 8, "all 8 SPLIT actions must be recorded in history"


def test_step_uses_reward_coeff_overrides(tiny_source_tree: Path, monkeypatch) -> None:
    """AB-PLUMB: reward_* kwargs must flow into compute_reward via env.step."""
    captured: dict = {}

    def _fake(gb, ga, *, lazy_load_broken=False, **kw):
        captured.update(kw)
        return RewardComponents(0.0, 0.0, 0.0, 0.0, 42.0)

    monkeypatch.setattr("src.env.skills_graph_env.compute_reward", _fake)
    env = SkillsGraphEnv(
        tiny_source_tree, seed=42,
        reward_alpha=2.5, reward_beta=3.5, reward_gamma=0.25, reward_p_skills=-7.0,
    )
    env.reset()
    assert env.step(_noop())[1] == 42.0
    assert captured == {"alpha": 2.5, "beta": 3.5, "gamma": 0.25, "p_skills": -7.0}


def test_compute_reward_uses_canonical_coeffs(monkeypatch: pytest.MonkeyPatch) -> None:
    """compute_reward picks alpha/beta/gamma/P_skills from config and returns
    a non-zero RewardComponents when metrics move between snapshots.
    """
    assert (reward_mod.alpha, reward_mod.beta, reward_mod.gamma, reward_mod.p_skills) == (1.0, 1.0, 0.5, -5.0)
    before, after = nx.DiGraph(), nx.DiGraph()
    before.add_nodes_from(range(3))
    after.add_nodes_from(range(5))
    fake = types.ModuleType("src.services.metrics")
    fake.compute_modularity = lambda g: {3: 0.2, 5: 0.7}[g.number_of_nodes()]
    fake.compute_cohesion = lambda g: {3: 0.4, 5: 0.6}[g.number_of_nodes()]
    fake.compute_coupling = lambda g: {3: 0.1, 5: 0.5}[g.number_of_nodes()]
    monkeypatch.setitem(sys.modules, "src.services.metrics", fake)
    out = compute_reward(before, after)
    assert isinstance(out, RewardComponents)
    actual = (out.delta_modularity, out.delta_cohesion, out.coupling_penalty, out.p_skills_term)
    assert actual == (pytest.approx(0.5), pytest.approx(0.2), pytest.approx(0.4), 0.0)
    # 1.0*0.5 + 1.0*0.2 - 0.5*0.4 = 0.5
    assert out.total == pytest.approx(0.5) and out.total != 0.0
