"""Unit tests for ``metric_trace`` — per-step modularity/cohesion/coupling
snapshots that feed the brief §3 "improvement in modularity, cohesion, coupling"
curves. ``graph_metrics`` is pure; ``policy_metric_rollout`` replays a policy.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.env.skills_graph_env import SkillsGraphEnv
from src.model.policy_net import PolicyNet
from src.services._metric_trace import graph_metrics, policy_metric_rollout

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SKILLS = REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"


def _two_module_graph() -> nx.DiGraph:
    """Two 3-node cohesive clusters joined by a single cross edge."""
    g = nx.DiGraph()
    g.add_edges_from([("a1", "a2"), ("a2", "a3"), ("a3", "a1"), ("b1", "b2"), ("b2", "b3"), ("a1", "b1")])
    return g


def test_graph_metrics_returns_three_float_keys() -> None:
    m = graph_metrics(_two_module_graph())
    assert set(m) == {"modularity", "cohesion", "coupling"}
    for key, value in m.items():
        assert isinstance(value, float), f"{key} not float: {value!r}"


def test_policy_metric_rollout_records_step0_plus_n_steps() -> None:
    env = SkillsGraphEnv(SAMPLE_SKILLS, seed=42)
    policy = PolicyNet()
    rows = policy_metric_rollout(env, policy, n_steps=3)
    assert [r["step"] for r in rows] == [0, 1, 2, 3]  # initial snapshot + 3 edits
    for row in rows:
        assert {"step", "modularity", "cohesion", "coupling"} <= set(row)
        for key in ("modularity", "cohesion", "coupling"):
            assert isinstance(row[key], float)
