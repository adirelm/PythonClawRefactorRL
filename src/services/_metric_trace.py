"""Per-step architecture-metric trace for the brief §3 improvement curves.

``graph_metrics`` snapshots ``(modularity, cohesion, coupling)`` for one graph;
``policy_metric_rollout`` replays a trained policy for a greedy eval episode and
records the snapshot at step 0 (the initial graph) and after every policy edit.
A renderer can then plot modularity/cohesion rising and coupling falling over
the rollout — the brief's "graphs of the improvement in modularity, cohesion,
and coupling reduction".

This is a read-only consumer of the env + metric services; it mutates nothing
the trainer relies on (it calls ``env.reset()`` itself, so it must run AFTER
any betweenness-budget-sensitive measurement on the same env).
"""

from __future__ import annotations

import torch

from src.env.actions import global_index_to_action
from src.env.skills_graph_env import SkillsGraphEnv
from src.model.policy_net import PolicyNet
from src.services._ppo_helpers import pad_state
from src.services.metrics.cohesion import compute_cohesion
from src.services.metrics.coupling import compute_coupling_penalty
from src.services.metrics.modularity import compute_modularity

__all__ = ["graph_metrics", "policy_metric_rollout"]

DEFAULT_ROLLOUT_STEPS = 128


def graph_metrics(graph) -> dict[str, float]:
    """Snapshot the three reward-component metrics for a single graph.

    ``coupling`` is the raw coupling penalty (higher = worse); the reward
    formula subtracts it, so an *improving* policy drives this term down.
    """
    return {
        "modularity": float(compute_modularity(graph)),
        "cohesion": float(compute_cohesion(graph)),
        "coupling": float(compute_coupling_penalty(graph)),
    }


def _row(step: int, graph) -> dict[str, float]:
    return {"step": float(step), **graph_metrics(graph)}


def policy_metric_rollout(
    env: SkillsGraphEnv, policy: PolicyNet, *, n_steps: int = DEFAULT_ROLLOUT_STEPS
) -> list[dict[str, float]]:
    """Greedy-replay ``policy`` on ``env`` for ``n_steps``; metrics each step.

    Returns ``n_steps + 1`` rows (step 0 is the pre-edit initial graph). The
    action-selection path mirrors ``PPOTrainer.collect_rollout`` (pad → policy
    → masked sample) so the trajectory matches what training optimised.
    """
    state, _ = env.reset()
    rows: list[dict[str, float]] = [_row(0, env.graph)]
    for step in range(1, n_steps + 1):
        with torch.no_grad():
            x, mask = pad_state(state)
            logits, _ = policy(x, mask)
            action_mask = env.get_action_mask().unsqueeze(0)
            action_idx, _ = policy.get_action(logits, action_mask)
        state, _, done, _ = env.step(global_index_to_action(int(action_idx.item())))
        rows.append(_row(step, env.graph))
        if done:
            state, _ = env.reset()
    return rows
