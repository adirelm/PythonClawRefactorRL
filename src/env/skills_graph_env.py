"""Non-gym refactor env (brief §2.2). 4-tuple step, betweenness 2x/seed.

Phase-3 ``_apply_action`` dispatches SPLIT/MERGE/REWIRE via
``src.env._apply_action`` (kept in a sibling module so this file stays
≤150 LOC per CLAUDE.md §1). Failed ops log + pass so the agent learns
to avoid them through the reward signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.env._apply_action import RefactorOpError
from src.env._apply_action import apply as _apply_action
from src.env.action_mask import compute_mask
from src.env.actions import Action
from src.env.reward import RewardComponents, compute_reward
from src.env.state import State
from src.graphify.local_impl import LocalGraphify
from src.services.centrality import CentralityScheduler

DEFAULT_MAX_EPISODE_STEPS = 64
MIN_NODES_FOR_STEP = 2

__all__ = ["DEFAULT_MAX_EPISODE_STEPS", "RefactorOpError", "SkillsGraphEnv"]


@dataclass
class _EpisodeInfo:
    seed: int
    num_nodes: int
    num_edges: int
    edge_counts: dict[str, int]
    step: int = 0
    history: list[Action] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "edge_counts": dict(self.edge_counts),
            "step": self.step,
            "history_len": len(self.history),
        }


class SkillsGraphEnv:
    """Custom training loop — NO Gymnasium (brief §2.2 ban)."""

    def __init__(  # noqa: PLR0913 - 4 AB-PLUMB reward overrides + canonical kwargs
        self,
        source_dir: Path,
        *,
        seed: int,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        reward_alpha: float | None = None,
        reward_beta: float | None = None,
        reward_gamma: float | None = None,
        reward_p_skills: float | None = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self._reward_coeffs: dict[str, float | None] = {
            "alpha": reward_alpha,
            "beta": reward_beta,
            "gamma": reward_gamma,
            "p_skills": reward_p_skills,
        }
        self._graph = LocalGraphify().build(self.source_dir, seed=self.seed)
        self.centrality = CentralityScheduler(seed=self.seed)
        self.lazy_monitor = None  # SDK supplies the real registry.
        self._initial_betweenness = self.centrality.compute_betweenness(self._graph)  # CALL 1/2
        self._initial_graph = self._graph.copy()  # restored on reset() so episodes are i.i.d.
        self._current_state = State.from_digraph(self._graph)
        self._initial_state = self._current_state
        self._info = _EpisodeInfo(
            seed=self.seed,
            num_nodes=self._graph.number_of_nodes(),
            num_edges=self._graph.number_of_edges(),
            edge_counts=dict(self._current_state.edge_type_counts),
        )

    def reset(self) -> tuple[State, dict]:
        """Restore the initial graph + state — episodes are truly i.i.d."""
        self._graph = self._initial_graph.copy()
        self._current_state = self._initial_state
        self._info.step = 0
        self._info.history.clear()
        self._info.num_nodes = self._graph.number_of_nodes()
        self._info.num_edges = self._graph.number_of_edges()
        self._info.edge_counts = dict(self._current_state.edge_type_counts)
        return self._current_state, self._info.as_dict()

    def step(self, action: Action) -> tuple[State, float, bool, dict]:
        """Advance one MDP step. Returns (state, reward, done, info)."""
        graph_before = self._graph.copy()
        self._graph = _apply_action(self._graph, action)
        self._current_state = State.from_digraph(self._graph)
        reward = _safe_reward(graph_before, self._graph, lazy_broken=False, coeffs=self._reward_coeffs)
        self._info.step += 1
        self._info.history.append(action)
        self._info.num_nodes = self._graph.number_of_nodes()
        self._info.num_edges = self._graph.number_of_edges()
        self._info.edge_counts = dict(self._current_state.edge_type_counts)
        done = self._info.step >= self.max_episode_steps or self._graph.number_of_nodes() < MIN_NODES_FOR_STEP
        return self._current_state, float(reward), bool(done), self._info.as_dict()

    def get_action_mask(self) -> torch.Tensor:
        """Pre-softmax legal-action mask for the current state."""
        return compute_mask(self._current_state, monitor=self.lazy_monitor)

    def final_betweenness(self) -> dict[str, float]:
        """Training-end betweenness call (CALL 2/2 per seed; ADR-006)."""
        return self.centrality.compute_betweenness(self._graph)  # CALL 2/2

    @property
    def graph(self):
        """Internal nx.DiGraph accessor (tests + SDK only; not policy-facing)."""
        return self._graph

    @property
    def current_state(self) -> State:
        return self._current_state


def _safe_reward(
    graph_before,
    graph_after,
    *,
    lazy_broken: bool,
    coeffs: dict[str, float | None] | None = None,
) -> float:
    """``compute_reward`` wrapped; falls back to ``P_skills``/``0.0`` on import gaps.

    ``coeffs`` forwards optional ``alpha/beta/gamma/p_skills`` overrides for
    the AB-PLUMB ablation; ``None`` entries fall back to canonical config.
    """
    overrides = {k: v for k, v in (coeffs or {}).items() if v is not None}
    try:
        comps: RewardComponents = compute_reward(
            graph_before, graph_after, lazy_load_broken=lazy_broken, **overrides
        )
        return float(comps.total)
    except (ModuleNotFoundError, ImportError):
        return float(-5.0 if lazy_broken else 0.0)
