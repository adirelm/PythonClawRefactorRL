"""Custom (non-gym) refactor environment over the PythonClaw Skills graph.

Brief §2.2 bans ``gymnasium``: no ``gym.Env`` inheritance, no ``gym.spaces``,
no ``gym.register``. Training loop is owned by the SDK and consumed by the
custom PPO service (ADR-007). Contract: ``reset()`` → ``(state, info)``;
``step(action)`` → ``(state, reward, done, info)`` (4-tuple, not gymnasium's
5-tuple); termination = ``max_episode_steps`` or ``|V|<2``. Betweenness is
computed exactly twice per seed — once in ``__init__`` (start), once in
``final_betweenness()`` (end), capped by ``CentralityScheduler``. Phase-2
``_apply_action`` records history but leaves the graph as-is; real
split/merge/rewire effects land in Phase 3 with the metrics service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.env.action_mask import compute_mask
from src.env.actions import Action, ActionKind
from src.env.reward import RewardComponents, compute_reward
from src.env.state import State
from src.graphify.local_impl import LocalGraphify
from src.services.centrality import CentralityScheduler

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODE_STEPS = 64  # Phase-2 default; SDK overrides per ADR-007.
MIN_NODES_FOR_STEP = 2  # |V| < this → done (no valid 2-node refactor possible)


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

    def __init__(
        self,
        source_dir: Path,
        *,
        seed: int,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self._graph = LocalGraphify().build(self.source_dir, seed=self.seed)
        self.centrality = CentralityScheduler(seed=self.seed)
        # registry=None is the env-only mode (SDK supplies the real registry).
        self.lazy_monitor = None
        self._initial_betweenness = self.centrality.compute_betweenness(self._graph)  # CALL 1/2
        self._current_state = State.from_digraph(self._graph)
        self._initial_state = self._current_state
        self._info = _EpisodeInfo(
            seed=self.seed,
            num_nodes=self._graph.number_of_nodes(),
            num_edges=self._graph.number_of_edges(),
            edge_counts=dict(self._current_state.edge_type_counts),
        )

    def reset(self) -> tuple[State, dict]:
        """Return the initial state and an info dict."""
        self._current_state = self._initial_state
        self._info.step = 0
        self._info.history.clear()
        return self._current_state, self._info.as_dict()

    def step(self, action: Action) -> tuple[State, float, bool, dict]:
        """Advance one MDP step. Returns (state, reward, done, info)."""
        graph_before = self._graph.copy()
        self._graph = _apply_action(self._graph, action)
        self._current_state = State.from_digraph(self._graph)
        reward = _safe_reward(graph_before, self._graph, lazy_broken=False)
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


def _apply_action(graph, action: Action):
    """Phase-2 stub: NOOP is a true no-op; other action kinds are recorded
    in ``_EpisodeInfo.history`` but leave the graph untouched. Real
    split/merge/rewire effects land in Phase 3 with the metrics service.
    """
    if action.kind is ActionKind.NOOP:
        return graph
    return graph


def _safe_reward(graph_before, graph_after, *, lazy_broken: bool) -> float:
    """Call ``compute_reward`` tolerating any future metrics-import gap.

    metrics services now exist in ``src/services/metrics/``; this catch is
    purely defensive against future relocations of that package. On import
    error we fall back to canonical ``P_skills=-5.0`` if ``lazy_broken``
    else ``0.0`` so the env never crashes a rollout mid-episode.
    """
    try:
        components: RewardComponents = compute_reward(graph_before, graph_after, lazy_load_broken=lazy_broken)
        return float(components.total)
    except (ModuleNotFoundError, ImportError):
        return float(-5.0 if lazy_broken else 0.0)
