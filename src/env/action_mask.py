"""Pre-softmax action mask (ADR-005 + ACTION_DESIGN §3).

Returns a ``bool`` tensor shape ``(A_max,)=(45057,)`` where ``True`` marks
legal actions. Illegal logits are forced to ``-inf`` pre-softmax per
**Huang & Ontañon (2022) "A Closer Look at Invalid Action Masking in
Policy Gradient Algorithms"** — pre-softmax masking keeps PPO unbiased.
Offsets / sizes come from ``src.env.actions`` (single source of truth).
"""

from __future__ import annotations

import torch

from src.env.actions import (
    A_MAX_TOTAL,
    E_MAX,
    K_SPLIT,
    M_MERGE,
    MERGE_OFFSET,
    NOOP_INDEX,
    R_REWIRE,
    REWIRE_OFFSET,
    SPLIT_OFFSET,
    V_MAX_DEFAULT,
)
from src.env.state import State
from src.services.lazy_load_monitor import LazyLoadEvent, LazyLoadMonitor

_COL_L1, _COL_L2, _COL_L3 = 5, 6, 7  # layer one-hots (STATE_DESIGN §3)
_COL_DIN, _COL_DOUT = 2, 3
_ONE_HOT_THRESHOLD = 0.5  # X col is a {0,1} one-hot — 0.5 separates the buckets
_MIN_NODES_FOR_MERGE = 2  # MERGE requires at least one possible partner


def _layer_of(state: State, node: int) -> int:
    """Return Skills layer ∈ {1,2,3} for ``node``, or 0 if non-Skill."""
    row = state.X[node]
    for layer, col in ((1, _COL_L1), (2, _COL_L2), (3, _COL_L3)):
        if float(row[col]) > _ONE_HOT_THRESHOLD:
            return layer
    return 0


def _children_count(state: State, node: int) -> int:
    """Out-edges from ``node`` (its children in the directed call/import graph)."""
    return int(state.A.indptr[node + 1] - state.A.indptr[node])


def _split_mask(state: State, k_choices: int, mask: torch.Tensor) -> None:
    """SPLIT(node, k): legal iff node < num_nodes AND children ≥ k+2."""
    if state.num_nodes <= 0:
        return
    for node in range(min(state.num_nodes, V_MAX_DEFAULT)):
        children = _children_count(state, node)
        for k in range(k_choices):
            if children >= k + 2:
                mask[SPLIT_OFFSET + node * k_choices + k] = True


def _topm_similar(state: State, node: int, m: int) -> list[int]:
    """Top-M most cosine-similar nodes to ``node`` (self excluded)."""
    if state.num_nodes < _MIN_NODES_FOR_MERGE:
        return []
    x = state.X[: state.num_nodes].float()
    xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sims = xn @ xn[node]
    sims[node] = float("-inf")
    return torch.topk(sims, min(m, state.num_nodes - 1)).indices.tolist()


def _breaks_lazy(state: State, a: int, b: int, monitor: LazyLoadMonitor | None) -> bool:
    """L1↔L3 merge would break Skills lazy-load invariant (ADR-005)."""
    la, lb = _layer_of(state, a), _layer_of(state, b)
    if la and lb and {la, lb} == {1, 3}:
        if monitor is not None:
            monitor.log_event(
                LazyLoadEvent(
                    skill_id=f"merge:{a}->{b}",
                    layer=3,
                    broken_check_name="merge_l1_and_l3",
                    actual=1,
                    threshold=0,
                )
            )
        return True
    return False


def _merge_mask(state: State, m_choices: int, mask: torch.Tensor, monitor: LazyLoadMonitor | None) -> None:
    """MERGE(a, idx): legal iff |V|≥2 AND idx in top-M sims AND no lazy break."""
    if state.num_nodes < _MIN_NODES_FOR_MERGE:
        return
    for node_a in range(min(state.num_nodes, V_MAX_DEFAULT)):
        for slot, partner in enumerate(_topm_similar(state, node_a, m_choices)):
            if slot >= m_choices:
                break
            if _breaks_lazy(state, node_a, partner, monitor):
                continue
            mask[MERGE_OFFSET + node_a * m_choices + slot] = True


def _rewire_mask(state: State, r_choices: int, mask: torch.Tensor) -> None:
    """REWIRE(edge, target): legal iff edge exists AND target is top-R lowest degree."""
    if state.num_edges <= 0 or state.num_nodes <= 0:
        return
    degree = state.X[: state.num_nodes, _COL_DIN] + state.X[: state.num_nodes, _COL_DOUT]
    r = min(r_choices, state.num_nodes)
    candidates = torch.topk(degree, r, largest=False).indices.tolist()
    n_slots = min(r_choices, len(candidates))
    for edge_id in range(min(state.num_edges, E_MAX)):
        for slot in range(n_slots):
            mask[REWIRE_OFFSET + edge_id * r_choices + slot] = True


def compute_mask(
    state: State,
    *,
    K: int = K_SPLIT,  # noqa: N803 — spec-mandated casing (CLAUDE.md §CANONICAL)
    M: int = M_MERGE,  # noqa: N803
    R: int = R_REWIRE,  # noqa: N803
    monitor: LazyLoadMonitor | None = None,
) -> torch.Tensor:
    """Build the pre-softmax legal-action mask for ``state``.

    Args:
        state: Live MDP state (``src.env.state.State``).
        K, M, R: Split / Merge / Rewire fan-outs — CLAUDE.md §CANONICAL.
        monitor: Optional ``LazyLoadMonitor``; receives one ``LazyLoadEvent``
            per merge masked out due to an L1↔L3 lazy-load break.

    Returns:
        ``torch.BoolTensor`` of shape ``(A_MAX_TOTAL,) == (45057,)`` where
        ``True`` marks a legal action.
    """
    mask = torch.zeros(A_MAX_TOTAL, dtype=torch.bool)
    _split_mask(state, K, mask)
    _merge_mask(state, M, mask, monitor)
    _rewire_mask(state, R, mask)
    mask[NOOP_INDEX] = True  # NOOP always legal — ACTION_DESIGN §2.4
    return mask
