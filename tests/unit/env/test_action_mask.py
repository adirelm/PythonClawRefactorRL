"""Unit tests for src.env.action_mask.compute_mask (Phase 2 — ACTION_DESIGN §3).

Covers the Huang & Ontañon (2022) pre-softmax legal-action mask contract:
mask is a ``bool`` tensor of shape ``(45057,)`` with True = legal. The four
tests below pin the corner cases the action space depends on, plus the
always-True NOOP slot and the L1↔L3 lazy-load break (ADR-005)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from src.env.action_mask import compute_mask
from src.env.actions import (
    A_MAX_TOTAL,
    K_SPLIT,
    M_MERGE,
    MERGE_OFFSET,
    NOOP_INDEX,
    R_REWIRE,
    REWIRE_OFFSET,
    SPLIT_OFFSET,
)
from src.env.state import State
from src.pythonclaw_shim.registry import SkillRegistry
from src.services.lazy_load_monitor import LazyLoadMonitor

_FEATURE_DIM = 16
_COL_L1, _COL_L3 = 5, 7


def _mk_state(
    *,
    num_nodes: int,
    edges: list[tuple[int, int]] | None = None,
    layers: dict[int, int] | None = None,
) -> State:
    """Synthesize a State without going through GraphifyAdapter.

    ``layers`` maps ``node_id -> {1,2,3}`` for Skills L1/L2/L3 one-hots.
    """
    edges = edges or []
    rows = [u for u, _ in edges]
    cols = [v for _, v in edges]
    data = [1.0] * len(edges)
    a = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    x = torch.zeros((num_nodes, _FEATURE_DIM), dtype=torch.float32)
    # Spread features so cosine sim is non-degenerate.
    for i in range(num_nodes):
        x[i, i % _FEATURE_DIM] = 1.0
    if layers:
        for nid, layer in layers.items():
            col = {1: 5, 2: 6, 3: 7}[layer]
            x[nid, col] = 1.0
    node_ids = tuple(f"n{i}" for i in range(num_nodes))
    return State(A=a, X=x, node_ids=node_ids, edge_type_counts={"call": len(edges)})


def test_noop_always_legal() -> None:
    """NOOP slot (index 45056) is True for every state — ACTION_DESIGN §2.4."""
    empty = _mk_state(num_nodes=0)
    populated = _mk_state(num_nodes=3, edges=[(0, 1), (0, 2)])
    assert bool(compute_mask(empty)[NOOP_INDEX])
    assert bool(compute_mask(populated)[NOOP_INDEX])


def test_mask_shape_matches_a_max() -> None:
    """Mask is exactly A_MAX_TOTAL=45057 booleans (frozen by config.action)."""
    mask = compute_mask(_mk_state(num_nodes=2, edges=[(0, 1)]))
    assert mask.shape == (45057,)
    assert mask.shape[0] == A_MAX_TOTAL
    assert mask.dtype == torch.bool


def test_split_illegal_when_no_children() -> None:
    """SPLIT(node, k) is False whenever children_count < k+2.

    Node 0 has zero children → every SPLIT slot k∈[0,K) is False; node 0 also
    has no out-edges, so a 1-node graph cannot split for any k.
    """
    state = _mk_state(num_nodes=1)  # no edges → no children
    mask = compute_mask(state)
    for k in range(K_SPLIT):
        assert not bool(mask[SPLIT_OFFSET + 0 * K_SPLIT + k]), f"SPLIT(0,{k}) should be illegal"


def test_merge_illegal_with_single_node() -> None:
    """|V|=1 → every MERGE slot is False (no partner can exist)."""
    state = _mk_state(num_nodes=1)
    mask = compute_mask(state)
    merge_block = mask[MERGE_OFFSET : MERGE_OFFSET + M_MERGE]
    assert not merge_block.any(), "MERGE block must be all-False when |V|=1"


def test_rewire_illegal_for_nonexistent_edge() -> None:
    """REWIRE(edge_id ≥ num_edges, *) must be False — no edge to redirect."""
    state = _mk_state(num_nodes=3, edges=[(0, 1)])  # only 1 edge
    mask = compute_mask(state)
    # edge_id=5 is past num_edges=1, so all R slots for that edge are False.
    nonexistent_edge = 5
    for slot in range(R_REWIRE):
        assert not bool(mask[REWIRE_OFFSET + nonexistent_edge * R_REWIRE + slot]), (
            f"REWIRE(edge={nonexistent_edge}, slot={slot}) must be illegal"
        )


def test_merge_l1_l3_breaks_lazy_load_and_logs_event() -> None:
    """L1↔L3 merge is masked False AND logs a LazyLoadEvent via the monitor."""
    state = _mk_state(num_nodes=2, layers={0: 1, 1: 3})
    monitor = LazyLoadMonitor(registry=SkillRegistry())
    mask = compute_mask(state, monitor=monitor)
    # Node 0's only possible partner is node 1; that merge is L1↔L3 → False.
    assert not bool(mask[MERGE_OFFSET + 0 * M_MERGE + 0])
    assert any(ev.broken_check_name == "merge_l1_and_l3" for ev in monitor.events)
