"""Action space for PythonClaw Refactor RL.

Discrete, masked, parametric action space per docs/ACTION_DESIGN.md §1.

A_max derivation (CLAUDE.md §CANONICAL VALUES, sealed value 45057):
    N_SPLIT  = V_max * K_split  = 512 * 8  =  4096
    N_MERGE  = V_max * M_merge  = 512 * 16 =  8192
    N_REWIRE = E_max * R_rewire = 4096 * 8 = 32768
    N_NOOP   = 1
    -----------------------------------------------
    A_max    = 4096 + 8192 + 32768 + 1     = 45057

Encoding (flat global index ∈ [0, A_max)):
- SPLIT  : idx = primary * K            + secondary     ; range [0, 4096)
- MERGE  : idx = 4096    + primary * M  + secondary     ; range [4096, 12288)
- REWIRE : idx = 12288   + primary * R  + secondary     ; range [12288, 45056)
- NOOP   : idx = 45056

Drift guard: any change to V_max / K_split / M_merge / E_max / R_rewire
breaks the policy-head sizing and the sealed contract — caught by the
import-time assert below AND by tests/architecture/test_a_max.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# Canonical sizing constants (mirrored from config.action / CLAUDE.md).
V_MAX_DEFAULT: int = 512
K_SPLIT: int = 8
M_MERGE: int = 16
R_REWIRE: int = 8
E_MAX: int = 4096

# Per-category counts.
N_SPLIT: int = V_MAX_DEFAULT * K_SPLIT  # 4096
N_MERGE: int = V_MAX_DEFAULT * M_MERGE  # 8192
N_REWIRE: int = E_MAX * R_REWIRE  # 32768
N_NOOP: int = 1

# Offsets into the flat index space.
SPLIT_OFFSET: int = 0
MERGE_OFFSET: int = SPLIT_OFFSET + N_SPLIT  # 4096
REWIRE_OFFSET: int = MERGE_OFFSET + N_MERGE  # 12288
NOOP_INDEX: int = REWIRE_OFFSET + N_REWIRE  # 45056

A_MAX_TOTAL: int = N_SPLIT + N_MERGE + N_REWIRE + N_NOOP  # 45057


class ActionKind(IntEnum):
    """Structural action categories. Values are stable for serialization."""

    NOOP = 0
    SPLIT = 1
    MERGE = 2
    REWIRE = 3


@dataclass(frozen=True)
class Action:
    """A single discrete action.

    `primary` is a node_id index (SPLIT / MERGE) or an edge_id index (REWIRE).
    `secondary` encodes:
      - SPLIT  : split_point ∈ [0, K_SPLIT)         (default 0)
      - MERGE  : merge_target_idx ∈ [0, M_MERGE)    (default 0)
      - REWIRE : rewire_target_idx ∈ [0, R_REWIRE)  (default 0)
      - NOOP   : unused; must be 0
    """

    kind: ActionKind
    primary: int
    secondary: int = 0


def _check_split(primary: int, secondary: int, v_max: int) -> None:
    if not (0 <= primary < v_max):
        raise ValueError(f"SPLIT primary={primary} out of range [0, {v_max})")
    if not (0 <= secondary < K_SPLIT):
        raise ValueError(f"SPLIT secondary={secondary} out of range [0, {K_SPLIT})")


def _check_merge(primary: int, secondary: int, v_max: int) -> None:
    if not (0 <= primary < v_max):
        raise ValueError(f"MERGE primary={primary} out of range [0, {v_max})")
    if not (0 <= secondary < M_MERGE):
        raise ValueError(f"MERGE secondary={secondary} out of range [0, {M_MERGE})")


def _check_rewire(primary: int, secondary: int) -> None:
    if not (0 <= primary < E_MAX):
        raise ValueError(f"REWIRE primary={primary} out of range [0, {E_MAX})")
    if not (0 <= secondary < R_REWIRE):
        raise ValueError(f"REWIRE secondary={secondary} out of range [0, {R_REWIRE})")


def action_to_global_index(action: Action, *, V_max: int = V_MAX_DEFAULT) -> int:  # noqa: N803 — spec-mandated casing (docs/ACTION_DESIGN.md)
    """Map an `Action` to its flat global index ∈ [0, A_MAX_TOTAL)."""
    kind = action.kind
    p, s = action.primary, action.secondary
    if kind is ActionKind.SPLIT:
        _check_split(p, s, V_max)
        return SPLIT_OFFSET + p * K_SPLIT + s
    if kind is ActionKind.MERGE:
        _check_merge(p, s, V_max)
        return MERGE_OFFSET + p * M_MERGE + s
    if kind is ActionKind.REWIRE:
        _check_rewire(p, s)
        return REWIRE_OFFSET + p * R_REWIRE + s
    if kind is ActionKind.NOOP:
        if p != 0 or s != 0:
            raise ValueError(f"NOOP must have primary=0, secondary=0; got ({p}, {s})")
        return NOOP_INDEX
    raise ValueError(f"Unknown ActionKind: {kind!r}")


def global_index_to_action(idx: int) -> Action:
    """Inverse of `action_to_global_index` over the flat index space."""
    if not (0 <= idx < A_MAX_TOTAL):
        raise ValueError(f"idx={idx} out of range [0, {A_MAX_TOTAL})")
    if idx == NOOP_INDEX:
        return Action(kind=ActionKind.NOOP, primary=0, secondary=0)
    if idx < MERGE_OFFSET:
        primary, secondary = divmod(idx - SPLIT_OFFSET, K_SPLIT)
        return Action(kind=ActionKind.SPLIT, primary=primary, secondary=secondary)
    if idx < REWIRE_OFFSET:
        primary, secondary = divmod(idx - MERGE_OFFSET, M_MERGE)
        return Action(kind=ActionKind.MERGE, primary=primary, secondary=secondary)
    primary, secondary = divmod(idx - REWIRE_OFFSET, R_REWIRE)
    return Action(kind=ActionKind.REWIRE, primary=primary, secondary=secondary)


# Sanity: A_MAX_TOTAL = 4096 + 8192 + 32768 + 1 = 45057 per docs/ACTION_DESIGN
# §1 and config.action.a_max_total. Asserted at import time so any constants
# drift fails fast before the policy head is sized.
_EXPECTED_NOOP_INDEX = 4096 + 8192 + 32768  # = 45056
_EXPECTED_A_MAX_TOTAL = _EXPECTED_NOOP_INDEX + 1  # = 45057
assert NOOP_INDEX == _EXPECTED_NOOP_INDEX, f"NOOP_INDEX drifted: {NOOP_INDEX}"
assert A_MAX_TOTAL == _EXPECTED_A_MAX_TOTAL, f"A_MAX_TOTAL drifted: {A_MAX_TOTAL}"
