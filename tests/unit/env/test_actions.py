"""Unit tests for src/env/actions.py — flat global action index encoding."""

from __future__ import annotations

import random

import pytest

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
    Action,
    ActionKind,
    action_to_global_index,
    global_index_to_action,
)
from src.utils.config_loader import load_config


def test_a_max_matches_config():
    """A_MAX_TOTAL must equal config.action.a_max_total (= 45057)."""
    cfg = load_config()
    assert cfg["action"]["a_max_total"] == 45057
    assert A_MAX_TOTAL == 45057
    assert A_MAX_TOTAL == 4096 + 8192 + 32768 + 1


def test_noop_index_is_45056():
    assert NOOP_INDEX == 45056
    a = Action(kind=ActionKind.NOOP, primary=0, secondary=0)
    assert action_to_global_index(a) == 45056
    decoded = global_index_to_action(NOOP_INDEX)
    assert decoded == Action(kind=ActionKind.NOOP, primary=0, secondary=0)


def test_split_range_correct():
    """SPLIT indices fill [0, 4096) and decode back to SPLIT kind."""
    assert SPLIT_OFFSET == 0 and MERGE_OFFSET == 4096
    a0 = Action(kind=ActionKind.SPLIT, primary=0, secondary=0)
    a_last = Action(kind=ActionKind.SPLIT, primary=V_MAX_DEFAULT - 1, secondary=K_SPLIT - 1)
    assert action_to_global_index(a0) == 0
    assert action_to_global_index(a_last) == 4095
    for idx in (0, 1, 7, 8, 4095):
        d = global_index_to_action(idx)
        assert d.kind is ActionKind.SPLIT
        assert 0 <= d.primary < V_MAX_DEFAULT and 0 <= d.secondary < K_SPLIT


def test_merge_range_correct():
    """MERGE indices fill [4096, 12288) and decode back to MERGE kind."""
    assert REWIRE_OFFSET == 12288
    a0 = Action(kind=ActionKind.MERGE, primary=0, secondary=0)
    a_last = Action(kind=ActionKind.MERGE, primary=V_MAX_DEFAULT - 1, secondary=M_MERGE - 1)
    assert action_to_global_index(a0) == 4096
    assert action_to_global_index(a_last) == 12287
    for idx in (4096, 4111, 4112, 12287):
        d = global_index_to_action(idx)
        assert d.kind is ActionKind.MERGE
        assert 0 <= d.primary < V_MAX_DEFAULT and 0 <= d.secondary < M_MERGE


def test_rewire_range_correct():
    """REWIRE indices fill [12288, 45056) and decode back to REWIRE kind."""
    a0 = Action(kind=ActionKind.REWIRE, primary=0, secondary=0)
    a_last = Action(kind=ActionKind.REWIRE, primary=E_MAX - 1, secondary=R_REWIRE - 1)
    assert action_to_global_index(a0) == 12288
    assert action_to_global_index(a_last) == 45055
    for idx in (12288, 12295, 12296, 45055):
        d = global_index_to_action(idx)
        assert d.kind is ActionKind.REWIRE
        assert 0 <= d.primary < E_MAX and 0 <= d.secondary < R_REWIRE


def test_encoding_decoding_roundtrip():
    """For 100 random valid actions across all kinds, decode(encode(a)) == a."""
    rng = random.Random(42)
    samples: list[Action] = []
    for _ in range(33):
        samples.append(Action(ActionKind.SPLIT, rng.randrange(V_MAX_DEFAULT), rng.randrange(K_SPLIT)))
        samples.append(Action(ActionKind.MERGE, rng.randrange(V_MAX_DEFAULT), rng.randrange(M_MERGE)))
        samples.append(Action(ActionKind.REWIRE, rng.randrange(E_MAX), rng.randrange(R_REWIRE)))
    samples.append(Action(ActionKind.NOOP, 0, 0))
    assert len(samples) >= 100
    for action in samples:
        assert global_index_to_action(action_to_global_index(action)) == action


def test_index_roundtrip_full_sweep():
    """Every valid flat index round-trips: encode(decode(idx)) == idx."""
    for idx in range(A_MAX_TOTAL):
        assert action_to_global_index(global_index_to_action(idx)) == idx


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        global_index_to_action(-1)
    with pytest.raises(ValueError):
        global_index_to_action(A_MAX_TOTAL)
    with pytest.raises(ValueError):
        action_to_global_index(Action(ActionKind.SPLIT, V_MAX_DEFAULT, 0))
    with pytest.raises(ValueError):
        action_to_global_index(Action(ActionKind.REWIRE, 0, R_REWIRE))
    with pytest.raises(ValueError):
        action_to_global_index(Action(ActionKind.NOOP, 1, 0))
