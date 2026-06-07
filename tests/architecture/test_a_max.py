"""Architectural contract: A_max = 45057 is sealed.

Pins the flat-index action-space size derivation per
docs/ACTION_DESIGN.md §1 + CLAUDE.md §CANONICAL VALUES:

    N_SPLIT  = V_max * K_split  = 512 * 8  =  4096
    N_MERGE  = V_max * M_merge  = 512 * 16 =  8192
    N_REWIRE = E_max * R_rewire = 4096 * 8 = 32768
    N_NOOP   = 1
    -----------------------------------------------
    A_max    = 4096 + 8192 + 32768 + 1     = 45057

Any drift in V_MAX_DEFAULT / K_SPLIT / M_MERGE / E_MAX / R_REWIRE / N_NOOP
breaks the policy-head sizing, the action-mask encoding, and every saved
checkpoint. The import-time assertion in src/env/actions.py catches it at
load time; these tests catch it during CI even if someone bypasses the
import (e.g. by re-wiring constants from config without restarting).
"""

from __future__ import annotations

from src.env.actions import (
    A_MAX_TOTAL,
    E_MAX,
    K_SPLIT,
    M_MERGE,
    MERGE_OFFSET,
    N_MERGE,
    N_NOOP,
    N_REWIRE,
    N_SPLIT,
    NOOP_INDEX,
    R_REWIRE,
    REWIRE_OFFSET,
    SPLIT_OFFSET,
    V_MAX_DEFAULT,
)


def test_a_max_total_is_sealed_45057() -> None:
    """The sealed canonical value (CLAUDE.md) must equal 45057 — literal lock."""
    assert A_MAX_TOTAL == 45057


def test_a_max_total_equals_sum_of_categories() -> None:
    """A_MAX_TOTAL is exactly N_SPLIT + N_MERGE + N_REWIRE + N_NOOP."""
    assert A_MAX_TOTAL == N_SPLIT + N_MERGE + N_REWIRE + N_NOOP


def test_category_counts_match_derivation() -> None:
    """Each N_* literal must equal its V_max/E_max × multiplier derivation."""
    assert N_SPLIT == V_MAX_DEFAULT * K_SPLIT == 4096
    assert N_MERGE == V_MAX_DEFAULT * M_MERGE == 8192
    assert N_REWIRE == E_MAX * R_REWIRE == 32768
    assert N_NOOP == 1


def test_sizing_constants_are_canonical() -> None:
    """V_max / E_max / K / M / R literals are sealed in CLAUDE.md."""
    assert V_MAX_DEFAULT == 512
    assert E_MAX == 4096
    assert K_SPLIT == 8
    assert M_MERGE == 16
    assert R_REWIRE == 8


def test_offsets_form_contiguous_partition() -> None:
    """SPLIT/MERGE/REWIRE/NOOP slabs tile [0, A_MAX_TOTAL) with no gaps."""
    assert SPLIT_OFFSET == 0
    assert MERGE_OFFSET == N_SPLIT == 4096
    assert REWIRE_OFFSET == N_SPLIT + N_MERGE == 12288
    assert NOOP_INDEX == N_SPLIT + N_MERGE + N_REWIRE == 45056


def test_noop_is_last_valid_index() -> None:
    """NOOP_INDEX is the final flat index; A_MAX_TOTAL is exclusive upper bound."""
    assert NOOP_INDEX == A_MAX_TOTAL - 1 == 45056
