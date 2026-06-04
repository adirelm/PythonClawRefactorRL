"""Unit tests for src/pythonclaw_shim/skill.py.

Pins the PRD-SKILLS §4 lazy-load invariant: reading .metadata or
.estimated_tokens(layer) MUST NOT trigger L2/L3 loading.
"""

from __future__ import annotations

import pytest

from src.pythonclaw_shim.skill import Skill


def _trip_wire_loader(flag: dict, layer: str, payload: dict):
    """Return a loader that flips ``flag[layer]`` when called."""

    def _load() -> dict:
        flag[layer] = True
        return payload

    return _load


def _make_skill(flag: dict | None = None) -> Skill:
    flag = flag if flag is not None else {"l2": False, "l3": False}
    return Skill(
        name="file_search",
        version="1.0.0",
        metadata={
            "name": "file_search",
            "version": "1.0.0",
            "description": "Searches the local filesystem.",
            "estimated_tokens": {"L1": 50, "L2": 500, "L3": 5000},
            "tags": ["search", "filesystem"],
        },
        instructions_loader=_trip_wire_loader(flag, "l2", {"prompt": "search"}),
        resources_loader=_trip_wire_loader(flag, "l3", {"files": ["a", "b"]}),
    )


def test_skill_construction() -> None:
    skill = _make_skill()
    assert skill.name == "file_search"
    assert skill.version == "1.0.0"
    assert skill.metadata["description"].startswith("Searches")
    assert skill.metadata["tags"] == ["search", "filesystem"]


def test_metadata_accessible_without_loading_l2_or_l3() -> None:
    flag = {"l2": False, "l3": False}
    skill = _make_skill(flag)
    # Touch metadata + estimated_tokens for L2/L3 — neither must fire loaders.
    _ = skill.metadata
    _ = skill.estimated_tokens(1)
    _ = skill.estimated_tokens(2)
    _ = skill.estimated_tokens(3)
    assert skill.has_instructions is False
    assert skill.has_resources is False
    assert flag == {"l2": False, "l3": False}


def test_load_instructions_populates_l2() -> None:
    flag = {"l2": False, "l3": False}
    skill = _make_skill(flag)
    out = skill.load_instructions()
    assert out == {"prompt": "search"}
    assert skill.has_instructions is True
    assert skill.has_resources is False
    assert flag["l2"] is True and flag["l3"] is False
    # Idempotent: second call must not re-trigger the loader.
    flag["l2"] = False
    skill.load_instructions()
    assert flag["l2"] is False


def test_load_resources_populates_l3() -> None:
    flag = {"l2": False, "l3": False}
    skill = _make_skill(flag)
    out = skill.load_resources()
    assert out == {"files": ["a", "b"]}
    assert skill.has_resources is True
    assert flag["l3"] is True


def test_estimated_tokens_l1_smaller_than_l3() -> None:
    skill = _make_skill()
    l1 = skill.estimated_tokens(1)
    l2 = skill.estimated_tokens(2)
    l3 = skill.estimated_tokens(3)
    assert l1 < l2 < l3


def test_estimated_tokens_invalid_layer_raises() -> None:
    skill = _make_skill()
    with pytest.raises(ValueError):
        skill.estimated_tokens(99)


def test_load_instructions_without_loader_raises() -> None:
    skill = Skill(name="bare", version="0.1.0", metadata={"name": "bare"})
    with pytest.raises(RuntimeError):
        skill.load_instructions()
    with pytest.raises(RuntimeError):
        skill.load_resources()


def test_estimated_tokens_uses_loaded_payload_when_available() -> None:
    """Once L2/L3 are loaded, estimated_tokens(layer) counts the real payload."""
    skill = _make_skill()
    skill.load_instructions()
    skill.load_resources()
    # Loaded payloads are tiny; hint metadata claims L2=500, L3=5000.
    # The loaded-payload branch must be used (real count, not the hint).
    assert skill.estimated_tokens(2) < 500
    assert skill.estimated_tokens(3) < 5000


def test_estimated_tokens_fallback_when_no_hint_in_metadata() -> None:
    """Without an estimated_tokens hint, fall back to metadata-derived counts."""
    skill = Skill(name="bare", version="0.1.0", metadata={"name": "bare"})
    l1 = skill.estimated_tokens(1)
    l2 = skill.estimated_tokens(2)
    l3 = skill.estimated_tokens(3)
    assert l1 >= 1
    assert l2 == l1 * 10
    assert l3 == l1 * 100


def test_instructions_property_lazy_loads() -> None:
    """Accessing .instructions materialises L2 via the configured loader."""
    flag = {"l2": False, "l3": False}
    skill = _make_skill(flag)
    assert skill.has_instructions is False
    payload = skill.instructions
    assert payload == {"prompt": "search"}
    assert skill.has_instructions is True
    assert flag["l2"] is True
    assert flag["l3"] is False


def test_resources_property_lazy_loads() -> None:
    """Accessing .resources materialises L3 via the configured loader."""
    flag = {"l2": False, "l3": False}
    skill = _make_skill(flag)
    assert skill.has_resources is False
    payload = skill.resources
    assert payload == {"files": ["a", "b"]}
    assert skill.has_resources is True
    assert flag["l3"] is True


def test_layer_advances_1_to_2_to_3() -> None:
    """layer property starts at 1 and advances 1→2→3 as L2/L3 materialise."""
    skill = _make_skill()
    assert skill.layer == 1
    _ = skill.instructions
    assert skill.layer == 2
    _ = skill.resources
    assert skill.layer == 3
