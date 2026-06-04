"""Architectural invariants for the lazy-load monitor (ADR-005).

These tests anchor the brief's core RL signal: when metadata-only access
spills into L2/L3, the monitor MUST notice and the reward service MUST
collect the NEGATIVE ``P_skills`` penalty. Equally, when actual token
counts blow past the p95 budget, the monitor flags a broken lazy-load.

Coverage matrix (per IMPL_SCHEMA file ownership):
    * metadata access does not load L2
    * metadata access does not load L3
    * token count under p95 → passes
    * token count over p95  → fails
    * a broken check emits exactly one ``LazyLoadEvent``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pythonclaw_shim.registry import SkillRegistry
from src.services.lazy_load_monitor import LazyLoadEvent, LazyLoadMonitor


def _write_skill(skills_dir: Path, name: str) -> None:
    """Write a minimal three-layer skill fixture (only L1 is consumed by L1 tests)."""
    metadata = {
        "name": name,
        "version": "1.0.0",
        "description": f"Test skill {name}.",
        "estimated_tokens": {"L1": 50, "L2": 500, "L3": 5000},
        "tags": ["test"],
        "depends_on": [],
    }
    (skills_dir / f"{name}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (skills_dir / f"{name}.instructions.json").write_text(json.dumps({"steps": ["a", "b"]}), encoding="utf-8")
    (skills_dir / f"{name}.resources.json").write_text(json.dumps({"files": ["x.py"]}), encoding="utf-8")


@pytest.fixture
def registry(tmp_path: Path) -> SkillRegistry:
    """A registry holding one deterministic ``code_review`` skill."""
    _write_skill(tmp_path, "code_review")
    return SkillRegistry(skills_dir=tmp_path)


@pytest.fixture
def monitor(registry: SkillRegistry) -> LazyLoadMonitor:
    """Default monitor with the canonical 1000-token p95 budget."""
    return LazyLoadMonitor(registry=registry, token_p95_threshold=1000)


def test_metadata_access_does_not_load_l2(registry: SkillRegistry, monitor: LazyLoadMonitor) -> None:
    """``load_metadata`` MUST NOT materialise L2 — ADR-005 hard invariant."""
    registry.load_metadata("code_review")
    skill = registry.get("code_review")
    assert skill.has_instructions is False
    assert monitor.check_metadata_access("code_review") is True


def test_metadata_access_does_not_load_l3(registry: SkillRegistry, monitor: LazyLoadMonitor) -> None:
    """``load_metadata`` MUST NOT materialise L3 — ADR-005 hard invariant."""
    registry.load_metadata("code_review")
    skill = registry.get("code_review")
    assert skill.has_resources is False
    assert monitor.check_metadata_access("code_review") is True


def test_token_count_under_p95_passes(monitor: LazyLoadMonitor) -> None:
    """50 tokens at L1 is well under the 1000-token p95 — no event fired."""
    assert monitor.check_token_count("foo", layer=1, actual_tokens=50) is True
    assert monitor.events == []


def test_token_count_over_p95_fails(monitor: LazyLoadMonitor) -> None:
    """99999 tokens blows past p95 — check returns False AND logs an event."""
    assert monitor.check_token_count("foo", layer=1, actual_tokens=99999) is False


def test_monitor_emits_event_on_break(monitor: LazyLoadMonitor) -> None:
    """A failing check produces exactly one ``LazyLoadEvent`` with correct fields."""
    assert monitor.check_token_count("foo", layer=2, actual_tokens=99999) is False
    assert len(monitor.events) == 1
    event = monitor.events[0]
    assert isinstance(event, LazyLoadEvent)
    assert event.skill_id == "foo"
    assert event.layer == 2
    assert event.broken_check_name == "token_count_over_p95"
    assert event.actual == 99999
    assert event.threshold == 1000


def test_log_event_is_append_only(monitor: LazyLoadMonitor) -> None:
    """``log_event`` appends; it never mutates or drops prior events."""
    first = LazyLoadEvent("a", 1, "metadata_access_loaded_l2_or_l3", 1, 0)
    second = LazyLoadEvent("b", 2, "token_count_over_p95", 9999, 1000)
    monitor.log_event(first)
    monitor.log_event(second)
    assert monitor.events == [first, second]


def test_check_metadata_access_tracks_load_count(monitor: LazyLoadMonitor) -> None:
    """Per-skill load counts increment on every metadata check."""
    assert monitor.check_metadata_access("code_review") is True
    assert monitor.check_metadata_access("code_review") is True
    assert monitor.load_counts["code_review"] == 2
