"""Unit tests for ``src/pythonclaw_shim/registry.py`` (ADR-011 §2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pythonclaw_shim.registry import SkillRegistry
from src.pythonclaw_shim.skill import Skill


def _write_skill(
    skills_dir: Path,
    name: str,
    *,
    instructions: dict | None = None,
    resources: dict | None = None,
) -> None:
    """Drop a ``{name}.metadata.json`` (+ optional L2/L3) into ``skills_dir``."""
    metadata = {
        "name": name,
        "version": "1.0.0",
        "description": f"Test skill {name}.",
        "estimated_tokens": {"L1": 50, "L2": 500, "L3": 5000},
        "tags": ["test"],
        "depends_on": [],
    }
    (skills_dir / f"{name}.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if instructions is not None:
        (skills_dir / f"{name}.instructions.json").write_text(json.dumps(instructions), encoding="utf-8")
    if resources is not None:
        (skills_dir / f"{name}.resources.json").write_text(json.dumps(resources), encoding="utf-8")


@pytest.fixture
def sample_registry(tmp_path: Path) -> SkillRegistry:
    """Registry pointed at a tmp dir with 3 deterministic skills."""
    _write_skill(
        tmp_path,
        "example_skill_1",
        instructions={"steps": ["a", "b"]},
        resources={"files": ["x.py"]},
    )
    _write_skill(tmp_path, "example_skill_2")
    _write_skill(tmp_path, "example_skill_3")
    return SkillRegistry(skills_dir=tmp_path)


def test_discover_returns_at_least_3_skills(sample_registry: SkillRegistry) -> None:
    """``discover()`` finds every ``*.metadata.json`` in the skills dir."""
    skills = sample_registry.discover()
    assert len(skills) >= 3
    assert all(isinstance(s, Skill) for s in skills)
    assert {s.name for s in skills} >= {
        "example_skill_1",
        "example_skill_2",
        "example_skill_3",
    }


def test_get_returns_skill_by_name(sample_registry: SkillRegistry) -> None:
    """``get(name)`` returns the Skill with matching ``.name``."""
    skill = sample_registry.get("example_skill_1")
    assert isinstance(skill, Skill)
    assert skill.name == "example_skill_1"
    assert skill.version == "1.0.0"
    assert skill.metadata["description"].startswith("Test skill")


def test_get_missing_raises_keyerror(sample_registry: SkillRegistry) -> None:
    """Unknown skill id surfaces as a plain KeyError."""
    with pytest.raises(KeyError):
        sample_registry.get("does_not_exist")


def test_lazy_load_invariant(sample_registry: SkillRegistry) -> None:
    """``discover()`` must not touch L2 or L3 on any returned Skill."""
    skills = sample_registry.discover()
    assert all(s.has_instructions is False for s in skills)
    assert all(s.has_resources is False for s in skills)

    # Reading metadata / estimated_tokens stays L1-only.
    s = sample_registry.get("example_skill_1")
    _ = s.metadata
    _ = s.estimated_tokens(2)
    assert s.has_instructions is False
    assert s.has_resources is False


def test_load_instructions_returns_dict(sample_registry: SkillRegistry) -> None:
    """``load_instructions`` returns a non-empty dict and flips ``has_instructions``."""
    payload = sample_registry.load_instructions("example_skill_1")
    assert isinstance(payload, dict)
    assert payload  # non-empty
    skill = sample_registry.get("example_skill_1")
    assert skill.has_instructions is True


def test_load_resources_returns_dict(sample_registry: SkillRegistry) -> None:
    """``load_resources`` returns a non-empty dict and flips ``has_resources``."""
    payload = sample_registry.load_resources("example_skill_1")
    assert isinstance(payload, dict)
    assert payload  # non-empty
    skill = sample_registry.get("example_skill_1")
    assert skill.has_resources is True


def test_load_metadata_matches_skill_metadata(sample_registry: SkillRegistry) -> None:
    """``load_metadata(name)`` returns the same dict as ``Skill.metadata``."""
    assert sample_registry.load_metadata("example_skill_1") is sample_registry.get("example_skill_1").metadata


def test_default_skills_dir_discovers_bundled_samples() -> None:
    """With no ``skills_dir``, the registry points at the bundled samples."""
    registry = SkillRegistry()
    skills = registry.discover()
    assert len(skills) >= 1
    assert all(s.has_instructions is False for s in skills)
    assert all(s.has_resources is False for s in skills)


def test_discover_results_cached(sample_registry: SkillRegistry) -> None:
    """``discover()`` is memoised per registry instance."""
    first = sample_registry.discover()
    second = sample_registry.discover()
    assert [s.name for s in first] == [s.name for s in second]
