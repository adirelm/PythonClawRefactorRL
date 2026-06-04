"""Unit tests for ``src/pythonclaw_shim/registry.py`` (ADR-011 §2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pythonclaw_shim import SkillNotFound
from src.pythonclaw_shim.registry import SkillRegistry
from src.pythonclaw_shim.skill import Skill


def _write_skill(
    skills_dir: Path, name: str, *, instructions: dict | None = None, resources: dict | None = None
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
    _write_skill(tmp_path, "example_skill_1", instructions={"a": 1}, resources={"b": 2})
    _write_skill(tmp_path, "example_skill_2")
    _write_skill(tmp_path, "example_skill_3")
    return SkillRegistry(skills_dir=tmp_path)


def test_discover_returns_at_least_3_skills(sample_registry: SkillRegistry) -> None:
    """``discover()`` finds every ``*.metadata.json`` in the skills dir."""
    skills = sample_registry.discover()
    assert len(skills) >= 3
    assert all(isinstance(s, Skill) for s in skills)
    assert {s.name for s in skills} >= {"example_skill_1", "example_skill_2", "example_skill_3"}


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
    assert all(s.has_instructions is False and s.has_resources is False for s in skills)
    # Reading metadata / estimated_tokens stays L1-only.
    s = sample_registry.get("example_skill_1")
    _, _ = s.metadata, s.estimated_tokens(2)
    assert s.has_instructions is False and s.has_resources is False


def test_load_instructions_returns_dict(sample_registry: SkillRegistry) -> None:
    """``load_instructions`` returns a non-empty dict and flips ``has_instructions``."""
    payload = sample_registry.load_instructions("example_skill_1")
    assert isinstance(payload, dict) and payload  # non-empty
    assert sample_registry.get("example_skill_1").has_instructions is True


def test_load_resources_returns_dict(sample_registry: SkillRegistry) -> None:
    """``load_resources`` returns a non-empty dict and flips ``has_resources``."""
    payload = sample_registry.load_resources("example_skill_1")
    assert isinstance(payload, dict) and payload  # non-empty
    assert sample_registry.get("example_skill_1").has_resources is True


def test_load_metadata_matches_skill_metadata(sample_registry: SkillRegistry) -> None:
    """``load_metadata(name)`` returns the same dict as ``Skill.metadata``."""
    assert sample_registry.load_metadata("example_skill_1") is sample_registry.get("example_skill_1").metadata


def test_default_skills_dir_discovers_bundled_samples() -> None:
    """With no ``skills_dir``, the registry points at the bundled samples."""
    skills = SkillRegistry().discover()
    assert len(skills) >= 1
    assert all(s.has_instructions is False and s.has_resources is False for s in skills)


def test_discover_results_cached(sample_registry: SkillRegistry) -> None:
    """``discover()`` is memoised per registry instance."""
    first, second = sample_registry.discover(), sample_registry.discover()
    assert [s.name for s in first] == [s.name for s in second]


def test_get_raises_skillnotfound(sample_registry: SkillRegistry) -> None:
    """Unknown skill id surfaces specifically as :class:`SkillNotFound`."""
    with pytest.raises(SkillNotFound):
        sample_registry.get("does_not_exist")


def test_skillnotfound_is_keyerror_subclass() -> None:
    """``SkillNotFound`` keeps the legacy ``raises KeyError`` contract."""
    assert issubclass(SkillNotFound, KeyError)
    assert isinstance(SkillNotFound("x"), KeyError)


def test_discover_with_path_rebinds_skills_dir(tmp_path: Path) -> None:
    """Passing ``path`` to ``discover`` rebinds ``skills_dir`` and busts cache."""
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    for d in (first_dir, second_dir):
        d.mkdir()
    _write_skill(first_dir, "only_in_first")
    _write_skill(second_dir, "only_in_second_a")
    _write_skill(second_dir, "only_in_second_b")
    registry = SkillRegistry(skills_dir=first_dir)
    assert {s.name for s in registry.discover()} == {"only_in_first"}
    rebuilt = registry.discover(path=second_dir)
    assert registry.skills_dir == second_dir
    assert {s.name for s in rebuilt} == {"only_in_second_a", "only_in_second_b"}


def test_len_returns_skill_count(sample_registry: SkillRegistry) -> None:
    """``len(registry)`` matches the discovered skill count."""
    assert len(sample_registry) == len(sample_registry.discover())
    # Triggers discovery via __len__ even when the cache starts empty.
    assert len(SkillRegistry(skills_dir=sample_registry.skills_dir)) == 3


def test_missing_instructions_raises_filenotfound(tmp_path: Path) -> None:
    """Missing L2 fixture no longer silently returns a shim placeholder."""
    _write_skill(tmp_path, "no_l2_or_l3")
    registry = SkillRegistry(skills_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load_instructions("no_l2_or_l3")
    with pytest.raises(FileNotFoundError):
        registry.load_resources("no_l2_or_l3")
