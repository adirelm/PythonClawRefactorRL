"""SkillRegistry — shim-side enumeration + lazy L1/L2/L3 access (ADR-011 §2).

The shim implements PRD-SKILLS §3 verbatim against on-disk JSON fixtures:

    {skill_id}.metadata.json      (L1 — loaded at discover())
    {skill_id}.instructions.json  (L2 — loaded on demand)
    {skill_id}.resources.json     (L3 — loaded on demand)

The standout invariant (PRD-SKILLS §4 / ADR-011 §2): touching ``skill.metadata``
or calling ``registry.discover()`` MUST NOT load L2 or L3. The lazy-load
monitor relies on this — every test path here asserts it.

`SkillRegistry.load_*` methods are the monitor-safe entry points (ADR-011 §2):
they bypass the `Skill` property form when the monitor needs to force a layer
load without the property side-effect tripping the very check it is performing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from src.pythonclaw_shim.skill import Skill

_DEFAULT_DIR = Path(__file__).parent / "sample_skills"


def _make_loader(path: Path, *, fallback_key: str) -> Callable[[], dict]:
    """Return a zero-arg loader that reads `path` (or a stub if absent).

    The stub keeps the shim deterministic in tests where only L1 fixtures
    exist — the real backend (ADR-001 swap target) always ships L2/L3
    payloads alongside metadata.
    """

    def _load() -> dict:
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {fallback_key: path.stem, "shim_placeholder": True}

    return _load


class SkillRegistry:
    """Discover + look up Skills via JSON fixtures in ``skills_dir``.

    Implements PRD-SKILLS §3 surface (``discover``, ``get``) plus the
    monitor-side seam (``load_metadata``, ``load_instructions``,
    ``load_resources``) per ADR-011 §2.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir: Path = Path(skills_dir) if skills_dir else _DEFAULT_DIR
        self._cache: dict[str, Skill] | None = None

    def discover(self) -> list[Skill]:
        """Scan ``skills_dir`` for ``*.metadata.json`` and build Skill handles.

        Loads L1 only. Results are cached per registry instance and sorted
        by (name, version) for deterministic order (PRD-SKILLS A4).
        """
        if self._cache is not None:
            return list(self._cache.values())

        built: dict[str, Skill] = {}
        for meta_path in sorted(self.skills_dir.glob("*.metadata.json")):
            with meta_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            stem = meta_path.name.removesuffix(".metadata.json")
            name = metadata.get("name") or stem
            version = str(metadata.get("version", "0.0.0"))
            built[name] = Skill(
                name=name,
                version=version,
                metadata=metadata,
                instructions_loader=_make_loader(
                    self.skills_dir / f"{stem}.instructions.json",
                    fallback_key="instructions",
                ),
                resources_loader=_make_loader(
                    self.skills_dir / f"{stem}.resources.json",
                    fallback_key="resources",
                ),
            )
        self._cache = dict(sorted(built.items(), key=lambda kv: (kv[0], kv[1].version)))
        return list(self._cache.values())

    def get(self, skill_id: str) -> Skill:
        """Return the Skill handle for ``skill_id`` or raise KeyError."""
        if self._cache is None:
            self.discover()
        assert self._cache is not None
        if skill_id not in self._cache:
            raise KeyError(f"unknown skill_id: {skill_id!r}")
        return self._cache[skill_id]

    def load_metadata(self, skill_id: str) -> dict:
        """Return the L1 metadata dict. No L2/L3 side-effects."""
        return self.get(skill_id).metadata

    def load_instructions(self, skill_id: str) -> dict:
        """Force L2 load via the Skill (monitor-safe entry point)."""
        return self.get(skill_id).load_instructions()

    def load_resources(self, skill_id: str) -> dict:
        """Force L3 load via the Skill (monitor-safe entry point)."""
        return self.get(skill_id).load_resources()
