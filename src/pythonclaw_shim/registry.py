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

from src.pythonclaw_shim import SkillNotFound
from src.pythonclaw_shim.skill import Skill

_DEFAULT_DIR = Path(__file__).parent / "sample_skills"


def _make_loader(path: Path) -> Callable[[], dict]:
    """Return a zero-arg loader that reads ``path`` as JSON.

    Per PRD-SKILLS A5 the shim no longer fabricates a placeholder when the
    on-disk L2/L3 payload is missing — silent stubs were masking missing
    fixtures in upstream callers. The loader now raises
    :class:`FileNotFoundError` with the offending path on first access.
    """

    def _load() -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"skill payload missing: {path} (expected L2/L3 JSON fixture alongside the L1 metadata)"
            )
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    return _load


class SkillRegistry:
    """Discover + look up Skills via JSON fixtures in ``skills_dir``.

    Implements PRD-SKILLS §3 surface (``discover``, ``get``, ``__len__``)
    plus the monitor-side seam (``load_metadata``, ``load_instructions``,
    ``load_resources``) per ADR-011 §2.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir: Path = Path(skills_dir) if skills_dir else _DEFAULT_DIR
        self._cache: dict[str, Skill] | None = None

    def discover(self, path: Path | None = None) -> list[Skill]:
        """Scan ``skills_dir`` for ``*.metadata.json`` and build Skill handles.

        When ``path`` is supplied the registry rebinds ``self.skills_dir`` to
        it and invalidates any previously cached results before scanning;
        otherwise the constructor-supplied directory is reused. Loads L1
        only. Results are cached per registry instance and sorted by
        (name, version) for deterministic order (PRD-SKILLS A4).
        """
        if path is not None:
            self.skills_dir = Path(path)
            self._cache = None

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
                ),
                resources_loader=_make_loader(
                    self.skills_dir / f"{stem}.resources.json",
                ),
            )
        self._cache = dict(sorted(built.items(), key=lambda kv: (kv[0], kv[1].version)))
        return list(self._cache.values())

    def __len__(self) -> int:
        """Return the number of discovered skills (triggers discover if needed)."""
        if self._cache is None:
            self.discover()
        assert self._cache is not None
        return len(self._cache)

    def get(self, skill_id: str) -> Skill:
        """Return the Skill handle for ``skill_id`` or raise :class:`SkillNotFound`."""
        if self._cache is None:
            self.discover()
        assert self._cache is not None
        if skill_id not in self._cache:
            available = sorted(self._cache)
            hint = ", ".join(available) if available else "<none discovered>"
            raise SkillNotFound(f"unknown skill_id: {skill_id!r} (available: {hint})")
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
