"""Skill class for the PythonClaw shim (ADR-011 + PRD-SKILLS §3/§4).

Three-layer object with a hard lazy-load invariant:
reading ``skill.metadata`` and ``skill.estimated_tokens(layer)`` MUST NOT
trigger loading of L2 (instructions) or L3 (resources) payloads.

L2 / L3 loaders are injected at construction (Callable[[], dict]) so the
registry decides how a layer materialises, while this module owns the
"is this layer loaded yet?" bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback path only on missing dep
    _ENCODER = None

_LAYER_L1 = 1
_LAYER_L2 = 2
_LAYER_L3 = 3
_VALID_LAYERS = (_LAYER_L1, _LAYER_L2, _LAYER_L3)
_LAYER_HINT_KEY = {_LAYER_L1: "L1", _LAYER_L2: "L2", _LAYER_L3: "L3"}
_LAYER_FALLBACK_MULTIPLIER = {_LAYER_L1: 1, _LAYER_L2: 10, _LAYER_L3: 100}


def _count_tokens(payload: object) -> int:
    """cl100k_base token count of ``str(payload)``; fallback to len//4."""
    text = str(payload)
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)


class Skill:
    """Three-layer Skill handle. L1 eager, L2/L3 lazy via loader callbacks."""

    def __init__(
        self,
        name: str,
        version: str,
        metadata: dict,
        instructions_loader: Callable[[], dict] | None = None,
        resources_loader: Callable[[], dict] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self._metadata = dict(metadata)
        self._instructions_loader = instructions_loader
        self._resources_loader = resources_loader
        self._instructions: dict | None = None
        self._resources: dict | None = None

    @property
    def metadata(self) -> dict:
        """L1 payload (eager). Reading this MUST NOT load L2/L3."""
        return self._metadata

    @property
    def has_instructions(self) -> bool:
        """True iff L2 has been materialised."""
        return self._instructions is not None

    @property
    def has_resources(self) -> bool:
        """True iff L3 has been materialised."""
        return self._resources is not None

    @property
    def instructions(self) -> dict:
        """L2 payload (lazy). First access triggers load_instructions()."""
        return self.load_instructions()

    @property
    def resources(self) -> dict:
        """L3 payload (lazy). First access triggers load_resources()."""
        return self.load_resources()

    @property
    def layer(self) -> int:
        """Current materialised layer: 1 (L1 only) → 2 (+L2) → 3 (+L3)."""
        if self.has_resources:
            return _LAYER_L3
        if self.has_instructions:
            return _LAYER_L2
        return _LAYER_L1

    def load_instructions(self) -> dict:
        """Force-load L2 (idempotent). Raises if no loader was wired."""
        if self._instructions is None:
            if self._instructions_loader is None:
                raise RuntimeError(f"skill {self.name!r}: no instructions loader configured")
            self._instructions = self._instructions_loader()
        return self._instructions

    def load_resources(self) -> dict:
        """Force-load L3 (idempotent). Raises if no loader was wired."""
        if self._resources is None:
            if self._resources_loader is None:
                raise RuntimeError(f"skill {self.name!r}: no resources loader configured")
            self._resources = self._resources_loader()
        return self._resources

    def _loaded_for_layer(self, layer: int) -> dict | None:
        """Already-materialised payload for ``layer``, or None."""
        if layer == _LAYER_L2:
            return self._instructions
        if layer == _LAYER_L3:
            return self._resources
        return None

    def estimated_tokens(self, layer: int) -> int:
        """Rough cl100k_base token count for layer ∈ {1, 2, 3}.

        Per PRD-SKILLS §4 this MUST NOT force the queried layer to load:
        we count over the already-loaded payload if cached, else over the
        ``estimated_tokens`` hint in metadata, else a per-layer fallback
        derived from the metadata token count.
        """
        if layer not in _VALID_LAYERS:
            raise ValueError(f"layer must be one of {_VALID_LAYERS} (got {layer!r})")
        loaded = self._loaded_for_layer(layer)
        if loaded is not None:
            return _count_tokens(loaded)
        hint = self._metadata.get("estimated_tokens", {})
        hint_key = _LAYER_HINT_KEY[layer]
        if isinstance(hint, dict) and hint_key in hint:
            return int(hint[hint_key])
        if layer == _LAYER_L1:
            return _count_tokens(self._metadata)
        return _count_tokens(self._metadata) * _LAYER_FALLBACK_MULTIPLIER[layer]
