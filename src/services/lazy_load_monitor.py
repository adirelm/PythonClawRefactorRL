"""Lazy-load monitor (ADR-005 + PRD-SKILLS §4).

Guards the hard invariant from CLAUDE.md and the brief:
    accessing ``skill.metadata`` MUST NOT load L2 (instructions) or L3
    (resources). If it does, the system emits the NEGATIVE ``P_skills``
    penalty (= -5.0 per ``config/config.yaml#reward.p_skills``) which the
    downstream reward service folds into R_t.

This module owns ONLY detection + event emission. It does NOT compute the
reward — the reward service consumes ``LazyLoadEvent`` instances and
applies ``P_skills`` per occurrence.

Two checks are exposed:
    * ``check_metadata_access(skill_id)`` — snapshots ``sys.modules`` and
      the per-skill L2/L3 load flags around a metadata-only access; the
      check passes only if no new module was imported AND neither
      ``has_instructions`` nor ``has_resources`` flipped to True.
    * ``check_token_count(skill_id, layer, actual_tokens)`` — fails if
      the observed token count blows past the p95 budget, which signals
      a broken lazy-load (entire L2/L3 payload streamed when only L1
      should have been touched).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from src.pythonclaw_shim.registry import SkillRegistry

_BROKEN_METADATA = "metadata_access_loaded_l2_or_l3"
_BROKEN_TOKEN = "token_count_over_p95"


@dataclass
class LazyLoadEvent:
    """One detected lazy-load breach.

    Attributes:
        skill_id: Skill name as known to ``SkillRegistry``.
        layer: 1 (metadata), 2 (instructions), or 3 (resources).
        broken_check_name: Stable identifier for the failing check.
        actual: The observed value (e.g. token count, or extra-modules count).
        threshold: The configured limit that was breached.
    """

    skill_id: str
    layer: int
    broken_check_name: str
    actual: int
    threshold: int


@dataclass
class LazyLoadMonitor:
    """Detects broken lazy-load behaviour around a ``SkillRegistry``.

    The monitor is intentionally side-effect-light: it observes registry
    state and records events. Downstream consumers (reward service) read
    ``self.events`` and apply ``P_skills`` per emitted event.
    """

    registry: SkillRegistry
    token_p95_threshold: int = 1000
    load_counts: dict[str, int] = field(default_factory=dict)
    events: list[LazyLoadEvent] = field(default_factory=list)

    def check_metadata_access(self, skill_id: str) -> bool:
        """Return True iff metadata-only access did not touch L2 or L3.

        Steps:
            1. Snapshot ``sys.modules`` keys.
            2. Force a metadata access via the registry.
            3. Verify no new modules were imported and that the skill's
               ``has_instructions`` / ``has_resources`` flags are still False.

        A False return value is paired with a logged ``LazyLoadEvent``
        carrying ``broken_check_name = "metadata_access_loaded_l2_or_l3"``.
        """
        skill = self.registry.get(skill_id)
        instructions_before = skill.has_instructions
        resources_before = skill.has_resources
        modules_before = set(sys.modules)

        _ = self.registry.load_metadata(skill_id)

        modules_after = set(sys.modules)
        new_modules = modules_after - modules_before
        l2_touched = skill.has_instructions and not instructions_before
        l3_touched = skill.has_resources and not resources_before
        broken = bool(new_modules) or l2_touched or l3_touched

        self.load_counts[skill_id] = self.load_counts.get(skill_id, 0) + 1

        if broken:
            self.log_event(
                LazyLoadEvent(
                    skill_id=skill_id,
                    layer=1,
                    broken_check_name=_BROKEN_METADATA,
                    actual=len(new_modules) + int(l2_touched) + int(l3_touched),
                    threshold=0,
                )
            )
            return False
        return True

    def check_token_count(self, skill_id: str, layer: int, actual_tokens: int) -> bool:
        """Return False (broken) when ``actual_tokens`` exceeds the p95 budget.

        A broken lazy-load typically materialises as an order-of-magnitude
        blow-up at L1 (because the entire L2/L3 payload streamed in). The
        p95 budget caps "reasonable" token counts; anything above signals
        breakage and is logged as a ``LazyLoadEvent``.
        """
        if actual_tokens > self.token_p95_threshold:
            self.log_event(
                LazyLoadEvent(
                    skill_id=skill_id,
                    layer=layer,
                    broken_check_name=_BROKEN_TOKEN,
                    actual=int(actual_tokens),
                    threshold=int(self.token_p95_threshold),
                )
            )
            return False
        return True

    def log_event(self, event: LazyLoadEvent) -> None:
        """Record a detected breach. Append-only; never mutates prior events."""
        self.events.append(event)
