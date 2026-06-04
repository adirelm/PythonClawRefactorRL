"""PythonClaw shim behind ADR-001 (24 h swap window once upstream lands).

Public surface:

* :class:`SkillNotFound` — raised by :meth:`SkillRegistry.get` when the
  requested skill id is unknown. Subclasses ``KeyError`` so callers that
  catch ``KeyError`` (and the legacy contract test) keep working.
"""

from __future__ import annotations


class SkillNotFound(KeyError):  # noqa: N818 — canonical name per PRD-SKILLS A5 / ADR-011
    """Raised when :meth:`SkillRegistry.get` cannot resolve a skill id.

    Subclassing :class:`KeyError` preserves the historical contract
    (``raises KeyError``) while giving callers a more specific type to
    catch when they need to distinguish skill-lookup misses from other
    dict-style ``KeyError``\\ s.
    """


__all__ = ["SkillNotFound"]
