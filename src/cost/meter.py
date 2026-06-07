"""Phase-4 token + char + byte triple-counter (ADR-003 + ADR-003a).

Headline cost units are tiktoken ``cl100k_base`` tokens; chars and bytes are
appendix metrics per ADR-003. The encoding name is asserted at import time so
silent drift to a different encoding is impossible: a future ``tiktoken``
release that renames the vocabulary would fail loudly here, rather than
quietly changing every cost number in the report.

Public surface (kept tiny on purpose):

* ``Counts`` — frozen dataclass of three ints (tokens, chars, bytes).
* ``TripleCounter`` — single ``count(text)`` method returning ``Counts``,
  plus an ``encoding_name`` property for audit.

The module degrades gracefully when ``tiktoken`` is not installed: import
still succeeds (so ``src.cost`` can be imported in environments that haven't
synced the optional dep yet), but any call to ``count()`` raises
``RuntimeError`` with an actionable message. This matches the ADR-003a
amendment's "fail loud, fail useful" stance.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - import-time guard, exercised by both branches in CI
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
    assert _ENCODING.name == "cl100k_base", (
        f"tiktoken encoding drift detected: got {_ENCODING.name!r}, expected 'cl100k_base' (ADR-003)"
    )
    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without tiktoken
    _ENCODING = None
    _TIKTOKEN_AVAILABLE = False


@dataclass(frozen=True)
class Counts:
    """Triple-count result for a single string.

    Attributes mirror the headline / appendix split in ADR-003:

    * ``tokens`` — tiktoken ``cl100k_base`` encoded length (headline metric).
    * ``chars`` — ``len(text)`` (grader sanity-check column).
    * ``bytes`` — ``len(text.encode('utf-8'))`` (raw I/O profiling column).
    """

    tokens: int
    chars: int
    bytes: int


class TripleCounter:
    """Counts tokens (cl100k_base) + chars + bytes for a text string.

    A single instance is cheap to construct and safe to reuse across phases:
    the underlying tiktoken encoding is loaded once at module import time and
    reused for every ``count()`` call, so amortised cost is one ``.encode()``
    plus two ``len()`` calls per measurement.
    """

    def count(self, text: str) -> Counts:
        """Return the triple-count for ``text``.

        Raises
        ------
        RuntimeError
            If ``tiktoken`` is not importable in the current environment.
            Install via ``uv add tiktoken`` (already pinned in
            ``pyproject.toml`` per ADR-003).
        """
        if not _TIKTOKEN_AVAILABLE:
            raise RuntimeError(
                "tiktoken not available; install via 'uv add tiktoken' (pinned in pyproject.toml per ADR-003)"
            )
        tokens = len(_ENCODING.encode(text))
        chars = len(text)
        bytes_ = len(text.encode("utf-8"))
        return Counts(tokens=tokens, chars=chars, bytes=bytes_)

    @property
    def encoding_name(self) -> str:
        """Return the tiktoken encoding name, or ``'unavailable'`` if absent.

        Exposed so cost-report writers can stamp the encoding identifier into
        the ``cost_table.csv`` header without re-importing tiktoken.
        """
        if not _TIKTOKEN_AVAILABLE:
            return "unavailable"
        return _ENCODING.name
