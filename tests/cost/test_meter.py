"""Unit tests for ``src.cost.meter.TripleCounter`` (Phase-4 COST-2).

Locks the ADR-003 contract: tiktoken ``cl100k_base`` headline tokens, plus
char and byte appendix metrics, computed in a single pass with a stable
result across repeated calls. Non-ASCII (Hebrew) is exercised explicitly
because that is the regime where BPE token count and char count diverge
most loudly — and that divergence is the whole reason ADR-003 reports all
three units.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tiktoken")

import src.cost.meter as meter_mod
from src.cost.meter import Counts, TripleCounter


@pytest.fixture
def counter() -> TripleCounter:
    return TripleCounter()


def test_ascii_short_string(counter: TripleCounter) -> None:
    """ASCII 'hello': cl100k_base merges to a single token; bytes == chars."""
    result = counter.count("hello")
    assert result == Counts(tokens=1, chars=5, bytes=5)


def test_ascii_two_word_string(counter: TripleCounter) -> None:
    """Two-word ASCII: tokens (2) is well below chars (11); bytes == chars."""
    result = counter.count("hello world")
    assert result.tokens == 2
    assert result.chars == 11
    assert result.bytes == 11
    assert result.bytes == result.chars  # all-ASCII invariant


def test_hebrew_string_tokens_diverge_from_chars(counter: TripleCounter) -> None:
    """Hebrew 'שלום עולם': BPE tokens exceed chars; UTF-8 bytes exceed chars.

    Confirmed empirically with tiktoken 0.13.0:
      'שלום עולם' -> tokens=10, chars=9, bytes=17
    The exact token count is part of the cl100k_base contract (pinned ADR-003);
    if it drifts the cost table drifts, so we assert the literal.
    """
    result = counter.count("שלום עולם")
    assert result.tokens == 10
    assert result.chars == 9
    assert result.bytes == 17
    assert result.tokens != result.chars  # non-ASCII -> BPE non-trivial
    assert result.bytes > result.chars  # UTF-8 multi-byte codepoints


def test_empty_string_all_zero(counter: TripleCounter) -> None:
    result = counter.count("")
    assert result == Counts(tokens=0, chars=0, bytes=0)


def test_encoding_name_is_cl100k_base(counter: TripleCounter) -> None:
    """ADR-003 pins cl100k_base; assert the public property returns it."""
    assert counter.encoding_name == "cl100k_base"


def test_repeated_calls_are_stable(counter: TripleCounter) -> None:
    """Same input twice must return byte-identical Counts (frozen dataclass)."""
    text = "def foo():\n    return 42\n"
    first = counter.count(text)
    second = counter.count(text)
    assert first == second
    assert first.tokens > 0


def test_counts_is_frozen() -> None:
    """Counts must be immutable so cost rows can't be mutated after the fact."""
    c = Counts(tokens=1, chars=2, bytes=3)
    with pytest.raises((AttributeError, TypeError)):
        c.tokens = 99  # type: ignore[misc]


def test_runtime_error_when_tiktoken_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If tiktoken is missing, count() raises RuntimeError with install hint."""
    monkeypatch.setattr(meter_mod, "_TIKTOKEN_AVAILABLE", False)
    monkeypatch.setattr(meter_mod, "_ENCODING", None)
    counter = TripleCounter()
    with pytest.raises(RuntimeError, match="tiktoken not available"):
        counter.count("hello")
    assert counter.encoding_name == "unavailable"
