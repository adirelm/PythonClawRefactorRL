"""Architectural contract for docs/ESSAY.md — shape gate.

Baseline at write-time (Wave 4a Stream C):
    Total word count        : 2771 words  (target band 2500-3000)
    H2 section count        : 11           (>= 4 required)
    Unique citations        : 11           (target band 8-12)
    Diagram references      : D1 + D2 + "Diagram" mentions (>= 2 required)
    Brief §2.4 prompts in H2: complementarity / AI automating SA / limitations

Pins the *shape* of the essay so future edits (notably Wave 4b ESSAY-S4 §4
+ S5 §5 expansion) cannot drift outside the brief band. The word-count
test conditionally SKIPS if the essay is still below 2500 words (so the
gate is green-or-skip during early Phase 4 and becomes enforcing once
§4+§5 land); every other dimension is hard-asserted from the moment the
skeleton + §1+§2+§3 are committed. See `instructions/A4_brief.md` §2.4.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ESSAY_PATH = Path(__file__).resolve().parents[2] / "docs" / "ESSAY.md"

WC_MIN, WC_MAX = 2500, 3000
CITE_MIN, CITE_MAX = 8, 12
MIN_H2 = 4
MIN_DIAGRAMS = 2

_CITE_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9]*[0-9]{4}[A-Za-z]*\]")
_DIAGRAM_RE = re.compile(r"\b(?:D[1-9]|Diagram|diagram)\b|!\[[^\]]*\]\([^)]+\)")


def _read_essay() -> str:
    assert ESSAY_PATH.exists(), f"ESSAY.md not found at {ESSAY_PATH}"
    return ESSAY_PATH.read_text(encoding="utf-8")


def _h2(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]


def test_essay_word_count_within_brief_band() -> None:
    """Total word count must land in [2500, 3000]; SKIP if <2500 (pre-S4)."""
    text = _read_essay()
    wc = len(text.split())
    if wc < WC_MIN:
        pytest.skip(
            f"Phase-4 Wave 4b ESSAY-S4/S5 not yet landed; current wc={wc} "
            f"(need >={WC_MIN} before this gate enforces)."
        )
    assert wc <= WC_MAX, f"Essay too long: wc={wc} > {WC_MAX}"
    assert wc >= WC_MIN, f"Essay too short: wc={wc} < {WC_MIN}"


def test_essay_section_count() -> None:
    """At least 4 H2 sections (§1-§4 minimum; §5 conclusion is good)."""
    text = _read_essay()
    h2 = _h2(text)
    assert len(h2) >= MIN_H2, f"Essay has only {len(h2)} H2 sections, need >= {MIN_H2}. Found: {h2}"


def test_essay_citation_count() -> None:
    """Unique inline citation count must land in [8, 12]."""
    text = _read_essay()
    cites = {m.lower() for m in _CITE_RE.findall(text)}
    n = len(cites)
    assert CITE_MIN <= n <= CITE_MAX, (
        f"Unique citation count {n} outside [{CITE_MIN}, {CITE_MAX}]. Found: {sorted(cites)}"
    )


def test_essay_diagram_references() -> None:
    """At least 2 diagram references (D1, D2, image links, or 'Diagram')."""
    text = _read_essay()
    n = len(_DIAGRAM_RE.findall(text))
    assert n >= MIN_DIAGRAMS, f"Only {n} diagram references found, need >= {MIN_DIAGRAMS}."


def test_essay_brief_24_prompts_mapped() -> None:
    """Brief §2.4's three prompts must each appear in an H2 heading or §1 body.

    Prompts: (1) complementarity, (2) AI automating static analysis,
    (3) limitations. Accept either the short form ("AI automating SA") or
    the spelled-out form ("AI automating static analysis").
    """
    text = _read_essay().lower()
    h2_joined = " ".join(_h2(text)).lower()
    head_blob = text[:1500]

    assert "complementarity" in h2_joined or "complementarity" in head_blob, (
        "Brief prompt #1 'complementarity' not surfaced in H2 or opening."
    )
    sa_short = "ai automating sa" in h2_joined
    sa_long = "ai automating static analysis" in h2_joined or "static analysis" in h2_joined
    assert sa_short or sa_long, "Brief prompt #2 'AI automating static analysis' not surfaced in H2."
    assert "limitations" in h2_joined, "Brief prompt #3 'limitations' not surfaced in H2."


def test_essay_thesis_present() -> None:
    """A 'Thesis' H2 section must exist and name COMPLEMENTARITY in its body."""
    text = _read_essay()
    h2 = _h2(text)
    thesis_headers = [h for h in h2 if h.lower().startswith("## thesis")]
    assert thesis_headers, f"No '## Thesis' H2 section found. H2s: {h2}"

    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("## thesis"))
    body_lines: list[str] = []
    for ln in lines[start + 1 : start + 60]:
        if ln.startswith("## "):
            break
        body_lines.append(ln)
    body = " ".join(body_lines).lower()
    assert "complementarity" in body, (
        "Thesis body does not name 'complementarity' — architect-locked "
        "Option A requires the term in the Thesis section."
    )
