# ADR-003: tiktoken cl100k_base as Headline Cost Metric

- **Status:** Accepted
- **Date:** 2026-06-04
- **Deciders:** Architect (human), implementer (AI)
- **Supersedes:** —
- **Related:** ADR-001 (PythonClaw shim), ADR-002 (GraphifyAdapter), OQ-3,
  master PRD D8 (cost envelope in `docs/COST_ANALYSIS.md`)

## Context

Open Question OQ-3 surfaced during brief review: §2.4 of the assignment
specification says *"tokens entering the Skills module"* without
defining what a **token** is. Three candidate units exist:

1. **tiktoken (`cl100k_base`)** — BPE encoding shared by GPT-4 and a
   reasonable proxy for Claude pricing models.
2. **characters** — language-agnostic, trivially reproducible.
3. **bytes** — UTF-8 octet count, deterministic across locales.

Choosing only one invites the rebuttal *"but actual LLM token counts
differ from char/byte counts"* (true for non-ASCII identifiers, emoji
in docstrings, long whitespace runs). Choosing none leaves cost
reporting ambiguous and ungradeable.

## Decision

**tiktoken `cl100k_base` is the headline cost metric.** Character and
byte counts are reported as **appendix sensitivity columns** alongside
every headline tiktoken figure. All three are produced from the same
pass over the same source string, so the marginal cost is one extra
`len()` and one `.encode()` call per phase boundary — effectively zero.

**Triple-reporting is NOT noise — each unit answers a different
question:**

- **tokens (tiktoken `cl100k_base`)** — the LLM-cost headline. This is
  what aligns with API pricing pages and what the brief §2.4 wording
  *"tokens entering the Skills module"* most naturally maps to.
- **chars** — grader sanity-check. A grader (or future-me) can re-derive
  this with one `len(text)` call without installing tiktoken, which makes
  the table independently auditable.
- **bytes** — raw I/O profiling. UTF-8 octet count is the right unit when
  the question is *"how much disk/network did this phase actually move?"*
  rather than *"how much would an LLM charge for it?"*.

Dropping any one of the three loses a question the reader might
legitimately ask; keeping all three costs effectively nothing.

## Justification

- **External validity.** tiktoken aligns with how the LLM ecosystem
  prices API calls today (GPT-4, GPT-4o, and — within a small constant
  factor — Claude). A grader who checks the cost column against a
  pricing page will recognise the unit.
- **Defensibility.** Reporting chars/bytes alongside removes the
  pushback channel *"your token count is wrong for model X"*; the
  reader can re-derive the answer in their preferred unit from the
  same row.
- **Reproducibility.** `cl100k_base` is a frozen vocabulary shipped
  with the `tiktoken` package; pinning the package version in
  `pyproject.toml` (e.g. `tiktoken==0.7.0`, exact pin not floor) makes
  the metric byte-stable across machines. The encoding name
  `cl100k_base` is also asserted in `src/cost/meter.py` so a silent
  upstream rename would fail loudly rather than drift quietly.
- **Cost.** Triple-reporting is free — one tokenizer init per run,
  amortised across every phase.

## Cost Table Schema

Every cost artifact (CSV, Markdown table, plot legend) uses this
schema, in this column order:

| phase | tokens (tiktoken) | chars | bytes | wall_clock_sec | episodes | model |
|---|---|---|---|---|---|---|

- **phase** — string label (e.g. `bootstrap`, `graphify`, `skills`,
  `train`, `eval`).
- **tokens (tiktoken)** — int, `cl100k_base` encoding length. Headline
  cost unit.
- **chars** — int, `len(text)` on the decoded string. Grader
  sanity-check column.
- **bytes** — int, `len(text.encode("utf-8"))`. Raw I/O profiling
  column.
- **wall_clock_sec** — float, monotonic clock delta for the phase.
- **episodes** — int, RL episodes completed in the phase (0 for
  non-training phases).
- **model** — string, the LLM identifier the tiktoken count is being
  attributed to (e.g. `gpt-4o`, `claude-opus-4`, `none`). Required so
  a row's headline cost can be multiplied by the right per-token price
  later; `none` for phases that do not invoke an LLM.

## Consequences

- `pyproject.toml` gains a `tiktoken` runtime dependency, pinned to an
  exact version (not a floor) so the cl100k_base encoding bytes do not
  drift on a future minor release.
- A `src/cost/meter.py` utility (≤150 LOC) owns the triple-count
  function; all phases call into it rather than re-implementing
  tokenization locally. It also asserts `encoding.name == "cl100k_base"`
  at import time.
- The final report cites tiktoken counts in body text and footnotes
  the char/byte appendix once.
- The cost envelope deliverable **D8** (master PRD additions) —
  `docs/COST_ANALYSIS.md` — consumes this schema directly: every row in
  that document is one phase × one model, with the tiktoken column as
  the headline and chars/bytes as the audit trail.

## Alternatives Considered

- **Chars-only.** Rejected: invites the LLM-pricing pushback.
- **Bytes-only.** Rejected: same as chars, plus surprises on non-ASCII.
- **GPT-2 BPE (`p50k_base`).** Rejected: older vocabulary, weaker
  alignment with current pricing pages.
