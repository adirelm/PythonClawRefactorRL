# ADR-003a: cost_table.csv 15-Column Schema Amendment

- **Status:** Accepted
- **Date:** 2026-06-07
- **Deciders:** Architect (human), implementer (AI)
- **Supersedes:** —
- **Amends:** ADR-003 (tiktoken cl100k_base as headline cost metric)
- **Related:** ADR-003 §Cost Table Schema, brief §2.4 ("tokens entering the
  Skills module"), master PRD deliverable D8 (`docs/COST_ANALYSIS.md`),
  Assignment-1 over-confidence retrospective (auditability gap).

## Context

ADR-003 locked tiktoken `cl100k_base` as the headline cost metric and
proposed a **7-column** cost table (`phase | tokens | chars | bytes |
wall_clock_sec | episodes | model`). Wiring up `docs/COST_ANALYSIS.md`
exposed three gaps:

1. **No input/output split.** LLM pricing pages bill input and output
   tokens at different rates (often 3–5× apart). A single `tokens`
   column hides that asymmetry and forces every downstream USD math
   to re-derive the split — defeating the auditability goal.
2. **No price provenance.** A row that says "subtotal = $0.42" is
   ungradable without the per-token price *and the timestamp at which
   that price was quoted*. Pricing pages change; a row without a
   price snapshot is a number, not evidence.
3. **No run identifier.** Multi-seed sweeps (5 seeds × N phases)
   produce dozens of rows that all share `phase=train, model=gpt-4o`.
   Without a `run_id`, the rows cannot be joined back to the
   `results/runs/<run_id>/` artefact directory.

ADR-003 also asserted "appendix metrics" (chars, bytes) without
specifying whether they apply to input, output, or both — another
ambiguity that surfaced only when filling the table.

The Assignment-1 retrospective flagged auditability as the single
biggest over-confidence failure mode: "the table looked complete but
a grader couldn't verify a row without re-running the experiment." This
amendment closes that gap before any cost numbers land.

## Decision

Lock the cost table at **15 columns**, in this exact order:

| # | column | type | unit / format |
|---|---|---|---|
| 1 | `phase` | str | `bootstrap` \| `graphify` \| `skills` \| `train` \| `eval` \| `report` |
| 2 | `model` | str | LLM identifier (`gpt-4o`, `claude-opus-4`, `none`) |
| 3 | `input_tokens` | int | tiktoken cl100k_base length of prompt-side text |
| 4 | `output_tokens` | int | tiktoken cl100k_base length of completion-side text |
| 5 | `chars_in` | int | `len(prompt_text)` |
| 6 | `chars_out` | int | `len(completion_text)` |
| 7 | `bytes_in` | int | `len(prompt_text.encode("utf-8"))` |
| 8 | `bytes_out` | int | `len(completion_text.encode("utf-8"))` |
| 9 | `wall_clock_sec` | float | monotonic clock delta for the phase |
| 10 | `episodes` | int | RL episodes completed (0 for non-training phases) |
| 11 | `price_in_per_M` | float | USD per 1M input tokens at quote time |
| 12 | `price_out_per_M` | float | USD per 1M output tokens at quote time |
| 13 | `price_timestamp_iso` | str | ISO-8601 UTC when prices were sourced |
| 14 | `subtotal_usd` | float | `(input_tokens·price_in + output_tokens·price_out)/1e6` |
| 15 | `run_id` | str | matches `results/runs/<run_id>/` directory |

`src/cost/meter.py` returns the `Counts(tokens, chars, bytes)` triple per
text string; the cost-report writer calls it twice per row — once for the
prompt, once for the completion — and emits columns 3–8 from those two
`Counts` objects.

## Justification

- **Brief §2.4 compliance.** "Tokens entering the Skills module" is
  literally `input_tokens` for `phase=skills`; the brief wording maps
  to a column rather than a footnote.
- **ADR-003 §Method preservation.** Headline unit is still tiktoken
  `cl100k_base`; chars and bytes are still appendix. The amendment
  only splits each of the three units along the prompt/completion axis
  and adds the pricing provenance triple (cols 11–13).
- **A1 over-confidence lesson.** A grader can now re-derive
  `subtotal_usd` from cols 3, 4, 11, 12 without trusting the writer's
  arithmetic. `price_timestamp_iso` makes the row reproducible months
  later even after pricing changes.
- **Run-level joinability.** `run_id` (col 15) is the foreign key into
  `results/runs/<run_id>/{config.yaml, metrics.json, ...}`, so any
  cost row can be audited against the artefact that produced it.

## Consequences

- `docs/COST_ANALYSIS.md` (D8) emits exactly these 15 columns, in this
  order, as CSV and as a Markdown table. Out-of-order columns are a
  build break.
- The 7-column table in ADR-003 §Cost Table Schema is **deprecated**
  but not deleted — that section still documents the *units* (tiktoken
  cl100k_base, len, utf-8 bytes), which the 15-column schema inherits
  verbatim.
- `src/cost/meter.py` does **not** know about prices or `run_id` — it
  is a pure text-to-counts function. Pricing and run-id stamping live
  in the cost-report writer (out of scope for this ADR).
- `tests/cost/test_meter.py` locks the meter contract; a separate
  schema test (Wave 2) will lock the 15-column CSV header.

## Alternatives Considered

- **Keep 7 columns, footnote the gaps.** Rejected: the A1 retrospective
  explicitly called out "tables that need a footnote to interpret" as
  the failure mode this amendment must prevent.
- **Add a `notes` column instead of `price_timestamp_iso`.** Rejected:
  free-text columns are unauditable; ISO-8601 is machine-checkable.
- **Drop chars/bytes now that input/output is split.** Rejected: the
  appendix metrics are still the rebuttal against "your token count is
  wrong for model X" (see ADR-003 §Justification, bullet 2).
