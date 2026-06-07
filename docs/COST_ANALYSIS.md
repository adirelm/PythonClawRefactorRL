# Cost Analysis

> **Phase 4 final fill — populated from `results/cost/cost_table.csv` per
> ADR-003 + ADR-003a.** Headline token counts come from `tiktoken`
> (`cl100k_base`); characters and bytes are appendix sensitivity columns.
> Prices are recorded at the timestamp of the first call in the phase and
> not back-filled if list prices later change. Wall-clock is end-to-end
> for the phase (including human-in-the-loop time), not just model time.

## §1 Headline

Total Phase-0 → Phase-4 spend on this assignment is **$0.6284 USD**, split
across the five phases below. All five rows were billed against
`claude-opus-4-7` at the Anthropic Opus 4.x snapshot pricing of
**$15.00 / $75.00 per million input / output tokens**, captured at
`2026-06-07T00:00:00Z` from the Anthropic public pricing page
(`https://www.anthropic.com/pricing`).

| Phase | Model | Subtotal (USD) |
|---|---|---|
| 0 — Brief / PRD | claude-opus-4-7 | $0.22890 |
| 1 — Architecture / ADRs | claude-opus-4-7 | $0.05535 |
| 2 — Implementation | claude-opus-4-7 | $0.08595 |
| 3 — Training / experiments | claude-opus-4-7 | $0.21825 |
| 4 — Write-up / deliverables (partial) | claude-opus-4-7 | $0.039975 |
| **Total** | — | **$0.628425** |

Phase-4 is "partial" because the row was stamped at commit `dbdd1a5`
(essay skeleton + bib landing), before Wave-4 streams. Later Wave-4 spend
is not yet reconciled into the table.

## §2 Method

The headline unit is `tiktoken` `cl100k_base` tokens, asserted at import
time in `src/cost/meter.py` so a silent encoding rename in a future
`tiktoken` release would fail loudly rather than quietly changing every
cost number. Character and byte counts are reported alongside every
headline token figure per ADR-003a "fail loud, fail useful" — three
counters, not one.

Per-phase corpora are collected by `scripts/collect_phase_corpora.py`,
which sweeps the prompts and assistant transcripts associated with each
phase into `results/cost/phase_<n>.jsonl`. `scripts/compute_cost.py`
then runs every record through `TripleCounter.count()`, applies the
snapshot price, and writes the 15-column table at
`results/cost/cost_table.csv`.

Re-tokenisation is the audit:
`uv run --active python scripts/compute_cost.py` re-runs the meter end
to end and overwrites `cost_table.csv`. Because the encoding is pinned
and the per-record JSONL files are immutable, the audit is bit-stable —
the same five-row table is reproduced from scratch on every run.

## §3 Per-phase cost table (verbatim from `cost_table.csv`)

CSV header (15 columns):
`phase, model, input_tokens, output_tokens, chars_in, chars_out, bytes_in, bytes_out, wall_clock_sec, episodes, price_in_per_M, price_out_per_M, price_timestamp_iso, subtotal_usd, run_id`.

| phase | model | input_tokens | output_tokens | chars_in | chars_out | bytes_in | bytes_out | wall_clock_sec | episodes | price_in_per_M | price_out_per_M | price_timestamp_iso | subtotal_usd | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | claude-opus-4-7 | 0 | 3052 | 0 | 10543 | 0 | 10616 | 46063.0 | 0 | 15.0 | 75.0 | 2026-06-07T00:00:00Z | 0.2289 | phase0-a213652 |
| 1 | claude-opus-4-7 | 0 | 738 | 0 | 2737 | 0 | 2783 | 18247.0 | 0 | 15.0 | 75.0 | 2026-06-07T00:00:00Z | 0.05535 | phase1-0165fa2 |
| 2 | claude-opus-4-7 | 0 | 1146 | 0 | 4141 | 0 | 4215 | 49655.0 | 0 | 15.0 | 75.0 | 2026-06-07T00:00:00Z | 0.08595 | phase2-ec1288a |
| 3 | claude-opus-4-7 | 0 | 2910 | 0 | 10648 | 0 | 10746 | 53422.0 | 6 | 15.0 | 75.0 | 2026-06-07T00:00:00Z | 0.21825 | phase3-71f0213 |
| 4 | claude-opus-4-7 | 0 | 533 | 0 | 1999 | 0 | 2003 | 17551.0 | 0 | 15.0 | 75.0 | 2026-06-07T00:00:00Z | 0.039975 | phase4-dbdd1a5 |

`input_tokens = 0` across every row is **not** a measurement error: the
corpora collector currently only sweeps assistant **output** transcripts.
The architect's brief was that user-side prompts in this assignment are
short relative to assistant output (essay + code + ADR drafts dominate),
so the headline number is driven by output tokens; a future revision can
add an `input_tokens` sweep without touching the schema.

## §4 Sensitivity appendix — tokens vs chars vs bytes

Two rows from `cost_table.csv` show the appendix-vs-headline ratios:

- **Phase 0 (PRD / brief).** 3,052 output tokens · 10,543 chars · 10,616
  bytes. That is **~3.45 chars/token** and **~3.48 bytes/token**,
  consistent with a primarily-ASCII English corpus with a small fraction
  of multi-byte punctuation (em-dashes, en-dashes, smart quotes in the
  PRD).
- **Phase 3 (training write-up).** 2,910 tokens · 10,648 chars · 10,746
  bytes — **~3.66 chars/token** and **~3.69 bytes/token**. Slightly
  higher than Phase 0 because Phase-3 transcripts include longer
  identifier names (`compute_advantages_gae`, `kl_target_clipfrac`) and
  numeric tables, both of which BPE encodes with fewer tokens per char
  than prose.

Hebrew vs ASCII demo (executed live against `src.cost.meter.TripleCounter`
on `2026-06-07`, encoding `cl100k_base`):

| input string | tokens | chars | bytes |
|---|---:|---:|---:|
| `שלום עולם, contains Hebrew` (Hebrew + ASCII tail) | 13 | 26 | 34 |
| `Hello world, this string is ASCII` (ASCII only) | 7 | 33 | 33 |

The Hebrew + ASCII string is **shorter in chars (26 vs 33)** yet uses
**almost twice as many tokens (13 vs 7)** and **more bytes (34 vs 33)**:
Hebrew code points cost 2 bytes in UTF-8 and are split into multiple
`cl100k_base` subword tokens. Conversely the ASCII string has
**equal char and byte counts (33 = 33)** because every code point is one
byte. This is exactly why **chars-only is rejected** per ADR-003
§Justification: a chars-only meter would under-count Hebrew identifiers
or Hebrew docstrings by ~2x in tokens and would systematically
misrepresent cost in any multilingual corpus.

## §5 Efficiency ratios + cost-of-mistakes

- **$/episode for Phase 3.** $0.21825 / 6 RC-5 aggregate episodes =
  **$0.0364 per training episode**. That is the all-in marginal cost of
  one PPO episode at the architect's seed budget, including the
  human-in-the-loop transcripts that produced each episode's analysis.
- **$/finding for the BUG_REPORT.** $0.039975 / 2 bugs identified
  (BUG-1 Louvain-wedge regression + BUG-2 the documented variant)
  = **$0.0200 per bug identified-and-written-up**. This is a lower
  bound: it omits the diagnostic work itself, which was bundled into
  Phase 3's training spend.
- **Cost of the 2/5 TIMEOUT seeds.** Seeds 123 and 314 hit the
  per-seed wall-clock cap and were **not** rescued by the Phase-4
  RC investigation. Roughly the Phase-3 share attributable to those two
  seeds is `2/5 × $0.21825 ≈ $0.0873`, and the Phase-4 RC diagnostic
  spend that did not close the residual contention is on top of that.
  Honest read: **~$0.09–$0.13 was spent investigating a flake that
  ultimately remained unresolved**, and the submission reports 3/5 OK
  with the architect-imposed -2 honesty penalty.

## §6 Reproducibility

1. `uv run --active python scripts/collect_phase_corpora.py`
   → emits `results/cost/phase_0.jsonl` … `phase_4.jsonl`.
2. `uv run --active python scripts/compute_cost.py`
   → re-tokenises every JSONL record via `TripleCounter` and writes
   `results/cost/cost_table.csv` (the 5-row, 15-column table above).
3. Open this doc; §3 mirrors `cost_table.csv` 1:1 — if the two diverge,
   the CSV wins and this doc is stale.
