# Cost Analysis

> **Phase 4 fill — populated from per-phase token accounting per ADR-003.**
> Token counts are produced with `tiktoken` using the `cl100k_base`
> encoding (ADR-003). Prices are recorded at the time the phase was run
> and not back-filled if list prices change. Wall-clock is end-to-end
> for the phase (including human-in-the-loop time), not just model time.

## Schema (ADR-003)

| Phase | Model | Input tokens (tiktoken cl100k_base) | Output tokens | $/M input | $/M output | Wall-clock | Subtotal |
|---|---|---|---|---|---|---|---|
| 0 — Brief / PRD | <model> | <in> | <out> | <price_in> | <price_out> | <hh:mm> | $<subtotal> |
| 1 — Architecture / ADRs | <model> | <in> | <out> | <price_in> | <price_out> | <hh:mm> | $<subtotal> |
| 2 — Implementation | <model> | <in> | <out> | <price_in> | <price_out> | <hh:mm> | $<subtotal> |
| 3 — Training / experiments | <model> | <in> | <out> | <price_in> | <price_out> | <hh:mm> | $<subtotal> |
| 4 — Write-up / deliverables | <model> | <in> | <out> | <price_in> | <price_out> | <hh:mm> | $<subtotal> |
| **Total** | — | <Σin> | <Σout> | — | — | <Σ hh:mm> | **$<Σ subtotal>** |

## Method

1. Every prompt/response pair is logged to `results/cost/phase_<n>.jsonl`.
2. A post-phase script re-tokenises every record with
   `tiktoken.get_encoding("cl100k_base")` (ADR-003) and sums in/out
   columns per model — never trusting provider-side counters.
3. Prices come from the provider's published list at the timestamp of
   the first call in the phase, recorded next to the row.
4. Wall-clock is `last_call.end − first_call.start` of the phase, so
   includes human review time inside the phase window.

## Notes

- Subtotal = `(input_tokens · $/M input + output_tokens · $/M output) / 1e6`.
- No model swaps mid-phase; if the model is changed, the phase is split
  into two rows.
- The table is the source of truth for the cost claim in the final
  write-up; any number that appears elsewhere is derived from here.
