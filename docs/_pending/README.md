# docs/_pending — Phase 4 deliverable templates

These files are **Phase 4 deliverables in template form**. They are kept
in `docs/_pending/` (rather than `docs/`) so that graders reading the
main docs tree do not mistake unfilled placeholder text — literal
`<title>`, `<symptom>`, `<model>` tokens — for finished content.

## Contents

- `BUG_REPORT.md` — D-id F13 (brief §3, ≥2 architectural bugs). Will be
  populated from ADR-007 rollout traces once the 5-seed Phase-3 run is
  unblocked and produces logs to mine.
- `COST_ANALYSIS.md` — D-id D8 (brief §2.4, tiktoken cl100k_base cost
  envelope). Will be populated from `results/cost/phase_<n>.jsonl`
  per-phase logs (ADR-003 schema).

## Promotion rule

When the body of a file no longer contains any `<placeholder>` tokens
and the cross-referenced trace/cost artefacts exist, the file is moved
back to `docs/` via `git mv` and the corresponding TRACE.md row is
flipped from `<pending>` to the filling commit's SHA.

References to `docs/BUG_REPORT.md` / `docs/COST_ANALYSIS.md` elsewhere
in the docs tree (PRD, PLAN, TRACE, TODO, ADRs) intentionally use the
**post-promotion path** — they describe where the filled doc will live,
not where the template currently sits.
