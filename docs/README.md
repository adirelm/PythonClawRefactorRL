# Docs Index — PythonClawRefactorRL

This directory holds every contract, plan, and audit log the project relies
on. Implementation code lives under `src/`; everything in here is prose +
diagrams + decision records.

## Top-Level Documents

| Doc | Purpose |
|---|---|
| [`PRD.md`](PRD.md) | Product requirements — scope, success criteria, brief §-mapping |
| [`PLAN.md`](PLAN.md) | Phased build plan (Phase 0 bootstrap → Phase N submission) |
| [`TODO.md`](TODO.md) | Live task list, organised by phase |
| [`TRACE.md`](TRACE.md) | ex04 brief § → deliverable trace matrix |
| [`QUALITY.md`](QUALITY.md) | Quality gates, self-audit log, ruff/coverage history |
| [`THEORY.md`](THEORY.md) | §2.4 essay — 2 500–3 000 words, 4 sections, 8–12 citations |

## Subdirectories

| Path | Contents |
|---|---|
| [`prd/`](prd/) | Per-section PRD addenda when the top-level PRD grows too long |
| [`adr/`](adr/) | Architecture Decision Records — ADR-001 PythonClaw shim, ADR-002 GraphifyAdapter, ADR-003 tiktoken cost metric, etc. |
| [`diagrams/`](diagrams/) | Mermaid / draw.io source + exported PNG/SVG |
| [`shared/`](shared/) | Cross-cutting artefacts (PROMPTS.md — literal prompts used, per §1.4 evidence) |
| [`assets/`](assets/) | Static images / PDFs referenced from shipped docs (all other PDFs are gitignored) |

## Reading Order for a New Contributor

1. `PRD.md` — what we're building and why.
2. `PLAN.md` — how the work is sliced into phases.
3. `adr/` — every non-obvious architecture choice and its rejected
   alternatives.
4. `THEORY.md` — the algorithmic + graph-theoretic backing.
5. `TRACE.md` — proof that every ex04 brief § is covered.

## Update Discipline

- Edit the PRD/PLAN **before** writing code that changes the contract
  (CLAUDE.md §1.4 — architect signs off first).
- ADRs are append-only; supersede, don't rewrite.
- Update `TRACE.md` in the same commit that adds the deliverable it traces.
