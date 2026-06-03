# ADR-009: Dual-Track Screenshot Pipeline (Programmatic + Manual)

- **Status:** Accepted
- **Date:** 2026-06-04
- **Deciders:** Architect (human), implementer (AI)
- **Supersedes:** —
- **Related:** ADR-002 (GraphifyAdapter), OQ-8

## Context

Open Question OQ-8 surfaced during planning: §2.4 of the assignment
requires visual evidence of the dependency graph and of the Obsidian
"before/after" reorganisation. Prior assignments (A1, A3) burned 1–2
days each on screenshot wrangling because the pipeline was invented at
submission time. Two failure modes recurred:

1. **Non-determinism.** Matplotlib / pyvis layouts shift between runs
   without a fixed seed; chart text reflows under different DPI; the
   reviewer sees a different image than the one cited in the report.
2. **Manual-only capture.** Obsidian's Graph View is a live, animated
   force layout — there is no headless export. Screenshots taken by
   hand the night before submission look rushed and inconsistent.

Picking one track and not the other costs us either reproducibility
(manual-only) or narrative quality (programmatic-only — Obsidian's
folder-coloured hairball is the §2.4 hero image and cannot be
reproduced from NetworkX alone).

## Decision

**Dual track, both built in Phase 0, both wired into CI where
applicable.**

1. **Programmatic track — `src/services/graph_renderer.py`.**
   NetworkX + pyvis renders driven by explicit seeds. Same seed →
   same image, byte-stable on Linux/macOS. CI runs the renderer on
   every push and asserts the output PNG hashes match a checked-in
   manifest. Output path: `results/figures/graph_seed_N.png` (one
   per seed in the configured set, default `N ∈ {0,1,2,3,4}`).

2. **Manual track — Obsidian Graph View hero shots.** Two captures,
   taken once, committed as binary assets: the pre-refactor
   "spaghetti" view and the post-refactor "cluster" view. Output
   paths: `results/figures/obsidian_before.png` and
   `results/figures/obsidian_after.png`. The capture procedure is
   documented in `docs/screenshot_capture.md` so it is repeatable
   even though it is not automated.

Every figure in the report and in `README.md` carries a **one-line
caption** that names which track produced it, e.g.
*"Fig. 3 — Skills module dependency graph (programmatic,
seed=2)"* or *"Fig. 1 — Repository graph before refactor
(Obsidian Graph View, manual capture 2026-06-04)"*.

## Justification

- **Lesson from A1/A3.** Both assignments lost 1–2 working days to
  last-minute screenshot debugging. Building the pipeline in Phase 0
  amortises that cost across the whole timeline rather than
  concentrating it at the deadline.
- **Reproducibility where it matters.** The graphs that back
  quantitative claims (centrality, modularity, cluster counts) are
  programmatic and CI-verified — a grader can re-run the seed and
  obtain pixel-identical output.
- **Narrative quality where it matters.** Obsidian's hand-tuned
  colouring and force layout communicate the refactor story better
  than any NetworkX render; we keep that channel open by accepting
  the manual capture cost for exactly two images.
- **Honest attribution.** Per-image captions prevent the reviewer
  from mistaking a manual screenshot for a reproducible artefact or
  vice versa.

## Consequences

- `src/services/graph_renderer.py` is a ≤150-LOC module owned by the
  graphify subsystem; it imports only `networkx`, `pyvis`,
  `matplotlib`, and stdlib.
- `tests/test_graph_renderer.py` asserts deterministic PNG hashes
  for the seed set and runs in CI.
- `results/figures/` is committed (binary), but
  `results/figures/*.tmp.png` is gitignored to keep iterative work
  out of the index.
- `docs/screenshot_capture.md` documents the Obsidian capture
  procedure (zoom level, viewport, colour scheme, OS dark mode off).

## Alternatives Considered

- **Programmatic only.** Rejected: loses the §2.4 hero shot quality
  that distinguishes this submission from a generic NetworkX report.
- **Manual only.** Rejected: repeats the A1/A3 failure mode and
  makes every quantitative figure un-reproducible.
- **Mermaid / Graphviz dot.** Considered for the programmatic track;
  rejected because pyvis already gives interactive HTML for free and
  NetworkX + matplotlib gives static PNG with a single API.
