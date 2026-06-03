# TRACE — Brief §-id → Artifact Map (Assignment 4)

> **Purpose.** This file is the grader's first-scan index. Every §-id in
> the ex04 brief maps to **(a)** the code file that implements it,
> **(b)** the test that verifies it, **(c)** the chart / artifact that
> evidences it, and **(d)** the commit SHA where it landed. If a row is
> ⬜ planned, the file/test paths are stubs reserved by Phase 0 bootstrap
> and will fill in during Phases 1–4.
>
> **Why it exists.** §1.4 of the submission guidelines frames the
> developer as architect and the AI as implementer. TRACE.md is the
> auditable surface of that contract: nothing ships unless it appears
> here with a real SHA and a ✅.
>
> **How to read.**
> - **Brief §** — the section id from `instructions/assignment-4/ex04_brief.md`.
> - **Requirement** — one-line restatement of what the brief asks for.
> - **Code file** — primary implementation path (≤150 LOC enforced).
> - **Test file** — RED→GREEN test that pins behaviour.
> - **Chart / artifact** — PNG / CSV / JSON / markdown the grader opens.
> - **Commit SHA** — short SHA of the commit that closed the row.
> - **Status** — ⬜ planned / 🟡 in-progress / ✅ done.
>
> Placeholder SHA `<bootstrap-commit>` is replaced post-commit by
> `scripts/stamp_trace.py` (Phase 0 utility).

---

## §1 — Refactor Track (PythonClaw → SOLID + DI)

| Brief § | Requirement | Code file | Test file | Chart / artifact | Commit SHA | Status |
|---|---|---|---|---|---|---|
| §1.1 | PythonClaw shim isolates legacy surface behind a typed boundary; swap window ≤24h. | `src/pythonclaw_shim/adapter.py` | `tests/unit/test_pythonclaw_shim.py` | `docs/adr/ADR-001-pythonclaw-shim-boundary.md` | `<bootstrap-commit>` | ⬜ planned |
| §1.2 | GRAPHIFY local re-implementation behind `GraphifyAdapter`; no external HTTP at import time. | `src/graphify/adapter.py`, `src/graphify/builder.py` | `tests/unit/test_graphify_adapter.py`, `tests/architecture/test_no_network_at_import.py` | `docs/adr/ADR-002-graphify-adapter.md`, `results/graphs/sample_graph.json` | `<bootstrap-commit>` | ⬜ planned |
| §1.3 | Cost-metric headline = tiktoken `cl100k_base`; chars/bytes appendix only. | `src/utils/token_cost.py` | `tests/unit/test_token_cost.py` | `results/cost/token_cost_table.csv`, `docs/adr/ADR-003-tiktoken-cost-metric.md` | `<bootstrap-commit>` | ⬜ planned |
| §1 — lazy-load semantics | Lazy-load broken pytest walks `sys.modules` AND enforces P95 token-cost budget. | `src/utils/lazy_load_guard.py` | `tests/architecture/test_lazy_load_broken.py` | `docs/adr/ADR-005-lazy-load-broken-semantics.md` | `<bootstrap-commit>` | ⬜ planned |
| §1 — SOLID | DI container, single-responsibility services, no god-objects. | `src/sdk/container.py`, `src/services/*.py` | `tests/architecture/test_solid_violations.py` | `docs/diagrams/solid_dep_graph.png` | `<bootstrap-commit>` | ⬜ planned |

---

## §2 — RL Track (PPO + GAE + Active Knowledge Architecture)

| Brief § | Requirement | Code file | Test file | Chart / artifact | Commit SHA | Status |
|---|---|---|---|---|---|---|
| §2.1 | Environment exposes Gymnasium-compatible `step`/`reset` over the refactored graph state. | `src/env/graph_env.py` | `tests/unit/test_graph_env_contract.py` | `docs/diagrams/env_state_diagram.png` | `<bootstrap-commit>` | ⬜ planned |
| §2.2 | PPO + GAE(λ) with proper advantage normalisation; SB3 buffer spike timeboxed 2h, padding/masking fallback gated by ADR-008. | `src/model/ppo_policy.py`, `src/model/gae.py` | `tests/unit/test_gae_math.py`, `tests/integration/test_ppo_one_update.py` | `results/training/ppo_learning_curve.png`, `docs/prd/PRD-PPO.md`, `docs/prd/PRD-GAE.md`, `docs/adr/ADR-008-sb3-variable-v-buffer.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.3 | Reward shaping with α/β/γ + P\_skills as **MUST**; full ablation matrix ≥5 seeds/cell. | `src/env/reward.py`, `scripts/run_ablation.py` | `tests/unit/test_reward_shape.py`, `tests/integration/test_ablation_matrix.py` | `results/ablation/reward_ablation_heatmap.png`, `results/ablation/seed_table.csv`, `docs/adr/ADR-007-reward-upgrade-MUST.md`, `docs/prd/PRD-SKILLS.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.4 | 2500–3000 word essay on Active Knowledge Architecture; 4 sections, 8–12 citations, 2 diagrams. | n/a (prose deliverable) | `tests/architecture/test_essay_wordcount.py` | `docs/essay/active_knowledge_architecture.md`, `docs/diagrams/aka_overview.png`, `docs/diagrams/aka_feedback_loop.png` | `<bootstrap-commit>` | ⬜ planned |
| §2 — convergence | Dual criterion: rolling-100 reward stable ±2% × 50 episodes **AND** entropy < threshold. | `src/services/convergence.py` | `tests/unit/test_convergence_dual_criterion.py` | `results/training/convergence_panel.png` | `<bootstrap-commit>` | ⬜ planned |
| §2 — betweenness | Betweenness centrality once per seed × ≥5 seeds; mean ± std + 95% CI. | `src/graphify/centrality.py` | `tests/unit/test_betweenness_seed_stability.py` | `results/graphs/betweenness_ci.png`, `results/graphs/betweenness_table.csv` | `<bootstrap-commit>` | ⬜ planned |
| §2 — encoder | GraphSAGE vs MLP encoder comparison gated by ADR-004. | `src/model/encoders/graphsage.py`, `src/model/encoders/mlp.py` | `tests/unit/test_encoder_parity.py` | `results/training/encoder_compare.png`, `docs/adr/ADR-004-graphsage-vs-mlp-encoder.md` | `<bootstrap-commit>` | ⬜ planned |

---

## Deliverables (per ex04 brief checklist)

| Brief § | Requirement | Code file | Test file | Chart / artifact | Commit SHA | Status |
|---|---|---|---|---|---|---|
| Deliv-1 | Runnable CLI: `uv run main.py --algo ppo` and `--mode refactor`. | `main.py`, `src/cli/entrypoint.py` | `tests/integration/test_cli_smoke.py` | `results/cli/help_screenshot.png` | `<bootstrap-commit>` | ⬜ planned |
| Deliv-2 | Programmatic NetworkX/pyvis screenshots; Obsidian hero shots committed. | `scripts/render_graphs.py`, `scripts/capture_obsidian.py` | `tests/integration/test_screenshot_pipeline.py` | `results/screenshots/hero_obsidian.png`, `results/screenshots/networkx_overview.png`, `docs/adr/ADR-009-screenshot-pipeline.md` | `<bootstrap-commit>` | ⬜ planned |
| Deliv-3 | PROMPTS.md — verbatim prompts used per Phase. | n/a | `tests/architecture/test_prompts_freshness.py` | `docs/shared/PROMPTS.md` | `<bootstrap-commit>` | ⬜ planned |
| Deliv-4 | CLAUDE.md global standards (≤150 LOC, TDD, OOP, no hardcoded values, DRY, ruff clean, uv-only). | `CLAUDE.md` | `tests/architecture/test_file_size_limit.py` | n/a | `<bootstrap-commit>` | ✅ done |
| Deliv-5 | `pyproject.toml` with uv + ruff + pytest-cov ≥85% gate. | `pyproject.toml` | `tests/architecture/test_coverage_gate.py` | n/a | `<bootstrap-commit>` | ✅ done |
| Deliv-6 | CI pipeline (lint + test + coverage + LOC guard). | `.github/workflows/ci.yml` | n/a (CI itself) | CI badge in `README.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-7 | ADR-001 PythonClaw shim boundary. | n/a (decision doc) | n/a | `docs/adr/ADR-001-pythonclaw-shim-boundary.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-8 | ADR-002 GRAPHIFY adapter. | n/a | n/a | `docs/adr/ADR-002-graphify-adapter.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-9 | ADR-003 tiktoken cost metric. | n/a | n/a | `docs/adr/ADR-003-tiktoken-cost-metric.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-10 | ADR-004 GraphSAGE vs MLP encoder. | n/a | n/a | `docs/adr/ADR-004-graphsage-vs-mlp-encoder.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-11 | ADR-005 lazy-load broken semantics. | n/a | n/a | `docs/adr/ADR-005-lazy-load-broken-semantics.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-12 | ADR-006 multi-seed eval discipline. | n/a | n/a | `docs/adr/ADR-006-multi-seed-eval-discipline.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-13 | ADR-007 reward upgrade MUST. | n/a | n/a | `docs/adr/ADR-007-reward-upgrade-MUST.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-14 | ADR-008 SB3 variable-V buffer. | n/a | n/a | `docs/adr/ADR-008-sb3-variable-v-buffer.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-15 | ADR-009 screenshot pipeline. | n/a | n/a | `docs/adr/ADR-009-screenshot-pipeline.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-16 | ADR-010 TRACE.md governance (this file's update protocol). | n/a | `tests/architecture/test_trace_freshness.py` | `docs/adr/ADR-010-trace-governance.md` | `<bootstrap-commit>` | ⬜ planned |
| Deliv-17 | PRD-PPO — PPO product requirements. | n/a | n/a | `docs/prd/PRD-PPO.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-18 | PRD-GAE — GAE product requirements. | n/a | n/a | `docs/prd/PRD-GAE.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-19 | PRD-GRAPHIFY — graphify product requirements. | n/a | n/a | `docs/prd/PRD-GRAPHIFY.md` | `<bootstrap-commit>` | ✅ done |
| Deliv-20 | PRD-SKILLS — skills/reward product requirements. | n/a | n/a | `docs/prd/PRD-SKILLS.md` | `<bootstrap-commit>` | ✅ done |

---

## Phase legend

- **Phase 0** — bootstrap (CLAUDE.md, pyproject, CI, ADR-001..010, PRDs, TRACE skeleton). Closes with `chore: bootstrap`.
- **Phase 1** — refactor track: PythonClaw shim, GRAPHIFY adapter, token cost.
- **Phase 2** — env + reward shaping + ablation matrix.
- **Phase 3** — PPO + GAE + encoder comparison + convergence gate.
- **Phase 4** — essay §2.4, screenshots, PROMPTS.md freeze, final TRACE stamp.

## Update protocol

1. When a row moves ⬜ → 🟡, open a PR that edits **only** the `Status` column.
2. When a row moves 🟡 → ✅, replace `<bootstrap-commit>` with the real short SHA in the same commit that lands the artifact.
3. `scripts/stamp_trace.py` (added in Phase 0) reads `git log --format=%h -1 -- <path>` for each row and validates that the recorded SHA matches.
4. CI fails if any ✅ row has SHA `<bootstrap-commit>` or if any ✅ row references a missing file.

## Cross-links

- Submission guideline §1.4 contract → `CLAUDE.md` Human↔AI table.
- Locked decisions log → `instructions/assignment-4/locked_decisions.md`.
- Lecturer feedback (verbatim) → `instructions/assignment-4/lecturer_feedback.md`.
- Per-phase progress tracker → `instructions/assignment-4/final_review_progress.md`.

---

## Notation & conventions

- Paths are **repo-relative**; the repo root is the directory containing this `docs/` folder.
- Multi-file rows list the primary file first, then comma-separated peers.
- "n/a" in the **Code file** column means the row is a prose / config / decision artifact with no executable surface.
- "n/a" in the **Test file** column means the row is exercised transitively (e.g. ADRs are validated by the architecture tests that pin the decisions they record).
- A row may appear in **§1** *and* **Deliverables** when an ADR both records a decision (Deliv row) and gates an implementation (§1 row); the SHA must agree across both.

## Audit checklist (run before declaring a Phase complete)

1. Every row in this file with Status ✅ has:
   - a real short SHA (not `<bootstrap-commit>`),
   - a code file that exists on disk and is ≤150 LOC,
   - a test file that exists and is referenced by `pytest --collect-only`,
   - a chart / artifact at the recorded path (or "n/a" honestly).
2. `scripts/stamp_trace.py --check` exits 0.
3. `uv run ruff check src/ tests/ main.py` exits 0.
4. `uv run pytest tests/ --cov=src --cov-fail-under=85` exits 0.
5. The Phase-closing commit subject matches regex `^(Phase \d+|Phase 0 bootstrap|chore: bootstrap)`.
6. The commit includes the trailer `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.

## Phase 0 closure criteria

Phase 0 is **bootstrap-only**. It closes when:

- All ✅ rows in the **Deliverables** table above resolve to real files on disk.
- `<bootstrap-commit>` placeholders are replaced with the actual short SHA of the single `chore: bootstrap` commit (or the final Phase 0 commit if bootstrap is split).
- `tests/architecture/test_trace_freshness.py` (added by Deliv-16, currently ⬜) passes.
- ADR-010 lands and the TRACE governance loop is closed.

## Phase 1–4 promotion rules

- A ⬜ row may not be promoted to 🟡 until its PRD/ADR is signed off (per the §1.4 contract).
- A 🟡 row may not be promoted to ✅ until:
  - the test file exists and is RED before the implementation,
  - the implementation file exists and turns the test GREEN,
  - the artifact (chart / CSV / JSON) is regenerated from a deterministic seed,
  - the commit SHA is stamped into this file in the same commit.

## Known gaps tracked against this file

- ADR-010 (TRACE governance) is referenced but not yet authored — Phase 0 task.
- `tests/architecture/test_trace_freshness.py` is referenced but not yet authored — Phase 0 task.
- `scripts/stamp_trace.py` is referenced but not yet authored — Phase 0 task.
- The `results/` subtree is empty at bootstrap; each Phase 1–4 row creates its own subdirectory.

## How a grader should use this file

1. Open `docs/TRACE.md` (this file).
2. Pick any brief §-id from the left column.
3. Follow the **Code file** link to read the implementation.
4. Follow the **Test file** link to confirm behaviour is pinned.
5. Open the **Chart / artifact** to see evidence.
6. `git show <sha>` to read the commit that closed the row.

If any of those four steps 404, the row is mis-marked and CI should have failed — please file an issue against the repo.
