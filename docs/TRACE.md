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

## Brief §-id discipline (governance)

Brief **§1.1 / §1.2 / §1.3 are INTRODUCTORY DEFINITIONS** (PythonClaw
context, GRAPHIFY context, cost-metric framing) — they are NOT
requirement rows in this matrix. All requirement rows below are
anchored to **§2.1 / §2.2 / §2.3 / §2.4** (the brief's normative
sections) plus the §3 deliverables checklist. Where a definition from
§1.x is the natural home for context, it appears in the requirement
text only — never in the §-id column.

## Brief §1.x cross-link (definitions → F-ids)

Although §1.1/§1.2/§1.3 are not requirement rows, they are anchored to
concrete F-id features / N-id notes / PRDs / ADRs below so a grader can
still trace from any introductory definition to the artifact that
implements its semantics.

| Brief § | Topic | Covered by |
|---|---|---|
| §1.1 | PythonClaw Skills L1/L2/L3 | F15 + F18 + N9 (PRD-SKILLS + ADR-005 + ADR-011) |
| §1.2 | GRAPHIFY G=(V,E) | F1 + F2 + ADR-002 (PRD-GRAPHIFY) |
| §1.3 | Obsidian Vault | F9 + F10 + F17 + D9 + ADR-009 |
| CLAUDE.md §4 | Canonical config single-source | F19 |

---

## §1 — Refactor Track (PythonClaw → SOLID + DI)

> Anchored to brief **§2.1 (refactor deliverables)**. §1.1/§1.2/§1.3 of
> the brief are definitional context for what PythonClaw, GRAPHIFY and
> the cost metric *are*; the contractual requirements live in §2.1.

| Brief § | Requirement | Code file | Test file | Chart / artifact | Commit SHA | Status |
|---|---|---|---|---|---|---|
| §2.1 — PRD-SHIM | PythonClaw shim isolates legacy surface behind a typed boundary; swap window ≤24h. | `src/pythonclaw_shim/skill.py`, `src/pythonclaw_shim/registry.py` | `tests/unit/test_pythonclaw_shim.py` | `docs/adr/ADR-001-pythonclaw-shim-boundary.md` | `0165fa2` | ✅ done |
| §2.1 — **F2** | GRAPHIFY local re-implementation behind `GraphifyAdapter` (`.build()` / `.load()`); no external HTTP at import time. Implementation surface: `src/graphify/local_impl.py` (parser + dependency-edge builder behind the adapter contract, ADR-002). | `src/graphify/adapter.py`, `src/graphify/local_impl.py` | `tests/unit/test_graphify_adapter.py`, `tests/architecture/test_no_network_at_import.py` | `docs/adr/ADR-002-graphify-adapter.md`, `results/graphify_output.gpickle` | `0165fa2` | ✅ done |
| §2.1 | Cost-metric headline = tiktoken `cl100k_base`; chars/bytes appendix only. | `src/utils/token_cost.py` | `tests/unit/test_token_cost.py` | `results/cost/token_cost_table.csv`, `docs/adr/ADR-003-tiktoken-cost-metric.md` | `58bf82f` | ⬜ planned |
| §2.1 — lazy-load semantics (**F18** + N9) | Lazy-load broken pytest walks `sys.modules` AND enforces P95 token-cost budget; emits `P_skills = -5.0` penalty on break. Runtime sensor: `src/services/lazy_load_monitor.py` (F18 — ADR-005 invariant monitor; observer-only, no `sys.modules` mutation). | `src/utils/lazy_load_guard.py`, `src/services/lazy_load_monitor.py` | `tests/architecture/test_lazy_load_broken.py` | `docs/adr/ADR-005-lazy-load-broken-semantics.md` | `0165fa2` | ✅ done |
| §2.1 — **F19** (CLAUDE.md §4) | Canonical config single-source: every algorithm-relevant parameter (α, β, γ, λ, ε, ablation grids, seed lists, feature_dim, entropy threshold, P_skills magnitude) flows through `src/utils/config_loader.py`; no `.py` under `src/` opens `config/config.yaml` directly. | `src/utils/config_loader.py` | `tests/architecture/test_config_single_source.py` | `config/config.yaml`, `CLAUDE.md` §4 | `0165fa2` | ✅ done |
| §2.1 — SOLID | DI container, single-responsibility services, no god-objects. | `src/sdk/container.py`, `src/services/*.py` | `tests/architecture/test_solid_violations.py` | `docs/diagrams/solid_dep_graph.png` | `<bootstrap-commit>` | ⬜ planned |

---

## §2 — RL Track (PPO + GAE + Active Knowledge Architecture)

> Anchored to brief **§2.2 (environment + reward) and §2.3 (PPO+GAE
> training loop)** and §2.4 (essay). **NO Gymnasium**: brief §2.2
> explicitly bans Gymnasium ("ללא סביבת Gymnasium"). The environment
> implements `step` / `reset` over the refactored graph state via a
> custom class — there is NO `gymnasium` import in `src/env/`, and no
> `gymnasium.vector.AsyncVectorEnv` anywhere; an AST-level architecture
> test pins this. Parallelism uses a custom multiprocess wrapper or
> single-process per ADR-007's parallel-processing scope note.

| Brief § | Requirement | Code file | Test file | Chart / artifact | Commit SHA | Status |
|---|---|---|---|---|---|---|
| §2.2 | Custom graph environment exposing `step`/`reset` over the refactored graph state — **NO Gymnasium (brief §2.2 ban)**. | `src/env/graph_env.py` | `tests/unit/test_graph_env_contract.py`, `tests/architecture/test_env_no_gym.py` | `docs/diagrams/env_state_diagram.png` | `<bootstrap-commit>` | ⬜ planned |
| §2.2 | Reward shaping `R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t` with α=1.0, β=1.0, γ=0.5, **P_skills = −5.0** (lazy-load-break penalty, NEGATIVE); full ablation matrix ≥5 seeds/cell. | `src/env/reward.py`, `scripts/run_ablation.py` | `tests/unit/test_reward_shape.py`, `tests/integration/test_ablation_matrix.py` | `results/ablation/reward_ablation_heatmap.png`, `results/ablation/seed_table.csv`, `docs/adr/ADR-007-reward-upgrade-MUST.md`, `docs/prd/PRD-SKILLS.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.2 — betweenness | Betweenness centrality computed **exactly twice per seed** (training start + training end), aggregated across ≥5 seeds with mean ± std + 95% CI for both endpoints AND Δ. | `src/graphify/centrality.py` | `tests/unit/test_betweenness_seed_stability.py`, `tests/architecture/test_betweenness_call_count.py` | `results/graphs/betweenness_ci.png`, `results/graphs/betweenness_table.csv`, `docs/adr/ADR-006-multi-seed-eval-discipline.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.3 | PPO (Schulman 2017, arXiv:1707.06347) + GAE(λ) (Schulman 2016, arXiv:1506.02438) with proper advantage normalisation; SB3 buffer spike timeboxed 2h, padding-to-V_max=512 fallback gated by ADR-008. | `src/model/ppo_policy.py`, `src/model/gae.py` | `tests/unit/test_gae_math.py`, `tests/integration/test_ppo_one_update.py` | `results/training/ppo_learning_curve.png`, `docs/prd/PRD-PPO.md`, `docs/prd/PRD-GAE.md`, `docs/adr/ADR-008-sb3-variable-v-buffer.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.3 — convergence | Dual criterion: non-overlapping rolling-100 ±2% AND \|dH/dt\| < entropy_slope_threshold (per ADR-010). | `src/services/convergence.py` | `tests/unit/test_convergence_dual_criterion.py` | `results/training/convergence_panel.png`, `docs/adr/ADR-010-dual-convergence-criterion.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.3 — encoder | GraphSAGE (primary, variable-\|V\| via PyG DataLoader) vs MLP (V_max=512 padding fallback) encoder comparison gated by ADR-004 + ADR-008. | `src/model/encoders/graphsage.py`, `src/model/encoders/mlp.py` | `tests/unit/test_encoder_parity.py` | `results/training/encoder_compare.png`, `docs/adr/ADR-004-graphsage-vs-mlp-encoder.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.4 — essay | 2500–3000 word essay on **GRAPHIFY × AI agents** (brief-verbatim: הקשר בין מנוע GRAPHIFY לבין עבודה עם סוכני AI) — feedback loop, failure modes, governance; 4 sections, 8–12 citations, 2 diagrams. AKA (Active Knowledge Architecture) is the lecture-side framing; the brief itself uses the GRAPHIFY×AI phrasing. | n/a (prose deliverable) | `tests/architecture/test_essay_wordcount.py`, `tests/architecture/test_graphify_ai_essay.py` | `docs/essay/graphify_x_ai.md`, `docs/diagrams/aka_overview.png`, `docs/diagrams/aka_feedback_loop.png` | `<bootstrap-commit>` | ⬜ planned |
| §2.4 — cost analysis | Cost envelope documented in `docs/COST_ANALYSIS.md` (tiktoken cl100k_base headline + chars/bytes appendix). | n/a (prose + table deliverable) | `tests/architecture/test_cost_envelope.py` | `docs/COST_ANALYSIS.md`, `results/cost/token_cost_table.csv` | `<bootstrap-commit>` | ⬜ planned |

---

## New artifacts (F15/F16 features + D6/D7/D8 deliverables)

| Brief § | Requirement | Code file | Test file | Chart / artifact | Commit SHA | Status |
|---|---|---|---|---|---|---|
| §2.1 — F15 | `docs/SKILLS_ARCHITECTURE.md` — L1/L2/L3 theoretical deep-dive with ≥2 concrete usage examples (brief §2.1 Skills mandate). | n/a (prose deliverable) | `tests/architecture/test_skills_architecture_doc.py` | `docs/SKILLS_ARCHITECTURE.md` | `0165fa2` | ✅ done |
| §2.3 — F16 | `tests/test_learning_curve.py` asserts reward-over-training PNG produced AND Δ-Reward numeric is reported. | `scripts/render_learning_curve.py` | `tests/test_learning_curve.py` | `results/learning_curves/reward_vs_episode.png` | `<bootstrap-commit>` | ⬜ planned |
| §2.3 — D6 | Learning curve `reward_vs_episode.png` — mean ± 95% CI over ≥5 seeds. | `scripts/render_learning_curve.py` | `tests/test_learning_curve.py` | `results/learning_curves/reward_vs_episode.png` | `<bootstrap-commit>` | ⬜ planned |
| §2.3 — D7 | ΔReward (final − initial mean) reported in `docs/ANALYSIS.md` as mean ± std + 95% CI. | n/a (prose deliverable) | `tests/architecture/test_analysis_delta_reward.py` | `docs/ANALYSIS.md` | `<bootstrap-commit>` | ⬜ planned |
| §2.4 — D8 | Cost envelope detail in `docs/COST_ANALYSIS.md` — tiktoken cl100k_base totals, per-phase breakdown, budget ceiling. | n/a (prose + table deliverable) | `tests/architecture/test_cost_envelope.py` | `docs/COST_ANALYSIS.md` | `<bootstrap-commit>` | ⬜ planned |

---

## Deliverables (per ex04 brief §3 checklist)

| Brief § | Requirement | Code file | Test file | Chart / artifact | Commit SHA | Status |
|---|---|---|---|---|---|---|
| Deliv-1 | Runnable CLI: `uv run main.py --algo ppo` and `--mode refactor`. | `main.py`, `src/cli/entrypoint.py` | `tests/integration/test_cli_smoke.py` | `results/cli/help_screenshot.png` | `<bootstrap-commit>` | ⬜ planned |
| Deliv-2 | Programmatic NetworkX/pyvis screenshots; Obsidian hero shots committed. | `scripts/render_graphs.py`, `scripts/capture_obsidian.py` | `tests/integration/test_screenshot_pipeline.py` | `results/screenshots/hero_obsidian.png`, `results/screenshots/networkx_overview.png`, `docs/adr/ADR-009-screenshot-pipeline.md` | `<bootstrap-commit>` | ⬜ planned |
| Deliv-3 | PROMPTS.md — verbatim prompts used per Phase. | n/a | `tests/architecture/test_prompts_freshness.py` | `docs/shared/PROMPTS.md` | `<bootstrap-commit>` | ⬜ planned |
| Deliv-4 | CLAUDE.md global standards (≤150 LOC, TDD, OOP, no hardcoded values, DRY, ruff clean, uv-only). | `CLAUDE.md` | `tests/architecture/test_file_size_limit.py` | n/a | `a213652` | ✅ done |
| Deliv-5 | `pyproject.toml` with uv + ruff + pytest-cov ≥85% gate. | `pyproject.toml` | `tests/architecture/test_coverage_gate.py` | n/a | `a213652` | ✅ done |
| Deliv-6 | CI pipeline (lint + test + coverage + LOC guard). | `.github/workflows/ci.yml` | n/a (CI itself) | CI badge in `README.md` | `a213652` | ✅ done |
| Deliv-7 | ADR-001 PythonClaw shim boundary. | n/a (decision doc) | n/a | `docs/adr/ADR-001-pythonclaw-shim-boundary.md` | `58bf82f` | ✅ done |
| Deliv-8 | ADR-002 GRAPHIFY adapter. | n/a | n/a | `docs/adr/ADR-002-graphify-adapter.md` | `58bf82f` | ✅ done |
| Deliv-9 | ADR-003 tiktoken cost metric. | n/a | n/a | `docs/adr/ADR-003-tiktoken-cost-metric.md` | `58bf82f` | ✅ done |
| Deliv-10 | ADR-004 GraphSAGE vs MLP encoder. | n/a | n/a | `docs/adr/ADR-004-graphsage-vs-mlp-encoder.md` | `58bf82f` | ✅ done |
| Deliv-11 | ADR-005 lazy-load broken semantics. | n/a | n/a | `docs/adr/ADR-005-lazy-load-broken-semantics.md` | `58bf82f` | ✅ done |
| Deliv-12 | ADR-006 multi-seed eval discipline. | n/a | n/a | `docs/adr/ADR-006-multi-seed-eval-discipline.md` | `58bf82f` | ✅ done |
| Deliv-13 | ADR-007 reward upgrade MUST. | n/a | n/a | `docs/adr/ADR-007-reward-upgrade-MUST.md` | `58bf82f` | ✅ done |
| Deliv-14 | ADR-008 SB3 variable-V buffer. | n/a | n/a | `docs/adr/ADR-008-sb3-variable-v-buffer.md` | `58bf82f` | ✅ done |
| Deliv-15 | ADR-009 screenshot pipeline. | n/a | n/a | `docs/adr/ADR-009-screenshot-pipeline.md` | `58bf82f` | ✅ done |
| Deliv-16 | ADR-010 dual-criterion convergence definition. | n/a | `tests/unit/test_convergence_dual_criterion.py` | `docs/adr/ADR-010-dual-convergence-criterion.md` | `58bf82f` | ✅ done |
| Deliv-17 | PRD-PPO — PPO product requirements. | n/a | n/a | `docs/prd/PRD-PPO.md` | `58bf82f` | ✅ done |
| Deliv-18 | PRD-GAE — GAE product requirements. | n/a | n/a | `docs/prd/PRD-GAE.md` | `58bf82f` | ✅ done |
| Deliv-19 | PRD-GRAPHIFY — graphify product requirements. | n/a | n/a | `docs/prd/PRD-GRAPHIFY.md` | `58bf82f` | ✅ done |
| Deliv-20 | PRD-SKILLS — skills/reward product requirements. | n/a | n/a | `docs/prd/PRD-SKILLS.md` | `58bf82f` | ✅ done |
| §3 bug-report | ≥2 architectural bugs documented | docs/BUG_REPORT.md | tests/test_bug_report.py | — | <pending> | ⬜ |
| §3 before/after (**F17** writer + **F10** screenshots + **D9** before-evidence) | Obsidian Vault writer (F17 — `src/services/vault_writer.py`, idempotent vault materialisation, brief §2.1 / §3) plus the pre-refactor hero shot landed Phase 1 (**D9**: `results/figures/obsidian_before.png`). "after" shot (closes F10) pending post-refactor in Phase 2. | `src/services/vault_writer.py`, `scripts/capture_obsidian_stub.py` | `tests/test_screenshots.py`, `tests/unit/test_vault_writer.py` | `results/vault/`, `results/figures/obsidian_before.png` (✅ Phase 1, **D9**); `results/figures/obsidian_after.png` (⬜ Phase 2+, F10) | `0165fa2` | 🟡 before-shot landed (D9 ✅); after-shot pending (F10 partial) |

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
- A row may appear in **§2.x** *and* **Deliverables** when an ADR both records a decision (Deliv row) and gates an implementation (§2.x row); the SHA must agree across both.

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
- `<bootstrap-commit>` placeholders are replaced with the actual short SHAs of the two Phase 0 commits: bootstrap (`a213652`) for Deliv-4/5/6, planning (`58bf82f`) for Deliv-7..20.
- `tests/architecture/test_trace_freshness.py` is authored (Phase 0 ruff-fix follow-up).
- ADR-010 (dual-criterion convergence) has landed at `docs/adr/ADR-010-dual-convergence-criterion.md` and is referenced from PRD-PPO, the convergence service test, and this file.

## Phase 1–4 promotion rules

- A ⬜ row may not be promoted to 🟡 until its PRD/ADR is signed off (per the §1.4 contract).
- A 🟡 row may not be promoted to ✅ until:
  - the test file exists and is RED before the implementation,
  - the implementation file exists and turns the test GREEN,
  - the artifact (chart / CSV / JSON) is regenerated from a deterministic seed,
  - the commit SHA is stamped into this file in the same commit.

## Known gaps (files referenced above that do not yet exist on disk)

Sweep performed at Phase 0 ruff-fix; each entry is a planned artifact
that Phase 1–4 will create. Listed for grader transparency.

- ~~`docs/SKILLS_ARCHITECTURE.md` — F15, Phase 2 will author.~~ ✅ landed in Phase 1 at `0165fa2`.
- `docs/ANALYSIS.md` — D7, Phase 3 will author.
- `docs/COST_ANALYSIS.md` — D8, Phase 4 will author.
- `docs/BUG_REPORT.md` — Phase 4 deliverable.
- `docs/essay/graphify_x_ai.md` — §2.4 GRAPHIFY × AI agents essay, Phase 4 (AKA diagrams `docs/diagrams/aka_*.png` support this single essay; topic relabeled to brief-verbatim phrasing).
- `docs/diagrams/env_state_diagram.png` — Phase 2.
- `docs/diagrams/solid_dep_graph.png` — Phase 1.
- `docs/diagrams/aka_overview.png` — Phase 4.
- `docs/diagrams/aka_feedback_loop.png` — Phase 4.
- `docs/shared/PROMPTS.md` — Phase 4 freeze.

ADR-001..010 (10 files in `docs/adr/`) and PRD-PPO/GAE/GRAPHIFY/SKILLS
(4 files in `docs/prd/`) **exist on disk** and are committed at
`58bf82f` — these are the ✅-done Deliv-7..20 rows above.

Process gaps (separate from missing artifacts):

- ADR-010 final scope = dual-criterion convergence definition (locked
  per grade-strategy round); the "TRACE governance" responsibility
  documented in `Phase 0 closure criteria` above is absorbed into the
  Phase 0 ruff-fix commit and the future `scripts/stamp_trace.py` rather
  than a separate ADR.
- `tests/architecture/test_trace_freshness.py` is referenced but not yet
  authored — Phase 0 ruff-fix follow-up.
- `scripts/stamp_trace.py` is referenced but not yet authored —
  Phase 0 ruff-fix follow-up.
- The `results/` subtree is empty at bootstrap; each Phase 1–4 row
  creates its own subdirectory.

## How a grader should use this file

1. Open `docs/TRACE.md` (this file).
2. Pick any brief §-id from the left column.
3. Follow the **Code file** link to read the implementation.
4. Follow the **Test file** link to confirm behaviour is pinned.
5. Open the **Chart / artifact** to see evidence.
6. `git show <sha>` to read the commit that closed the row.

If any of those four steps 404, the row is mis-marked and CI should have failed — please file an issue against the repo.
