# PRD — PythonClawRefactorRL (Assignment 4 product requirements)

The project-wide requirements document for Assignment 4 of the Bar-Ilan
Vibe Coding Workshop ("Reverse Engineering, Structural Bug Discovery &
Infrastructure Optimization with Reinforcement Learning Agents on the
PythonClaw Platform", brief v1.0 dated 03-06-2026, lecturer Dr. Yoram
Segal). Per-component design lives in `PLAN.md`; the literal prompts
and AI workflow live in `docs/shared/PROMPTS.md`; the phased task list
with definition-of-done lives in `TODO.md`; the brief→test traceability
table lives in `docs/TRACE.md`.

> Teaching artefact, **not a production refactoring service**. The agent
> operates on a frozen snapshot of the PythonClaw `Skills` module; the
> "bugs" it surfaces are structural (architecture-smell category), not
> behavioural test failures. See §6 and §7 for the full honest-limitations
> list and the deliberate decision to self-grade below 100.

---

## 1. Vision

### 1.1 Headline goal

Per brief §1, reverse-engineer the architecture of one self-contained
module — the **PythonClaw `Skills` subsystem** — by (a) lifting its
source code into a directed weighted dependency graph G = (V, E) via a
local `GRAPHIFY`-equivalent static analyser, (b) wrapping that graph as
a Reinforcement-Learning environment **without** `gymnasium` (brief §2.2
explicitly forbids it), and (c) training a Deep-RL agent (PPO + GAE,
brief §2.3) to apply structural refactoring actions — `split_module`,
`merge_modules`, `rewire_dependency` — that improve modularity and
cohesion while reducing coupling. The agent's final state, compared
against the initial state, surfaces at least two architectural bugs
(brief §3, "Bug Report") and produces an Obsidian Vault with two graph
screenshots ("before" and "after" the agent's optimisation pass).

### 1.2 Three artefacts, one SDK

The submission ships three trainable / produced components behind one
Python facade (`RefactorSDK`):

1. **GRAPHIFY adapter** — converts the PythonClaw `Skills` source tree
   into a directed weighted graph G = (V, E) where v ∈ V is a code entity
   (class, module, function) and e = (u, v) ∈ E is a dependency edge
   (brief §1.2). Locally re-implemented behind `GraphifyAdapter` per
   ADR-002 because the upstream GRAPHIFY tool is not redistributable.
2. **Graph-as-MDP environment** — a custom training loop (no
   `gymnasium` — brief §2.2 forbids it because graph dimensionality
   changes per step) exposing `reset()`, `step(action)`, `state()`, and
   `done()` over the live G = (V, E) representation.
3. **PPO + GAE agent** (Schulman et al. 2017; Schulman et al. 2016,
   high-dimensional continuous control / GAE) — clipped surrogate
   objective with ε = 0.2 (brief §2.3) and GAE λ = 0.95 (brief §2.3),
   built on Stable-Baselines3 with a 2-hour timebox spike (per locked
   decision) and a padding/masking fallback if the per-step graph-shape
   change cannot be reconciled with SB3's fixed-shape replay buffer.

Plus one comparison artefact (before / after Obsidian Graph View
screenshots — brief §3), one bug-report deliverable (brief §3
"Bug Report"), and the deep-research essay on GRAPHIFY × AI agents
(brief §2.4, 2500–3000 words, 4 sections, 8–12 citations, 2 diagrams).

### 1.3 In scope

- Static analysis of the `Skills` module **only** (brief §1.1 — the
  module is the "tiered loading" mechanism with `Metadata` L1,
  `Instructions` L2, `Resources` L3).
- Local re-implementation of GRAPHIFY in `src/graphify/` behind the
  `GraphifyAdapter` interface (ADR-002).
- A directed, weighted dependency graph G = (V, E) with adjacency
  matrix A ∈ ℝ^{|V|×|V|} and node-feature matrix X ∈ ℝ^{|V|×D} where D
  covers LOC, cyclomatic complexity, in-degree, out-degree, and
  Degree Centrality (per-step) plus Betweenness Centrality (once per
  seed × ≥5 seeds, mean ± std + 95 % CI — per locked decision; brief §2.2
  explicitly flags Betweenness as a CPU bottleneck not to compute every
  step).
- A discrete action space {`split_module`, `merge_modules`,
  `rewire_dependency`} (brief §2.2) — exact arity is up to the
  architect; we ship 3 actions with per-action parameter binding via a
  parameterised-action wrapper.
- Reward `R_t = ΔModularity + ΔCohesion − Coupling_Penalty` (brief §2.2,
  baseline form) **upgraded** per the brief's "highly recommended"
  upgrade to `R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling_Penalty
  + λ·skills_loading_safety_penalty` where `α, β, γ, λ` are calibrated
  per ADR-003 and the **full ablation matrix** runs ≥ 5 seeds per cell
  (locked decision). The `skills_loading_safety_penalty` term (brief
  §2.3, "advanced punishment") prevents the agent from breaking the
  L1 → L2 → L3 tiered-load contract that defines the `Skills` module.
- PPO clipping at ε = 0.2 (brief §2.3) + GAE at λ = 0.95 (brief §2.3) +
  a "trust region" interpretation in the report (brief §2.3 verbatim).
- Token-budget accounting in `tiktoken cl100k_base` headline (locked
  decision) with chars / bytes appendix; brief §2.4 requires a token
  count of the `Skills` module fed to the agent as input.
- Compute-resource and runtime accounting for the PPO training run
  (brief §2.4 verbatim).
- Obsidian Vault generation: programmatic NetworkX / pyvis hero shots
  (locked decision) plus the "before" / "after" Graph View screenshots
  required by brief §3.
- Deep-research essay (`docs/RESEARCH_ESSAY.md`, brief §2.4) on the
  GRAPHIFY × AI-agent relationship, 2500–3000 words, 4 sections,
  8–12 citations, 2 diagrams (locked decision).
- Bug-report deliverable (`docs/BUG_REPORT.md`, brief §3) detailing at
  least two architectural / structural bugs the agent surfaced.

### 1.4 Out of scope

- The full PythonClaw platform — only the `Skills` module (brief §1.1
  explicit scoping).
- Behavioural / runtime test failures in PythonClaw — the agent finds
  *structural* bugs only (architecture smells, not stack traces).
- `gymnasium` environment wrapping (brief §2.2 explicit prohibition —
  the graph dimensions change per step, which violates the rigid
  Gymnasium interface).
- A live commit of the agent's refactor back to the upstream PythonClaw
  repo — the agent's proposed actions are written to
  `results/proposed_refactor.json`, not pushed.
- A multi-module sweep — single `Skills` module per the brief.
- A "real" GRAPHIFY install — the upstream tool is replaced by the
  local re-impl per ADR-002.

---

## 2. Objectives & KPIs

Quantitative success criteria (every claim in §7 acceptance criteria
links back to one of these KPIs).

### 2.1 Algorithmic objectives

- **O1**. Train PPO + GAE to convergence under the dual criterion
  (locked decision): rolling-100-episode mean reward stays within
  ±2 % for 50 consecutive episodes **AND** policy entropy falls below
  `entropy_threshold` (calibrated per ablation). Both gates must fire,
  not one.
- **O2**. Final-state modularity (Newman-Girvan modularity Q) **strictly
  greater** than initial-state modularity, mean ± std over ≥ 5 seeds,
  95 % CI excludes zero.
- **O3**. Final-state cohesion (LCOM4 or equivalent — picked in
  ADR-005) strictly greater than initial; same statistical bar as O2.
- **O4**. Final-state coupling (afferent + efferent coupling sum,
  normalised) strictly less than initial; same statistical bar.
- **O5**. The full α / β / γ + `P_skills` ablation matrix runs at ≥ 5
  seeds per cell (locked decision); winning cell's KPI uplift over the
  baseline cell is reported with paired-bootstrap 95 % CI.
- **O6**. Betweenness Centrality is computed exactly once per seed
  (locked decision), reported as mean ± std + 95 % CI across ≥ 5 seeds
  — never per `step()`.

### 2.2 Engineering KPIs

- **E1**. ≥ 85 % `pytest` coverage on `src/` (CLAUDE.md hard rule),
  enforced from day 1 in CI.
- **E2**. Zero ruff violations (CLAUDE.md hard rule).
- **E3**. Every `.py` ≤ 150 LOC (CLAUDE.md hard rule).
- **E4**. `uv sync --dev && uv run pytest` reproduces a green build on
  a clean checkout.
- **E5**. CI green on every commit from the bootstrap commit forward.
- **E6**. Reproducibility test: two seeded PPO rollouts produce
  bit-identical trajectories on CPU; CUDA / MPS drift named in README.

### 2.3 Deliverable KPIs

- **D1**. Obsidian Vault contains exactly two Graph View screenshots
  ("before" and "after") plus auto-linked NetworkX / pyvis hero shots.
- **D2**. Bug-report names ≥ 2 distinct architectural bugs the agent
  surfaced — each with a graph-evidence pointer (the offending node /
  edge subset) and a proposed structural remedy.
- **D3**. Research essay hits 2500–3000 words, 4 sections, 8–12
  citations (DOI / arXiv / ACM where applicable), 2 diagrams.
- **D4**. Token count of the `Skills` module input (cl100k_base
  headline, chars + bytes appendix) reported in `docs/COST_ANALYSIS.md`.
- **D5**. PPO training wall-clock + CPU-seconds reported alongside the
  token count.

---

## 3. Functional requirements

Each `F#` is traceable to a brief §-id. The trace matrix
(`docs/TRACE.md`) carries the full bidirectional mapping.

### 3.1 Static analysis & graph construction (brief §1.2, §2.1, §2.2)

- **F1 (State representation, brief §2.2)**. Convert the `Skills`
  source tree into a directed weighted dependency graph G = (V, E)
  with adjacency matrix A ∈ ℝ^{|V|×|V|} (binary or weighted by edge
  frequency) and a node-feature matrix X ∈ ℝ^{|V|×D} carrying LOC,
  cyclomatic complexity (radon McCabe), in-degree, out-degree, and
  Degree Centrality. The state representation is the per-step input
  to the policy network.

- **F2 (GRAPHIFY adapter, brief §1.2)**. Provide a `GraphifyAdapter`
  interface (ADR-002) backed by a local `src/graphify/` implementation
  that parses Python AST → entities → dependency edges. The adapter
  hides which implementation produced the graph so an upstream
  GRAPHIFY install could swap in later (24 h swap window per locked
  decision).

- **F3 (Centrality metrics, brief §2.2)**. Compute Degree Centrality
  cheaply on every `step()`; compute Betweenness Centrality **once per
  seed** (locked decision), aggregate across ≥ 5 seeds as mean ± std
  + 95 % CI, report at training-end only — never as part of the
  per-step state vector (brief §2.2 explicit guidance to avoid the
  CPU bottleneck).

### 3.2 RL environment (brief §2.2, "no Gymnasium")

- **F4 (Action space, brief §2.2)**. Discrete action set
  {`split_module(v)`, `merge_modules(u, v)`, `rewire_dependency(u, v, w)`}
  parameterised per action via a parameterised-action wrapper. Each
  action transforms G = (V, E) in place and the next state is
  re-extracted by the adapter.

- **F5 (Custom training loop, brief §2.2)**. Implement
  `RefactorEnv.reset()`, `RefactorEnv.step(action)`, `RefactorEnv.state()`,
  `RefactorEnv.done()` **without** subclassing `gymnasium.Env` — the
  brief explicitly bans it because |V| and |E| change per step.

- **F6 (Reward, brief §2.2 + §2.3 "highly recommended" upgrade +
  "advanced punishment")**.
  Baseline: `R_t = ΔModularity + ΔCohesion − Coupling_Penalty`.
  Upgraded (shipped): `R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling_Penalty
  + λ·P_skills_loading_safety`. Coefficients calibrated via the §3.6
  ablation matrix.

### 3.3 PPO + GAE trainer (brief §2.3)

- **F7 (PPO trainer, brief §2.3)**. Implement PPO with clipped
  surrogate objective at ε = 0.2 (brief §2.3 verbatim) using
  Stable-Baselines3 (brief §2.3 explicitly permits SB3 / RLlib instead
  of hand-rolled). The 2-hour SB3 spike (locked decision) verifies
  the variable-graph-shape ↔ SB3 fixed-buffer reconciliation; if it
  fails, the padding / masking fallback ships (ADR-006).

- **F8 (GAE, brief §2.3)**. λ = 0.95 (brief §2.3 verbatim) for the
  Generalised Advantage Estimator
  `A_t = Σ (γλ)^l δ_{t+l}`, `δ_t = r_t + γV(s_{t+1}) − V(s_t)`
  (Schulman et al. 2016).

### 3.4 GRAPHIFY adapter + Obsidian export (brief §2.1, §3)

- **F9 (Obsidian Vault, brief §2.1)**. Export the live G = (V, E) at
  training start and training end as a Markdown-based vault under
  `results/obsidian_vault/` with one `.md` file per node and
  `[[wikilink]]` edges. Programmatic NetworkX / pyvis hero shots
  (locked decision) supplement the two Graph View screenshots.

- **F10 (Before / After screenshots, brief §3)**. Save
  `results/screenshots/before.png` and `results/screenshots/after.png`
  as deterministic snapshots of the Obsidian Graph View (or pyvis
  equivalent if Obsidian's layout proves non-deterministic — see §6).

### 3.5 Ablation matrix (brief §2.2 "highly recommended" upgrade)

- **F11 (Ablation matrix, locked decision)**. Sweep α ∈ {0.5, 1.0, 2.0},
  β ∈ {0.5, 1.0, 2.0}, γ ∈ {0.5, 1.0, 2.0}, and `P_skills` ∈ {on, off}
  → 54 cells (locked: 5 seeds per cell, 270 runs total budgeted). The
  matrix lives in `docs/ABLATION.md` with mean ± std + paired-bootstrap
  95 % CI per cell. Compute budget capped at the brief §2.4 envelope.

### 3.6 Multi-seed evaluation (locked decision, brief §2.4 honest accounting)

- **F12 (Multi-seed eval, locked decision)**. Every headline number in
  the report — final modularity, final cohesion, final coupling,
  Betweenness Centrality — runs ≥ 5 seeds and is reported as
  mean ± std + 95 % CI. No single-seed claims.

### 3.7 Bug report (brief §3)

- **F13 (Bug report, brief §3)**. `docs/BUG_REPORT.md` names ≥ 2
  architectural bugs (e.g. "L2 instructions reach into L3 resources
  directly, bypassing the L1 metadata gate", or "circular dependency
  between Skill registry and Skill executor"). Each bug carries: the
  evidence subgraph, the agent's surfacing trajectory (which step it
  appeared on), the proposed structural remedy, and a falsifiable
  test that would fail if the remedy were not applied.

### 3.8 GRAPHIFY × AI essay (brief §2.4, locked decision)

- **F14 (Research essay, brief §2.4)**. `docs/RESEARCH_ESSAY.md`,
  2500–3000 words, 4 sections (Introduction; Static analysis tools
  landscape; AI-agent integration patterns; Limitations and future
  work), 8–12 citations with DOI / arXiv / ACM where applicable, 2
  diagrams. Answers the three brief §2.4 prompts verbatim: "What is
  the connection between GRAPHIFY and AI agents?", "Which tools can
  help automate the analysis?", "When are they appropriate, and what
  are their limitations?"

---

## 4. Non-functional requirements

Inherited from `CLAUDE.md` Hard Constraints, plus A4-specific additions.

- **N1**. Every `.py` file ≤ 150 LOC (hard, enforced by a CI check;
  CLAUDE.md §1).
- **N2**. TDD: tests written before implementation; coverage **≥ 85 %**
  on the `src/` package; enforced from day 1 in CI (CLAUDE.md §2).
- **N3**. Zero ruff violations on `src/`, `tests/`, `main.py`,
  `scripts/` (CLAUDE.md §6).
- **N4**. **uv-only** dependency management; `uv sync --dev` reproduces
  the environment from `uv.lock`. No pip, no conda (CLAUDE.md §7).
- **N5**. CI green from the bootstrap commit forward. The
  `.github/workflows/ci.yml` runs `uv sync --dev && uv run ruff check
  && uv run pytest --cov=src --cov-fail-under=85` on every push.
- **N6**. Deterministic seeds. `src/utils/seeding.py` exposes
  `set_global_seed(seed: int)` that seeds Python `random`, `numpy`,
  `torch` (CPU + CUDA), `PYTHONHASHSEED`,
  `torch.backends.cudnn.deterministic=True`,
  `torch.use_deterministic_algorithms(True, warn_only=True)`.
  Known non-determinism (CUDA `scatter_add` / `index_add`,
  mixed-precision drift, multi-worker DataLoader shuffle order,
  MPS kernels) named explicitly in the README.
- **N7**. No hardcoded algorithm parameters. All α, β, γ, λ, ε, γ_PPO,
  λ_GAE, learning rates, batch sizes, entropy coefficients,
  ablation grids, and seed lists live in `config/config.yaml`
  (CLAUDE.md §4).
- **N8**. Commit subject regex `^(Phase \d+|Phase 0 bootstrap|chore: bootstrap)`
  (locked decision); co-author trailer
  `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` on every
  AI-implemented commit.
- **N9**. Lazy-load test. `tests/test_skills_lazy_load.py` walks
  `sys.modules` after a `from skills import ...` to assert that
  L3 `Resources` are not eagerly loaded by an L1 `Metadata` import
  — guards the "lazy-load broken" failure mode (locked decision).
- **N10**. Token P95 budget. The `Skills` module token count (cl100k_base)
  must not regress > 5 % from the baseline snapshot taken at Phase 0;
  enforced by `tests/test_token_budget.py`.
- **N11**. Forbidden PII list ("REDACTED-NAME", "REDACTED-HANDLE", "REDACTED",
  "REDACTED-ID", "GoogleDrive-REDACTED-HANDLE") never appears in any tracked
  file. Pre-commit grep guard.

---

## 5. Acceptance criteria

### 5.1 Per-requirement DoD

| Req | Acceptance criterion | Evidence pointer |
|-----|---------------------|------------------|
| F1  | `state()` returns `(A, X)` with `A.shape == (|V|, |V|)` and `X.shape == (|V|, D)` where D ≥ 5 | `tests/test_state_repr.py::test_state_shape` |
| F2  | `GraphifyAdapter.parse(path) -> G` returns a `networkx.DiGraph` with weighted edges; ADR-002 swap window verified by a stub-swap unit test | `tests/test_graphify_adapter.py::test_adapter_contract` |
| F3  | Degree Centrality computed every step; Betweenness Centrality called exactly once per `train_seed()` call | `tests/test_centrality.py::test_betweenness_call_count` |
| F4  | `step(action)` mutates G and returns `(state, reward, done, info)` for all three action types | `tests/test_env_step.py::test_all_actions` |
| F5  | `RefactorEnv` does not import `gymnasium`; `grep -r "gymnasium" src/env/` returns empty | `tests/test_env_no_gym.py::test_no_gymnasium_import` |
| F6  | Reward computation matches `α·ΔMod + β·ΔCoh − γ·CouplingPenalty + λ·P_skills` symbolically | `tests/test_reward.py::test_reward_form` |
| F7  | PPO clip ratio bounded in `[1 − ε, 1 + ε]` with ε = 0.2 over a synthetic rollout | `tests/test_ppo.py::test_clip_ratio_bounds` |
| F8  | GAE advantage matches the closed-form on a 5-step synthetic trajectory | `tests/test_gae.py::test_gae_closed_form` |
| F9  | `results/obsidian_vault/` contains one `.md` per node and `[[wikilink]]` edges | `tests/test_obsidian_export.py::test_vault_structure` |
| F10 | `results/screenshots/before.png` and `after.png` exist, non-empty, > 1 KB each | `tests/test_screenshots.py::test_before_after_exist` |
| F11 | `docs/ABLATION.md` contains 54 cells × 5 seeds = 270 rows, each with mean / std / 95 % CI | `tests/test_ablation_table.py::test_matrix_complete` |
| F12 | No headline number in `docs/ANALYSIS.md` is single-seed; regex check for "seed=" plus "mean" plus "std" co-occurrence | `tests/test_analysis_seed_audit.py::test_no_single_seed_claims` |
| F13 | `docs/BUG_REPORT.md` lists ≥ 2 bugs with subgraph evidence and remedy | `tests/test_bug_report.py::test_min_two_bugs` |
| F14 | `docs/RESEARCH_ESSAY.md` is 2500–3000 words, 4 H2 sections, ≥ 8 citations, 2 figure references | `tests/test_essay_shape.py::test_word_count_and_structure` |

### 5.2 Project-level DoD

- Clean checkout: `uv sync --dev && uv run pytest` is green, ≥ 85 %
  coverage.
- `uv run ruff check src/ tests/ main.py scripts/` → zero violations.
- `uv run python scripts/generate_results.py` reproduces the before /
  after screenshots, the ablation table, the bug report, and all
  headline numbers in `docs/ANALYSIS.md`.
- Every `F#` row in §5.1 has an evidence pointer.
- `docs/TRACE.md` covers every brief §-id → at least one `F#` and at
  least one test.
- All numeric claims in `docs/ANALYSIS.md` cite **seed**, **episode
  count**, and **mean ± std + 95 % CI** — no bare adjectives like
  "converges fast" or "outperforms".
- CI is green on the merge commit.
- The forbidden-PII grep guard (N11) reports zero hits.

---

## 6. Dependencies & limitations

### 6.1 External dependencies

- **PythonClaw source repo (URL unconfirmed at PRD time)**. The brief
  §2.1 says "clone the official PythonClaw source from GitHub" but
  does not give a URL. ADR-001 documents the BOOTSTRAP_NOW path:
  ship a `PythonClawShim` (a small representative `Skills`-module
  stand-in) for 24 hours while the real URL is confirmed by the
  lecturer; swap to the real source within the 24-hour window. This is
  a locked decision.
- **GRAPHIFY**. Upstream tool is not redistributable; we ship a local
  re-implementation under `src/graphify/` behind `GraphifyAdapter`
  (ADR-002). The adapter contract lets a real GRAPHIFY install swap
  in later without breaking the env / agent / report.
- **Obsidian**. Used for the Graph View screenshots (brief §1.3 + §3).
  Obsidian's Graph View layout is force-directed and **not
  deterministic** across runs — see §6.2 limitation L2. We fall back
  to pyvis-rendered deterministic layouts as the grader-facing
  artefacts, and treat the Obsidian screenshots as "best-effort hero
  shots" rather than reproducibility-critical evidence.
- **Stable-Baselines3**. Brief §2.3 explicitly permits SB3. We adopt
  it, with the variable-graph-shape ↔ fixed-buffer reconciliation
  risk timeboxed to a 2-hour spike (locked decision); fallback is a
  padding / masking wrapper documented in ADR-006.
- **tiktoken** (cl100k_base) for the headline token count
  (locked decision); chars / bytes appendix in `docs/COST_ANALYSIS.md`.

### 6.2 Honest limitations

The submission ships as a pedagogical artefact. The following caveats
are called out head-on (rather than buried) so the analysis in
`docs/ANALYSIS.md` is interpreted in the right frame:

- **L1 — PythonClaw URL unconfirmed.** Until the lecturer confirms the
  canonical GitHub URL for the PythonClaw `Skills` module, the agent
  trains against the `PythonClawShim` (ADR-001). Any claim about
  "PythonClaw's architecture" before the swap is, strictly, a claim
  about the shim. The 24-hour swap window is the contract: swap and
  re-run within 24 h of URL confirmation.
- **L2 — Obsidian Graph View layout is non-deterministic.** Obsidian's
  force-directed layout converges to different fixed points across
  runs even with identical input vaults. The "before" / "after"
  screenshots are therefore visually illustrative, not pixel-stable.
  The pyvis / NetworkX hero shots (locked decision) are the
  reproducibility-critical visual artefacts; Obsidian is the brief-
  required ceremony layer on top.
- **L3 — SB3 abstracts PPO internals.** Using SB3 means the PPO clip
  ratio, GAE buffer, and advantage normalisation live in third-party
  code; we cannot unit-test the exact float values inside SB3's
  implementation. We compensate by (a) testing our adapter's outputs
  against closed-form GAE on synthetic trajectories (F8 evidence) and
  (b) reading SB3's source at the version we pin and naming the
  commit hash in `docs/THEORY.md`.
- **L4 — Structural bugs ≠ behavioural bugs.** The bug report (F13,
  brief §3) names architectural smells (e.g. tier-skip in the
  L1 → L2 → L3 contract). It does **not** claim these bugs cause
  PythonClaw to crash or return wrong results. A behavioural failure
  would require running PythonClaw's test suite, which is out of
  scope.
- **L5 — Single-module scope.** Only the `Skills` module is analysed
  (brief §1.1 scoping). Results do not generalise to PythonClaw as a
  whole and we make no claim that they do.
- **L6 — Reward is hand-designed.** α, β, γ, λ are author choices
  calibrated via the §3.6 ablation matrix, not learned from data.
  An inverse-RL formulation that learns the reward from human-rated
  refactors is future work, not A4 scope.
- **L7 — Reproducibility has known gaps.** CPU determinism is enforced
  (N6); CUDA `scatter_add` / `index_add`, mixed-precision drift,
  multi-worker DataLoader shuffle order, and MPS kernels can still
  introduce run-to-run drift. Named in the README's
  "Reproducibility caveats" subsection.

---

## 7. Honest framing — self-grade target

The architect's standing decision (Assignment 1 lesson, locked across
all subsequent assignments): **self-grade target is not 100**. A1's
over-confidence in declaring "all gates passed" without surfacing the
five hidden limitations cost credibility, and the lecturer's feedback
on A1 made the cost explicit. A4 inherits that lesson and prices it in
up front:

- **Self-grade target**: 88–92 out of 100, depending on how the
  L1 (PythonClaw URL), L2 (Obsidian non-determinism), L3 (SB3
  black box), and L4 (structural-vs-behavioural-bug) limitations
  resolve at submission time.
- **Why not higher**: every one of L1–L7 is a real reduction in the
  strength of the claim the submission can honestly make, and the
  rubric rewards honest framing over inflated certainty (A1
  feedback verbatim).
- **Why not lower**: §5 acceptance criteria are quantitative and
  testable; §3 functional coverage is exhaustive against the brief;
  the ablation matrix and multi-seed gates are stricter than the
  brief requires.

This section is the contract the submission writes with itself **before**
seeing the grade — not retroactive cover after.

---

## 8. §-id traceability anchor

The full brief-§ → F# → test mapping lives in `docs/TRACE.md` (this
file is the anchor; TRACE.md is the live table). Bidirectional
coverage:

- **Forward** (brief → code): every brief §-id from §1.1 through §3 has
  at least one `F#` in §3 of this PRD and at least one test in `tests/`.
- **Reverse** (code → brief): every `F#` in §3 names the brief §-id it
  satisfies inline.

TRACE.md is regenerated by `scripts/regen_trace.py` on every PR; CI
fails the PR if any brief §-id loses coverage.

Brief §-id coverage anchored here:

- §1.1 (PythonClaw `Skills` module — tiered L1/L2/L3 loading) → §1.3
  scope statement above, F1, F2, F9, N9 (lazy-load test).
- §1.2 (GRAPHIFY definition: G = (V, E), v ∈ V code entities, e edges
  as dependencies) → F1, F2, ADR-002.
- §1.3 (Obsidian Markdown + Graph View) → F9, F10, L2.
- §2.1 (Stage 1: clone, static analysis, Obsidian) → F2, F9, F10,
  L1, ADR-001.
- §2.2 (Stage 2: Dataset, graph→RL without Gymnasium; state rep,
  centrality, action space, reward) → F1, F3, F4, F5, F6.
- §2.3 (Stage 3: PPO + GAE as reverse-engineering engine; ε = 0.2,
  λ = 0.95, advanced punishment) → F6, F7, F8.
- §2.4 (Stage 4: cost / resource / AI-agent analysis; token count;
  GRAPHIFY × AI essay) → D4, D5, F14.
- §3 (Deliverables: before/after Obsidian screenshots, ablation
  analysis, bug report) → F10, F11, F13, D1, D2.

---

## 9. References

### Brief

- Assignment 4 brief v1.0 dated 03-06-2026, lecturer Dr. Yoram Segal,
  "Reverse Engineering, Structural Bug Discovery & Infrastructure
  Optimization with Reinforcement Learning Agents on the PythonClaw
  Platform". §-ids cited verbatim throughout §3 and §8 above.

### Lecture material

- Lecture 7 (PPO + GAE) — the verbatim derivations the brief §2.3
  points at; transcribed in `docs/THEORY.md`.
- Active Knowledge Architecture (AKA) source — informs the
  L1 / L2 / L3 tiered-loading discussion in `docs/RESEARCH_ESSAY.md`.

### Papers (anchor set — full list in `docs/RESEARCH_ESSAY.md`)

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.
  (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016).
  *High-Dimensional Continuous Control Using Generalized Advantage
  Estimation.* ICLR.
- Newman, M. E. J., Girvan, M. (2004). *Finding and evaluating
  community structure in networks.* Phys. Rev. E 69, 026113.
- Henderson, B., Sellers, K. (1996). *Application of Cohesion and
  Coupling Metrics for Object-Oriented Design.* JOOP.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement
  Learning Implementations.* JMLR 22(268).
- Freeman, L. C. (1977). *A set of measures of centrality based on
  betweenness.* Sociometry 40(1).

### Project documents

- `CLAUDE.md` — global coding standards + §1.4 architect/implementer
  contract.
- `PLAN.md` — architecture, ADRs (001 PythonClawShim, 002 GraphifyAdapter,
  003 reward calibration, 004 action space, 005 cohesion metric,
  006 PPO buffer fallback), module-level design.
- `docs/shared/PROMPTS.md` — the literal prompts and AI-workflow
  narrative.
- `TODO.md` — phased task list with definition-of-done.
- `docs/THEORY.md` — verbatim LaTeX for PPO clipped surrogate, GAE
  advantage, modularity Q, cohesion, coupling, and the
  `P_skills_loading_safety` penalty.
- `docs/ANALYSIS.md` — multi-seed results (mean ± std + 95 % CI).
- `docs/ABLATION.md` — 54-cell × 5-seed ablation matrix.
- `docs/BUG_REPORT.md` — ≥ 2 architectural bugs the agent surfaced.
- `docs/RESEARCH_ESSAY.md` — 2500–3000-word GRAPHIFY × AI essay
  (brief §2.4, F14).
- `docs/COST_ANALYSIS.md` — tiktoken cl100k_base headline + chars /
  bytes appendix + PPO runtime accounting (D4, D5, brief §2.4).
- `docs/TRACE.md` — bidirectional brief §-id ↔ F# ↔ test mapping.
- `docs/adr/ADR-001-pythonclaw-shim.md` — BOOTSTRAP_NOW path + 24 h
  swap window.
- `docs/adr/ADR-002-graphify-adapter.md` — local GRAPHIFY re-impl
  behind adapter interface.
