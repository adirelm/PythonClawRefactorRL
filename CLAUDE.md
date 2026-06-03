# CLAUDE.md — Global Coding Standards for PythonClawRefactorRL

## Project Context

PythonClawRefactorRL is the Assignment-4 deliverable for the Bar-Ilan University
*Vibe Coding & Reinforcement Learning* workshop. The pipeline applies PPO with
Generalised Advantage Estimation (GAE) to the **PythonClaw Skills graph** to
learn a refactoring policy that minimises module coupling and maximises
cohesion / modularity, with a hard penalty on lazy-load breakage.

1. **Skills-only scope** — we operate strictly on the Skills-layer dependency
   graph (no app code, no third-party deps). Scope lock asserted in tests.
2. **PPO + GAE** (Schulman et al. 2015 / 2017) — clipped-objective policy
   gradient on top of node-action moves over the live Skills graph.
3. **Custom training loop** — we do *not* wrap the env as a `gym.Env`; the
   loop lives in `src/services/ppo_trainer.py` so we keep full control over
   variable-|V| handling, betweenness-call discipline, and seed plumbing.
   Stable-Baselines3 is a permitted reference per the brief but is not the
   path of record.

See `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md`, and `docs/THEORY.md` for the
binding specification.

## Human ↔ AI Responsibility Contract (§1.4)

§1.4 of the submission guidelines frames the developer as **architect** and the
AI as **implementer**. This file is the contract that boundary makes explicit.
Each row says **who decides** before any code is generated.

| Concern | Human-decided (non-delegable) | AI-delegated |
|---|---|---|
| Requirements (PRD, scope, success criteria, brief §-alignment) | ✅ | — |
| Architecture (ADRs, layer boundaries, public API shape) | ✅ | — |
| State / action / reward design choices | ✅ | — |
| Test acceptance criteria + the assertions that must hold | ✅ | — |
| Final code-review sign-off + commit-message intent | ✅ | — |
| Self-score / grade claim against the rubric | ✅ | — |
| Code generation against an approved spec | — | ✅ |
| Refactoring within an existing public API | — | ✅ |
| Test scaffolding + boilerplate from a written spec | — | ✅ |
| Docstring / notebook-prose drafts (human edits before commit) | — | ✅ |
| Routine doc maintenance (link fixes, freshness sweeps) | — | ✅ |
| Lint / format auto-fixes | — | ✅ |

**Operating rule.** If any AI-generated change would alter a human-decided
column above (e.g. add a new public SDK method, change a test assertion, weaken
a quality gate, choose between two architectures), the human must sign off
explicitly *before* the code lands — typically by approving the PRD/PLAN edit
first, *then* letting the AI execute against it.

This contract is also evidenced in `docs/shared/PROMPTS.md` (the literal prompts used)
and in the per-section commit messages that name the § of the submission
guidelines being addressed.

## Hard Constraints (Apply to ALL Files)

### 1. File Size Limit — 150 Lines Maximum

Every Python file (`.py`) must not exceed 150 lines of code. If a file
approaches 150 lines, split into separate modules.

### 2. Test-Driven Development (TDD)

Write tests BEFORE implementation. RED → GREEN → REFACTOR. All new code must
achieve **≥ 85 %** test coverage (statement + branch).

Run: `uv run pytest tests/ --cov=src --cov-report=term-missing`

### 3. Object-Oriented Programming (OOP)

Use inheritance where it pays — `BaseAgent → PPOAgent`, `BaseReward →
ModularityReward / CohesionReward / CouplingPenalty`. No code duplication;
shared logic in base classes. **`RefactorSDK` is the single business-logic
entry point** for all UIs (CLI, GUI, notebook). UIs never import services /
model / env directly.

### 4. No Hardcoded Values

ALL **algorithm-relevant** parameters, rewards, and thresholds live in
`config/config.yaml` and are accessed via the config loader. This covers: PPO
hyperparameters (clip_eps, n_steps, batch_size, n_epochs, lr, vf_coef,
ent_coef, max_grad_norm), GAE (λ, γ), reward weights (α, β, γ, P_skills),
centrality-call budget, seeds, padding caps, and the convergence criterion.

Local UI-styling literals (button pixel dimensions, dashboard line offsets,
matplotlib `alpha` / `fontsize` / `dpi` values) stay in their rendering
modules. The test for "should this be in config" is: *"would I expect a
grader, contributor, or future-me to ever want to change this without
editing source?"* If yes → config. If no → keep it local.

### 5. No Code Duplication (DRY)

Extract common logic into shared methods/classes. If a pattern appears twice,
create a utility or base class method.

### 6. Linting — Zero Ruff Violations

Run: `uv run ruff check src/ tests/ scripts/`. Must produce zero errors
before every commit.

### 7. Package Manager — UV Only

Use `uv` exclusively. No pip, no conda.
Run app: `uv run python -m src.cli`
Install deps: `uv sync --dev`

## Algorithm Requirements (brief ex04)

### PPO + GAE (the only RL algorithm in this assignment)

- **PPO clip ε = 0.2 — FIXED** by ex04 §2.3. Asserted in
  `tests/architecture/test_ppo_constants.py`. Do not tune.
- **GAE λ = 0.95 — FIXED** by ex04 §2.3. Asserted in the same test.
  Do not tune.
- γ = 0.99 (configurable but defaulted in `config/config.yaml`).
- Clipped surrogate objective (Schulman 2017 eq. 7):
  `L^CLIP(θ) = Ê_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]`.
- GAE advantage (Schulman 2015 eq. 16):
  `Â_t = Σ_{l=0..∞} (γλ)^l · δ_{t+l}` with `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`.

### Custom Training Loop (NOT gym.Env)

- The env is a Python object that yields `(state, action_mask, reward, done)`
  but is **not** registered as a `gym.Env`. The PPO trainer lives in
  `src/services/ppo_trainer.py` and owns rollout collection, GAE computation,
  minibatch SGD, and logging. Asserted in `tests/architecture/test_no_gym_env.py`.
- Stable-Baselines3 is allowed by the brief for the comparison baseline only,
  and the SB3 path gets a 2-hour timebox spike with a padding/masking
  fallback (ADR-003).

### Skills-only scope lock

- The graph builder must reject any module path not under the Skills layer.
  Asserted in `tests/architecture/test_skills_only_scope.py` — any non-Skills
  node in a built graph fails the test.

### Centrality discipline

- **Degree** centrality is cheap and may be recomputed per step (it ships in
  the observation).
- **Betweenness** is O(|V|·|E|) and is computed only at episode start,
  episode end, and the final-evaluation sweep — i.e. exactly
  `centrality.betweenness_calls_per_seed = 3` calls per seed.
  Asserted in `tests/architecture/test_centrality_discipline.py`.

### Reward (brief ex04 §2.3)

`r_t = α·ΔModularity_t + β·ΔCohesion_t − γ·ΔCoupling_t + P_skills·1[lazy_load_broken]`

with α = 1.0, β = 1.0, γ = 0.5, P_skills = −5.0, all in `config.yaml`.
The lazy-load-break detector walks `sys.modules` after each step and
asserts the P95 token budget; any regression triggers `P_skills`.

## Comparison Requirements (brief ex04 §2.4 / §2.5)

Generate the ablation matrix:
- α/β/γ + P_skills full ablation grid, **mean ± 1σ + 95 % CI** over
  **≥ 5 seeds per cell** (`seeds: [42, 7, 123, 314, 271]` in config).
- Convergence is the **dual criterion**: rolling-100-episode reward within
  ±2 % drift for 50 consecutive episodes AND policy entropy below
  `convergence.entropy_threshold`.
- Betweenness reported once per seed × ≥ 5 seeds, mean ± std + 95 % CI.
- Tokens reported as tiktoken `cl100k_base` headline + chars/bytes appendix.

Save plots as PNG in `results/figures/` and ablation CSVs in
`results/ablation/`.

## Honest Reporting Rules

- No "achieves", "solves", "demonstrates", "earns the same ~XX" phrasings.
- Every numeric claim names: seed, episode count, mean ± std.
- README and PRD carry an "Honest Limitations" section listing the
  PythonClaw shim caveat (ADR-001 24h swap window), the Skills-only scope
  restriction, and the single-codebase generalisation gap.

## Version Control

- New repository for A4 (NOT the same as A1/A2/A3). Branch: `main`.
- Initial version: `1.2.0` on the assignment-4 deliverable tag.
- Commit-message convention: `<Phase N|Phase 0 bootstrap|chore: bootstrap>:
  <imperative summary>`.
- Co-author trailer required on every Claude-generated commit:
  `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.

## Config Structure

All config in `config/config.yaml`:
- `ppo` — clip_eps (FIXED 0.2), n_steps, batch_size, n_epochs, lr, vf_coef,
  ent_coef, max_grad_norm
- `gae` — lambda (FIXED 0.95), gamma
- `reward` — alpha, beta, gamma, p_skills
- `centrality` — betweenness_calls_per_seed (= 3)
- `seeds` — list of ≥ 5 seeds for paired comparisons
- `training` — max_episodes + convergence (rolling_window, reward_drift_pct,
  consecutive_episodes, entropy_threshold)
- `environment` — max_nodes_v (padding cap)
- `paths` — vault_dir, figures_dir, ablation_dir
- `version` — semantic version string

## Pre-Submission Review Methodology

Before submission (or whenever asked "is this ready?"), walk the 9-phase
iterative pre-submission audit. The methodology and per-phase self-critique
prompts live under `instructions/review_methodology/` (gitignored, local
only — do NOT reference in shipped docs).

The methodology is reusable across assignments. Update its phase files only
when the *process* improves, not when the assignment content changes.
