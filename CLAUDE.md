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
model / env directly. **Developer tooling under `scripts/` is explicitly
exempt**: it is not a UI surface but reproduction/experiment glue (training
drivers, figure renderers, corpus collectors) and may import `src.*` internals
directly. The single-entry rule binds the user-facing surfaces (CLI + notebook),
which is where it is verified.

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

- **PPO clip ε = 0.2 — FIXED** by ex04 §2.3 (Phase 1 follow-up: dedicated
  arch test for the constant is not yet in `tests/architecture/`). Do not tune.
- **GAE λ = 0.95 — FIXED** by ex04 §2.3 (same Phase 1 follow-up).
  Do not tune.
- γ = 0.99 (configurable but defaulted in `config/config.yaml`).
- Clipped surrogate objective (Schulman 2017 eq. 7):
  `L^CLIP(θ) = Ê_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]`.
- GAE advantage (Schulman 2015 eq. 16):
  `Â_t = Σ_{l=0..∞} (γλ)^l · δ_{t+l}` with `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`.

### Custom Training Loop (NOT gym.Env)

- Brief §2.2 verbatim BANS Gymnasium (Hebrew: ללא סביבת Gymnasium). The env
  is a Python object yielding `(state, action_mask, reward, done)` but is
  NEVER registered as a `gym.Env`. NO `gymnasium` import allowed under
  `src/env/` or `src/services/`. AST-level enforcement:
  `tests/architecture/test_env_no_gym.py` asserts no `ImportFrom 'gymnasium'`
  node in the tree.
- The PPO trainer lives in `src/services/ppo_trainer.py` and owns rollout
  collection, GAE computation, minibatch SGD, and logging.
- Stable-Baselines3 is allowed by the brief for the comparison baseline only,
  and the SB3 path gets a 2-hour timebox spike with a padding/masking
  fallback (ADR-003).

### Skills-only scope lock

- The graph builder must reject any module path not under the Skills layer.
  Phase 1 follow-up: a dedicated arch test for this scope lock is not yet in
  `tests/architecture/`; until then the lock is enforced inside the
  `GraphifyAdapter.build()` path itself.

### Centrality discipline

- **Degree** centrality is cheap and may be recomputed per step (it ships in
  the observation).
- **Betweenness** is O(|V|·|E|) and is computed exactly twice per seed
  (start + end ONLY) per brief §2.2 — NO "final-eval" third call.
  `centrality.betweenness_calls_per_seed = 2`. Enforced by
  `tests/architecture/test_betweenness_call_count.py`.

### Reward (brief ex04 §2.3)

`R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t`

with α = 1.0, β = 1.0, γ = 0.5, P_skills = −5.0 (NEGATIVE penalty), all in
`config.yaml` under `reward.p_skills`. Canonical form matches ADR-007 and is
asserted by `tests/architecture/test_reward_formula.py`.
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
- Initial version `1.2.0` (`v1.2.0` tag); `1.3.0` (`v1.3.0` — brief-§3 per-metric
  improvement curves + P4-E3 extended convergence experiment); current `1.4.0`
  (`v1.4.0` — fixes the action-mask↔env slot-resolution bug so MERGE/REWIRE apply
  the masked-legal partner, retrains all results, wires every PPO hyperparameter
  from config, and adds Skills-module architectural bugs to the §3 report).
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
- `centrality` — betweenness_calls_per_seed (= 2; start + end ONLY per brief §2.2)
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
