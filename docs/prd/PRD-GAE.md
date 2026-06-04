---
doc_id: PRD-GAE
version: 1.0.2
status: Draft
owner: A4 Architect
linked_to: docs/PRD.md §3.3
---

# PRD-GAE — Generalized Advantage Estimation Component

## 1. Purpose

This PRD defines the **Generalized Advantage Estimation (GAE)** component of the
A4 PythonClaw Refactor RL pipeline. GAE provides the advantage signal consumed
by the PPO policy-gradient update.

Source authority:

- Assignment brief **§2.3** — mandates a low-variance advantage estimator for
  the policy-gradient learner driving the refactor agent.
- **Schulman et al. 2016**, *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation*, arXiv:1506.02438 — the canonical GAE
  reference and source of the recurrence and λ semantics used here.

## 2. Functional Requirements

| ID    | Requirement                                                     | Value      |
|-------|-----------------------------------------------------------------|------------|
| FR-1  | GAE smoothing parameter λ — **FIXED** for all experiments       | `0.95`     |
| FR-2  | Discount factor γ                                               | `0.99`     |
| FR-3  | Recurrence over a rollout segment of length `T`                 | see §2.1   |
| FR-4  | λ is exposed via `config/config.yaml` under `ppo.gae_lambda`    | read-only at runtime |
| FR-5  | γ is exposed via `config/config.yaml` under `ppo.gamma`         | read-only at runtime |

### 2.1 Recurrence

The advantage at step `t` is computed in **reverse-time order** over the
rollout, with a **terminal mask** `(1 − done_t)` that zeroes the bootstrap
across episode boundaries (matches
`stable_baselines3.common.buffers.RolloutBuffer.compute_returns_and_advantage`):

$$A_t = \delta_t + \gamma \lambda (1 - \text{done}_t) A_{t+1}, \qquad A_T = 0$$

The mask ensures that when step `t` is terminal (`done_t = 1`), the
advantage does not propagate from the next episode's first step — i.e. we
do **not** bootstrap across episode boundaries. This matches SB3's
`RolloutBuffer.compute_returns_and_advantage` and Schulman 2016 §3.

## 3. TD-Residual Formula

Each per-step temporal-difference residual is defined with the same
**terminal mask** `(1 − done_t)` as the recurrence in §2.1, so the
bootstrap term `V(s_{t+1})` is zeroed at episode boundaries:

$$\delta_t = r_t + \gamma (1 - \text{done}_t) V(s_{t+1}) - V(s_t)$$

where `V(·)` is the critic network shipped with the SB3 PPO actor-critic.
The mask matches `stable_baselines3.common.buffers.RolloutBuffer.compute_returns_and_advantage`
and Schulman 2016 §3 (no bootstrap into the next episode).

## 4. Why GAE vs Single-Step TD vs Monte Carlo

GAE is an **exponentially-weighted average** of `k`-step advantage estimators
controlled by λ. It explicitly interpolates the classic variance/bias tradeoff:

| Estimator             | Bias       | Variance   | Behaviour                                                |
|-----------------------|------------|------------|----------------------------------------------------------|
| Single-step TD(0)     | **High**   | **Low**    | Bootstraps off a possibly-wrong `V(s_{t+1})`             |
| Monte Carlo return    | **Low**    | **High**   | Uses the full discounted return — unbiased but noisy     |
| **GAE(λ)**            | Tunable    | Tunable    | Smooth interpolation; λ=0.95 is within Schulman 2016 Table 2 range, is the SB3 default, and is mandated by brief §2.3 |

At λ=0.95 we keep most of the bias-reduction benefit of multi-step returns
while retaining a usable variance level for stable PPO updates. To be
precise about attribution: **the GAE paper (Schulman 2016) does not
specifically "recommend" λ=0.95** — it sweeps λ ∈ [0.9, 0.99] in Table 2
and reports λ=0.95 among the better-performing settings on
continuous-control benchmarks. Our use of λ=0.95 here is justified by
three converging reasons:

1. It sits in the **Schulman 2016 Table 2 range** of empirically-good values.
2. It is the **stable-baselines3 default** for `gae_lambda` (so we inherit
   the SB3 community's tuning history rather than introducing a new value).
3. It is **mandated by the assignment brief §2.3** for the policy-gradient
   learner driving the refactor agent.

## 5. Acceptance Criteria

- **AC-1** — Advantage normalization is preserved: advantages are standardized
  to mean 0 and standard deviation 1 across the minibatch *before* each policy
  update (this is SB3 PPO's `normalize_advantage=True` default).
- **AC-2** — Setting λ→0 reduces the estimator to **TD(0)**:
  `A_t = δ_t = r_t + γ V(s_{t+1}) − V(s_t)`.
- **AC-3** — Setting λ→1 reduces the estimator to the **Monte Carlo**
  advantage. With `done_T = 1` and the terminal mask from §2.1, the
  recurrence telescopes over rewards `r_t .. r_{T-1}` (NOT `r_T`, which
  is past the terminal step), so the correct upper bound is `T−1−t`:
  `A_t = Σ_{k=0..T−1−t} γ^k r_{t+k} − V(s_t)`.
  This matches Schulman 2016 Eq. 14.
- **AC-4** — The recurrence in §2.1 is implemented in **reverse-time order**
  across the rollout buffer (a forward-time implementation is a bug).
- **AC-5** — λ and γ are not hardcoded in source; both are loaded from
  `config/config.yaml` at PPO construction time.

## 6. Implementation Note

GAE ships natively inside **stable-baselines3** PPO
(`stable_baselines3.common.buffers.RolloutBuffer.compute_returns_and_advantage`).
A4 does **not** re-implement GAE. Instead we:

1. Expose `λ` as `ppo.gae_lambda` in `config/config.yaml`.
2. Expose `γ` as `ppo.gamma` in `config/config.yaml`.
3. Pass both values through to `stable_baselines3.PPO(...)` at construction.
4. Cover AC-1 through AC-5 via unit tests that mock the rollout buffer and
   assert the limiting behaviours (λ→0, λ→1) and the normalization step.

No custom CUDA / numpy GAE kernel is in scope for A4.

## 7. Reference

This PRD realizes the **GAE half of the PPO + GAE trainer** described in
`docs/PRD.md §3.3` (functional requirement F5 — custom training loop —
plus the PPO+GAE trainer narrative in §3.3 "PPO + GAE trainer, brief §2.3").
Any change to λ, γ, the normalization rule, or the terminal-mask formula
above requires a coordinated edit to `docs/PRD.md §3.3` and a new
PRD-GAE minor version.

The advantages produced here feed the PPO update whose **convergence
detection** is specified in **ADR-010** (non-overlapping windows
A=[t−200:t−100] / B=[t−100:t], step=100; reward criterion
`|mean(B) − mean(A)| / |mean(A)| ≤ 0.02`; entropy convergence by **slope**
`|dH/dt| < entropy_slope_threshold`, not an absolute floor). The choice
of λ=0.95 directly affects the variance of the advantage signal that
ADR-010's reward-window statistic operates on, so any future change to
λ here must be reviewed against ADR-010's window size and threshold.

## 8. References

- **Schulman et al. 2016**, *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation*, **arXiv:1506.02438**. Canonical GAE
  reference; source of the recurrence in §2.1 (with terminal mask per
  §3 of that paper), the TD-residual in §3 above, the λ→1 Monte-Carlo
  limit in AC-3, and the λ ∈ [0.9, 0.99] empirical sweep in Table 2.
- **Schulman et al. 2017**, *Proximal Policy Optimization Algorithms*,
  arXiv:1707.06347. PPO consumer of the GAE advantages produced here.
- **stable-baselines3** —
  `stable_baselines3.common.buffers.RolloutBuffer.compute_returns_and_advantage`
  is the canonical implementation A4 uses; it includes the `(1 − done_t)`
  terminal mask in both `delta_t` and the recurrence, matching §2.1 / §3.
