---
doc_id: PRD-GAE
version: 1.0.0
status: Draft
owner: A4 Architect
linked_to: PRD-Master.md §3 F5
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

The advantage at step `t` is computed in **reverse-time order** over the rollout:

$$A_t = \delta_t + \gamma \lambda A_{t+1}, \qquad A_T = 0$$

## 3. TD-Residual Formula

Each per-step temporal-difference residual is defined as:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

where `V(·)` is the critic network shipped with the SB3 PPO actor-critic.

## 4. Why GAE vs Single-Step TD vs Monte Carlo

GAE is an **exponentially-weighted average** of `k`-step advantage estimators
controlled by λ. It explicitly interpolates the classic variance/bias tradeoff:

| Estimator             | Bias       | Variance   | Behaviour                                                |
|-----------------------|------------|------------|----------------------------------------------------------|
| Single-step TD(0)     | **High**   | **Low**    | Bootstraps off a possibly-wrong `V(s_{t+1})`             |
| Monte Carlo return    | **Low**    | **High**   | Uses the full discounted return — unbiased but noisy     |
| **GAE(λ)**            | Tunable    | Tunable    | Smooth interpolation; λ=0.95 chosen per Schulman 2016    |

At λ=0.95 we keep most of the bias-reduction benefit of multi-step returns while
retaining a usable variance level for stable PPO updates — this is the default
recommended in the GAE paper for continuous-control workloads.

## 5. Acceptance Criteria

- **AC-1** — Advantage normalization is preserved: advantages are standardized
  to mean 0 and standard deviation 1 across the minibatch *before* each policy
  update (this is SB3 PPO's `normalize_advantage=True` default).
- **AC-2** — Setting λ→0 reduces the estimator to **TD(0)**:
  `A_t = δ_t = r_t + γ V(s_{t+1}) − V(s_t)`.
- **AC-3** — Setting λ→1 reduces the estimator to the **Monte Carlo** advantage:
  `A_t = Σ_{k=0..T−t} γ^k r_{t+k} − V(s_t)`.
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

This PRD realizes **F5** in `docs/prd/PRD-Master.md §3` (functional
requirements table). Any change to λ, γ, or the normalization rule above
requires a coordinated edit to PRD-Master §3 F5 and a new PRD-GAE minor
version.
