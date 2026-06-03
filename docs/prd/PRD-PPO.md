# PRD-PPO — Proximal Policy Optimization Component

Status: Draft v0.1
Owner: human architect (per CLAUDE.md §1.4)
Master reference: `docs/prd/PRD-MASTER.md` §3 F4.

## 1. Purpose

PPO is the policy-optimization core of Assignment 4. It consumes
state from GRAPHIFY (PRD-GRAPHIFY) and the skill-conditioned action
space from PRD-SKILLS, and outputs a policy that refactors Python
source under the configured reward signal.

Satisfies brief `§2.3 (Algorithm: PPO with GAE)` and the derivation
in `Lecture 7 (PPO + GAE)`. Vanilla PG and TRPO are out of scope as
primary optimizers.

## 2. Functional Requirements

| ID | Requirement | Value |
|----|-------------|-------|
| F4.1 | Clip epsilon | **FIXED 0.2** (no schedule, no tuning) |
| F4.2 | Learning rate | Linear `3e-4 -> 0` over total timesteps |
| F4.3 | `n_steps` | 2048 per env |
| F4.4 | `batch_size` | 64 |
| F4.5 | `n_epochs` | 10 |
| F4.6 | `gamma` | 0.99 |
| F4.7 | GAE `lambda` | 0.95 |
| F4.8 | Value-fn clipping | **Enabled**, same eps=0.2 |
| F4.9 | Entropy coef | 0.0 |
| F4.10 | Max grad norm | 0.5 |
| F4.11 | Seeds | >=5 per ablation cell |

All values live in `config/config.yaml` per CLAUDE.md §4.

## 3. L^CLIP Loss

With ratio `r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)`
and GAE-lambda advantage `A_t`:

```
L^CLIP(theta) = E_t [ min( r_t * A_t,
                           clip(r_t, 1 - eps, 1 + eps) * A_t ) ]
```

`eps = 0.2`. Full objective adds clipped value loss and entropy:

```
L(theta) = E_t [ L^CLIP_t - c1 * L^VF_t + c2 * S[pi_theta](s_t) ]
```

`c1 = 0.5`, `c2 = 0.0` (see F4.9).

## 4. Why PPO (not vanilla PG, not TRPO)

- **Vanilla PG:** no trust region; unstable on noisy refactor rewards.
- **TRPO:** hard KL via CG + line search; correct but expensive and
  awkward to wrap around graph-conditioned policies.
- **PPO:** first-order clipped ratio approximates the trust region,
  keeps SGD-friendly updates, matches Schulman et al., *Proximal
  Policy Optimization Algorithms*, arXiv:1707.06347, 2017.

Matches brief §2.3 and Lecture 7.

## 5. Acceptance Criteria

Accepted iff **all** hold over the final 20% of training, averaged
across >=5 seeds:

1. **Clip-fraction sanity.** `clip_fraction in [0.10, 0.30]`.
2. **KL bounded.** Mean `approx_kl <= 0.02` per update; no single
   update exceeds `0.05`.
3. **Value loss converges.** `explained_variance >= 0.6` and rolling
   `value_loss` slope <= 0.
4. **Dual convergence (locked).** Rolling-100 reward within +/-2%
   for 50 consecutive episodes AND entropy below the config threshold.

Failure on any criterion blocks the run from the ablation matrix.

## 6. Implementation Path — SB3 Wrapper

Per the locked grade-strategy decision, PPO is **not** reimplemented.
`src/ppo/ppo_adapter.py` (<=150 LOC) wraps `stable_baselines3.PPO`:

- Accepts `GraphifyAdapter` observations (ADR-002).
- Accepts skill-conditioned action space (PRD-SKILLS).
- Exposes `train`, `predict`, `save`, `load`.
- Logs `clip_fraction`, `approx_kl`, `explained_variance`,
  `value_loss`, `policy_gradient_loss`, `entropy_loss` per update.
- 2h timebox spike verifies SB3 buffer fits the graph-state shape;
  padding/masking is the documented fallback.

## 7. References

- Master PRD: `docs/prd/PRD-MASTER.md` §3 F4.
- Brief: `instructions/assignment-4/ex04.pdf` §2.3.
- Lecture 7: `instructions/assignment-4/lec7_ppo.pdf`.
- Schulman et al., 2017, arXiv:1707.06347.
- ADR-001, ADR-002.
