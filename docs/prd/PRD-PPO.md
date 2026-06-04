# PRD-PPO — Proximal Policy Optimization Component

Status: Draft v0.2
Owner: human architect (per CLAUDE.md §1.4)
Master reference: `docs/PRD.md` §3.3 (PPO + GAE trainer, brief §2.3).

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
| F4.1 | Clip epsilon `eps` | **FIXED 0.2** (Schulman 2017) — no schedule, no tuning |
| F4.2 | Learning rate | Linear `3e-4 -> 0` over total timesteps (SB3 default) |
| F4.3 | `n_steps` | 2048 per env (SB3 default) |
| F4.4 | `batch_size` | 64 (SB3 default) |
| F4.5 | `n_epochs` | 10 (SB3 default) |
| F4.6 | `gamma` | 0.99 (brief §2.3) |
| F4.7 | GAE `lambda` | 0.95 (Schulman 2016 Table 2 range; SB3 default; mandated by brief §2.3 — NOT a paper-recommended single value) |
| F4.8 | Value-fn clipping | **Enabled**, same `eps=0.2` (SB3 default; Engstrom 2020 ablation) |
| F4.9 | Entropy coef `c2` | **0.0 (configurable)** — see §3 + ADR-010 entropy-slope note |
| F4.10 | Max grad norm | 0.5 (SB3 default, NOT Schulman 2017) |
| F4.11 | Value coef `c1` | 0.5 (SB3 default, NOT Schulman 2017) |
| F4.12 | Seeds | >=5 per ablation cell (ADR-006) |

All values live in `config/config.yaml` per CLAUDE.md §4.

**Citation discipline.** Of the canonical PPO hyperparameters, only
`eps=0.2` (F4.1) is attributed to Schulman 2017. `c1=0.5`, value-fn
clipping (F4.8), and `max_grad_norm=0.5` are **SB3 implementation
defaults**, supported by the Engstrom 2020 ablation
("Implementation Matters in Deep Policy Gradients", arXiv:2005.12729 /
ICLR 2020) — not by the original PPO paper. `lambda=0.95` is within
Schulman 2016's Table 2 swept range but is not paper-recommended as
a single value; it is mandated here by brief §2.3 and matches the SB3
default.

## 3. L^CLIP Loss

Verbatim from Schulman et al., 2017, **Eq. (7)**:

```
L^CLIP(θ) = Ê_t [ min( r_t(θ) Â_t,
                       clip(r_t(θ), 1 − ε, 1 + ε) Â_t ) ]
```

with ratio `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` and GAE-λ
advantage estimate `Â_t`. `ε = 0.2` (F4.1).

The full PPO objective (Schulman 2017 Eq. 9, with SB3's value-fn
clipping toggled per F4.8) is:

```
L(θ) = Ê_t [ L^CLIP_t(θ) − c1 · L^VF_t(θ) + c2 · S[π_θ](s_t) ]
```

- `c1 = 0.5` (F4.11) — SB3 default, **not** from Schulman 2017.
- `c2 = 0.0` (F4.9) — configurable. The dual-convergence acceptance
  criterion in §5.4 uses an **entropy-slope** check (ADR-010),
  **not** an absolute entropy floor, so `c2` does not need to be
  positive for ADR-010 to hold. If `c2` is raised to e.g. `0.01`,
  the slope check is unaffected — this avoids the circular
  dependency where an absolute floor would have to be re-derived
  from `c2`.
- `S[π_θ](s_t)` is the policy entropy at state `s_t`.

## 4. Why PPO (not vanilla PG, not TRPO)

- **Vanilla PG:** no trust region; unstable on noisy refactor rewards.
- **TRPO:** hard KL via CG + line search; correct but expensive and
  awkward to wrap around graph-conditioned policies.
- **PPO:** first-order clipped ratio approximates the trust region,
  keeps SGD-friendly updates, matches Schulman et al., *Proximal
  Policy Optimization Algorithms*, arXiv:1707.06347, 2017.

Matches brief §2.3 and Lecture 7.

## 5. Acceptance Criteria

Evaluated over the final 20% of training, averaged across >=5 seeds:

1. **Clip-fraction sanity.** `clip_fraction` in the typical `[0.1, 0.3]`
   band. Values outside this range are **flagged for review** (not an
   automatic reject) — e.g. very low `clip_fraction` may indicate `ε`
   is too loose for the observed advantage scale, very high may
   indicate policy is updating too aggressively. Decision is a human
   call per CLAUDE.md §1.4.
2. **KL bounded.** Mean `approx_kl <= 0.02` per update; no single
   update exceeds `0.05`.
3. **Value loss converges.** `explained_variance >= 0.6` and rolling
   `value_loss` slope <= 0.
4. **Dual convergence (ADR-010).** Two non-overlapping windows:
   - Window A: episodes `[t-200 : t-100]` (100 episodes)
   - Window B: episodes `[t-100 : t]` (100 episodes, step=100)
   Reward criterion: `|mean(B) − mean(A)| / |mean(A)| <= 0.02`.
   Entropy criterion: rolling **entropy slope** `|dH/dt|` below the
   config-defined threshold over the same window (per ADR-010 — this
   replaces the earlier absolute-floor formulation, which coupled the
   criterion to `c2` and to the unknown entropy scale of the
   skill-conditioned action space).

Criteria 1 is advisory; criteria 2–4 are blocking for the ablation
matrix.

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

- Master PRD: `docs/PRD.md` §3.3 (PPO + GAE trainer).
- Brief: `instructions/assignment-4/ex04.pdf` §2.3.
- Lecture 7: `instructions/assignment-4/lec7_ppo.pdf`.
- Schulman, Wolski, Dhariwal, Radford, Klimov, 2017, "Proximal Policy
  Optimization Algorithms", **arXiv:1707.06347** — source of L^CLIP
  (Eq. 7) and `ε = 0.2`.
- Schulman, Moritz, Levine, Jordan, Abbeel, 2016, "High-Dimensional
  Continuous Control Using Generalized Advantage Estimation",
  **arXiv:1506.02438** (ICLR 2016) — GAE derivation; `λ = 0.95` is
  inside the Table 2 swept range.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry, 2020,
  "Implementation Matters in Deep Policy Gradients: A Case Study on
  PPO and TRPO", **arXiv:2005.12729** (ICLR 2020) — empirical evidence
  that SB3-style code-level optimizations (value-fn clipping,
  `max_grad_norm`, advantage normalization, LR annealing) carry a
  large share of PPO's real-world performance.
- Stable-Baselines3 PPO docs: https://stable-baselines3.readthedocs.io/
  en/master/modules/ppo.html — source of `c1=0.5`, value-fn clipping,
  `max_grad_norm=0.5`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`.
- ADR-001 (PythonClaw shim boundary), ADR-002 (GRAPHIFY adapter),
  ADR-006 (multi-seed eval discipline), ADR-010 (dual-convergence
  criterion — entropy-slope formulation).
