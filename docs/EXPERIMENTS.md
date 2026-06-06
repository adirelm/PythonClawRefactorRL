# Experiments

> **Phase 4 fill — populated as each planned experiment is executed.**
> Every row maps to a reproducible run: pinned seeds {42, 7, 123, 314, 271},
> PPO with ε=0.2 (fixed), GAE λ=0.95 (fixed), γ=0.99, and the canonical
> reward `R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t`
> with α=1.0, β=1.0, γ=0.5, P_skills=−5.0.

## Planned experiments (per Phase)

| ID | Hypothesis | Setup | Result | Verdict |
|---|---|---|---|---|
| P0-E1 | <hypothesis> | <setup: corpus, seeds, hyperparams, budget> | <result: metric + 95% CI> | <Verdict: supported / refuted / inconclusive> |
| P1-E1 | <hypothesis> | <setup> | <result> | <verdict> |
| P1-E2 | <hypothesis> | <setup> | <result> | <verdict> |
| P2-E1 | <hypothesis> | <setup> | <result> | <verdict> |
| P2-E2 | <hypothesis> | <setup> | <result> | <verdict> |
| P3-E1 | 5-seed PPO smoke | Original `71f0213`: 2/5 seeds done (42, 7); 123/314/271 hung on `Categorical(-inf)`. Rework R1 `5dd14ca` landed NOOP-slot defensive fallback. Rework R2 retrain (`uv run scripts/train_ppo.py --total-steps 1000`) stopped by run-window hook mid-run: only seed_42 completed (final_reward=-0.44, btw_calls=2); seed_7 mid-rollout; 123/314/271 never started. | 1/5 seeds on disk post-R2; no `aggregate.json`; R1 NOOP fix UNVALIDATED on hang-prone seeds; INCONCLUSIVE_RETRAIN_STILL_OWED (resolution: re-run retrain to completion so seeds 7/123/314/271 land and aggregate regenerates with num_seeds=5, dof=4) |
| P3-E2 | <hypothesis> | <setup> | <result> | <verdict> |
| P3-E3 | <hypothesis> | <setup> | <result> | <verdict> |
| P4-E1 | <hypothesis> | <setup> | <result> | <verdict> |
| P4-E2 | <hypothesis> | <setup> | <result> | <verdict> |

## Conventions

- **ID**: `P<phase>-E<n>`; do not renumber once an experiment has run.
- **Hypothesis**: one sentence, falsifiable, names the metric it moves.
- **Setup**: corpus snapshot, seed list, hyperparams that differ from
  the canonical defaults above, and compute budget.
- **Result**: point estimate + 95% CI across the seed set; link to
  artefact under `results/experiments/<ID>/`.
- **Verdict**: supported / refuted / inconclusive, plus a one-line
  reason. Inconclusive rows must list what additional evidence would
  resolve them.

## Convergence rule (ADR-010)

A run is considered converged when, on non-overlapping windows
A=[t-200:t-100] and B=[t-100:t] with step=100:

- reward criterion: `|mean(B) − mean(A)| / |mean(A)| ≤ 0.02`, and
- entropy slope: `|dH/dt| < entropy_slope_threshold`
  (slope of the policy entropy, NOT an absolute floor).

Any row whose result depends on convergence must cite the step at which
both criteria first hold.
