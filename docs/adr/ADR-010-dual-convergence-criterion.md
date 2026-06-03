# ADR-010: Dual-Criterion Convergence Definition (Reward Stability AND Entropy Floor)

- **Status:** Accepted
- **Date:** 2026-06-04
- **Deciders:** Architect (human), implementer (AI)
- **Supersedes:** —
- **Related:** ADR-006 (multi-seed eval discipline), ADR-007 (reward MUST),
  PRD-PPO, OQ-10

## Context

Open Question OQ-10 surfaced during the grade-strategy round: §2 of the
ex04 brief requires us to declare *when* a training run has "converged"
before quoting any headline number (final reward, ablation delta,
encoder win-rate). The literature gives us at least three operational
definitions and they disagree often enough that the choice is itself
an architectural decision, not a hyperparameter:

1. **Rolling-mean reward stability** — `mean(rewards[-100:])` changes by
   less than ±2% across a sliding window of 50 episodes.
2. **Policy-entropy floor** — `H(π_θ)` drops below a fixed threshold
   (here `0.5` for a small discrete action space), indicating the policy
   has committed to a near-deterministic mode.
3. **Gradient-norm collapse** — `‖∇L‖₂` stabilises; rejected upfront
   because PPO's clipped objective makes this noisy and seed-dependent.

Lessons from A2 (DQN) and A3 (A2C):

- **Reward-only** convergence fired early on "lucky" seeds where reward
  plateaued at a local maximum while the policy was still high-entropy
  and would have escaped given more steps. Headline numbers were
  inflated by ~7–12% versus a longer-run baseline.
- **Entropy-only** convergence fired when entropy collapsed onto a
  *wrong* policy (e.g. always-merge in the skills environment), making
  the "converged" agent strictly worse than a random baseline.

Either criterion alone is a false-positive generator. Both together
constrain the failure mode to genuinely-stalled training, which is the
behaviour we want to detect.

## Decision

**Convergence is declared if and only if BOTH conditions hold at the
same training step `t`:**

1. **Reward stability.** `|mean(rewards[t-100:t]) - mean(rewards[t-150:t-50])| ≤
   0.02 · |mean(rewards[t-100:t])|` for 50 consecutive episodes.
2. **Entropy floor.** `H(π_θ(·|s_t)) < 0.5` averaged over the most
   recent rollout batch.

Both thresholds — the `±2%` reward band, the `50`-episode dwell time,
and the `0.5` entropy ceiling — live in `config/config.yaml` under
`convergence:` and are **not** hardcoded in `src/services/convergence.py`.
The service exposes `is_converged(history) -> bool` and
`why(history) -> ConvergenceVerdict` (a dataclass naming which criterion
fired, which did not, and at which step).

If only **one** criterion holds, the run is logged as
`PARTIAL_CONVERGENCE` and **must not** appear in headline tables; it is
permitted in appendix plots with a hatched "partial" marker.

## Justification

- **Eliminates the A2/A3 false-positive class.** Reward-only and
  entropy-only failure modes are exactly the modes both prior
  assignments tripped on. The dual gate closes both at once.
- **Cheap to compute.** Reward stability is a 100-window mean diff;
  entropy is already logged for the PPO loss. No extra rollouts.
- **Auditable.** `ConvergenceVerdict` makes the grader's life easy:
  every "converged" claim in the report can be traced to a verdict
  dataclass dump in `results/training/verdicts/seed_N.json`.
- **Honest about partial runs.** The `PARTIAL_CONVERGENCE` class
  prevents us from quietly cherry-picking seeds that only satisfied
  one criterion — they are visibly hatched in plots.

## Consequences

- `src/services/convergence.py` is a ≤150-LOC module owned by the RL
  subsystem. Stateless functions; the caller passes the reward and
  entropy history slices.
- `tests/unit/test_convergence_dual_criterion.py` covers:
  - both criteria hold → `True`,
  - reward-only → `False` and verdict explains why,
  - entropy-only → `False` and verdict explains why,
  - neither → `False`,
  - off-by-one at the 50-episode dwell boundary.
- `config/config.yaml` gains a `convergence:` block with explicit
  thresholds and dwell time; changing the band/floor does not require
  a code edit.
- Reports and PRD-PPO carry the phrase "converged under the dual
  criterion of ADR-010" wherever a final-reward number is quoted, so
  the definition travels with the claim.
- The `results/training/verdicts/` directory is committed (small JSON
  files, one per seed × algorithm) and indexed by TRACE.md.

## Alternatives Considered

- **Reward-only stability.** Rejected: documented false-positive mode
  from A2/A3 retrospectives.
- **Entropy-only floor.** Rejected: same, opposite direction.
- **Three-of-three including gradient-norm collapse.** Rejected: PPO
  clip noise makes the gradient-norm signal seed-unstable; adds cost
  without information gain at our scale.
- **Statistical test (Mann-Kendall trend).** Considered for the reward
  channel; rejected as over-engineered for a ≤150-LOC service and
  hard for a grader to audit without re-deriving the test.
