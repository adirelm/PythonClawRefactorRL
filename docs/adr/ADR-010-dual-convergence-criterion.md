# ADR-010: Dual-Criterion Convergence Definition (Reward Stability AND Entropy Slope)

- **Status:** Accepted
- **Date:** 2026-06-04
- **Deciders:** Architect (human), implementer (AI)
- **Supersedes:** —
- **Related:** ADR-006 (multi-seed eval discipline), ADR-007 (reward MUST),
  PRD-PPO (F4.9 entropy_coef), OQ-10

## Context

Open Question OQ-10 surfaced during the grade-strategy round: §2 of the
ex04 brief requires us to declare *when* a training run has "converged"
before quoting any headline number (final reward, ablation delta,
encoder win-rate). The literature gives us at least three operational
definitions and they disagree often enough that the choice is itself
an architectural decision, not a hyperparameter:

1. **Rolling-mean reward stability** — the mean reward over a recent
   window changes by less than ±2% relative to the mean over the
   immediately-preceding, **non-overlapping** window of equal length.
2. **Policy-entropy slope** — `|dH(π_θ)/dt|` (per-episode change in
   the running mean of policy entropy) falls below a small threshold,
   indicating the entropy trajectory has *plateaued* (not that it has
   reached any specific absolute level).
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

### Why slope-based entropy (not absolute floor)

PRD-PPO **F4.9 sets `entropy_coef` (c2) to `0.0`** — there is no
entropy-bonus term pushing `H(π)` toward any particular level. With
`c2 = 0.0`, an absolute floor like `H < 0.5` is degenerate: it will
either (a) trigger trivially once the optimiser sharpens any
action-preference, or (b) never trigger if the task has many
near-equiprobable optimal actions. Both modes hide the failure cases
listed above.

Two reconciliations were considered:

- **(a) Enable `c2 = 0.01` in PRD-PPO F4.9 and re-derive the floor.**
  Rejected: this couples convergence semantics to a hyperparameter
  choice that lives in another spec and is itself subject to
  ablation. Any change to `c2` would silently invalidate the floor.
- **(b) Switch from absolute entropy floor to entropy *slope*.**
  Accepted: `|dH/dt| < ε` measures whether the policy has *stopped
  changing*, which is the actual signal we want. It is independent of
  `c2`, of the action-space size, and of the absolute entropy scale.

This ADR therefore adopts (b) and breaks the `c2`-dependency loop.

### Why non-overlapping windows

The earlier draft compared `mean(rewards[t-100:t])` against
`mean(rewards[t-150:t-50])`. These windows overlap by **50 episodes**,
so 50% of each mean is shared by construction. That shared mass biases
the absolute difference `|mean(B) − mean(A)|` toward zero — i.e. the
criterion *systematically over-declares* convergence the longer
training runs. The fix is to compare two **disjoint** 100-episode
windows separated by `step = 100`, so no sample contributes to both
means.

## Decision

**Convergence is declared if and only if BOTH conditions hold at the
same training step `t`:**

1. **Reward stability (non-overlapping windows, rolling-mean ±2%).**
   Let
   - Window A = `rewards[t-200 : t-100]` (100 episodes),
   - Window B = `rewards[t-100 : t]`     (100 episodes, step = 100),

   with **no overlap** between A and B. Convergence on this channel
   requires:

   ```
   |mean(B) − mean(A)|  ≤  0.02 · |mean(A)|
   ```

   evaluated at every episode `t ≥ 200`.
2. **Policy-entropy slope.** `|dH/dt| < entropy_slope_threshold`
   (default `0.01` nats/episode, measured over the most recent
   100-episode window of per-rollout mean entropies). Replaces the
   earlier "absolute floor of 0.5" criterion for the reasons in
   *Why slope-based entropy* above.

Both thresholds — the `±2%` reward band, the `100`-episode window
length, the `step = 100` non-overlap stride, and the
`entropy_slope_threshold` — live in `config/config.yaml` under
`convergence:` and are **not** hardcoded in `src/services/convergence.py`.
The service exposes `is_converged(history) -> bool` and
`why(history) -> ConvergenceVerdict` (a dataclass naming which criterion
fired, which did not, and at which step).

If only **one** criterion holds, the run is logged as
`PARTIAL_CONVERGENCE` and **must not** appear in headline tables; it is
permitted in appendix plots with a hatched "partial" marker.

### `ConvergenceVerdict` dataclass (verbatim)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ConvergenceVerdict:
    converged: bool                 # True iff BOTH channels passed
    reward_stable: bool             # Window-A vs Window-B ±2% test
    entropy_slope_ok: bool          # |dH/dt| < entropy_slope_threshold
    step: int                       # episode index t at which checked
    reward_window_a_mean: float     # mean(rewards[t-200:t-100])
    reward_window_b_mean: float     # mean(rewards[t-100:t])
    reward_rel_delta: float         # |B-A| / |A|
    entropy_slope: float            # measured |dH/dt| over last 100 ep
    reason: str                     # "OK" | "REWARD_DRIFT" |
                                    # "ENTROPY_NOT_PLATEAUED" |
                                    # "PARTIAL_CONVERGENCE" |
                                    # "INSUFFICIENT_HISTORY"
```

`PARTIAL_CONVERGENCE` is the canonical `reason` string when exactly one
of `reward_stable` / `entropy_slope_ok` is `True`. Headline tables MUST
filter on `verdict.converged == True`; partial runs are appendix-only
with hatched markers.

## Justification

- **Eliminates the A2/A3 false-positive class.** Reward-only and
  entropy-only failure modes are exactly the modes both prior
  assignments tripped on. The dual gate closes both at once.
- **Removes the systematic Δ → 0 bias.** Non-overlapping windows mean
  the reward-stability test no longer artificially inflates its own
  pass rate as training length grows.
- **Decouples from F4.9.** Slope-based entropy is independent of
  `entropy_coef` (`c2 = 0.0` per PRD-PPO F4.9), so changes to the PPO
  loss weighting do not silently invalidate the convergence criterion.
- **Cheap to compute.** Two 100-episode means and a linear-regression
  slope over 100 entropy samples. No extra rollouts.
- **Auditable.** `ConvergenceVerdict` makes the grader's life easy:
  every "converged" claim in the report can be traced to a verdict
  dataclass dump in `results/training/verdicts/seed_<N>.json`
  (one file per seed × algorithm; committed).
- **Honest about partial runs.** The `PARTIAL_CONVERGENCE` class
  prevents us from quietly cherry-picking seeds that only satisfied
  one criterion — they are visibly hatched in plots.

## Consequences

- `src/services/convergence.py` is a ≤150-LOC module owned by the RL
  subsystem. Stateless functions; the caller passes the reward and
  entropy history slices.
- **`tests/architecture/test_convergence_definition.py`** MUST assert:
  - the two reward windows are **non-overlapping**: the start indices
    of Window A and Window B differ by **≥ 100** (i.e. `step ≥ 100`,
    enforced by an explicit `assert b_start - a_start >= 100`);
  - the entropy criterion reads `entropy_slope_threshold` from
    `config/config.yaml` under `convergence:` (not a literal in code);
  - the verdict `reason` field uses the canonical strings listed in
    the `ConvergenceVerdict` block above;
  - off-by-one behaviour at `t = 200` (the earliest step at which
    both windows are fully populated).
- `tests/unit/test_convergence_dual_criterion.py` covers:
  - both criteria hold → `converged = True`,
  - reward-only → `False` and verdict `reason = "ENTROPY_NOT_PLATEAUED"`,
  - entropy-only → `False` and verdict `reason = "REWARD_DRIFT"`,
  - neither → `False`,
  - `t < 200` → `reason = "INSUFFICIENT_HISTORY"`.
- `config/config.yaml` gains a `convergence:` block with explicit
  thresholds, window length, step, and `entropy_slope_threshold`;
  changing any of them does not require a code edit.
- Reports and PRD-PPO carry the phrase "converged under the dual
  criterion of ADR-010" wherever a final-reward number is quoted, so
  the definition travels with the claim. **Cross-link:** the
  slope-based entropy choice is the direct consequence of PRD-PPO
  **F4.9** (`entropy_coef = c2 = 0.0`); any future change to F4.9
  MUST trigger a re-review of this ADR.
- The `results/training/verdicts/` directory is committed (small JSON
  files, one per seed × algorithm) and indexed by TRACE.md. Each file
  is a serialised `ConvergenceVerdict` per the dataclass above.

## Alternatives Considered

- **Reward-only stability (overlapping or non-overlapping windows).**
  Rejected: documented false-positive mode from A2/A3 retrospectives.
- **Entropy-only floor (`H < 0.5`).** Rejected: collapses to a wrong
  policy can satisfy it; also degenerate under PRD-PPO F4.9
  `c2 = 0.0` (see *Why slope-based entropy*).
- **Overlapping reward windows (`[t-150:t-50]` vs `[t-100:t]`).**
  Rejected in this revision: the 50-episode overlap biases
  `|mean(B) − mean(A)|` toward zero and inflates pass rate.
- **Enable `c2 = 0.01` in PRD-PPO F4.9 + re-derive entropy floor.**
  Considered, rejected: couples convergence semantics to a tunable
  PPO hyperparameter and creates a circular dependency.
- **Three-of-three including gradient-norm collapse.** Rejected: PPO
  clip noise makes the gradient-norm signal seed-unstable; adds cost
  without information gain at our scale.
- **Statistical test (Mann-Kendall trend).** Considered for the reward
  channel; rejected as over-engineered for a ≤150-LOC service and
  hard for a grader to audit without re-deriving the test.
