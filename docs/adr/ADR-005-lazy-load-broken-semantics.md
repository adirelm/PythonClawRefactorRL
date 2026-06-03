# ADR-005: Operational Semantics of "Broken Lazy-Loading"

- **Status:** Accepted
- **Date:** 2026-06-04
- **Deciders:** Architect (human), implementer (AI)
- **Supersedes:** —
- **Related:** ADR-001 (PythonClaw shim), ADR-003 (tiktoken cost),
  ADR-004 (convergence dual criterion), OQ-5

## Context

Open Question OQ-5 surfaced during brief review: §2.4 names a
**"Skills-protection penalty"** that fires when the agent "breaks lazy
loading," but the brief never tells us what *breaks* means. Treated as
folklore the rule is ungradeable — every reviewer can read it
differently, and the RL reward signal becomes a coin flip whose payout
depends on the marker's intuition rather than a measurable property of
the produced code.

Two reading exist in the wild:

1. **Import-graph reading.** Touching a Skills submodule should not
   pull in code that wasn't declared as a dependency of *that*
   submodule. If it does, lazy loading has degraded into eager loading
   through a side-channel (e.g. a top-level `import torch` inside a
   helper that was only supposed to expose a constant).
2. **Token-budget reading.** A Skills call that suddenly costs many
   more tokens than the historical baseline implies the cache missed
   and the module was re-fetched in full — which is exactly the
   failure mode lazy loading is supposed to prevent.

Both readings are real failures. Picking one and ignoring the other
leaves a class of regressions invisible to the reward function.

## Decision

**"Broken lazy-loading" is operationally defined as the disjunction
(logical OR) of two measurable conditions. The monitor returns `True`
(i.e. *broken*) when ANY of the following holds:**

- **(a) Import-graph leak.** Importing a Skills submodule causes more
  than one transitive import to appear outside the submodule's
  declared dependency list. Measured by a pytest fixture that
  snapshots `sys.modules` before and after the import and diffs against
  the declared deps recorded in `src/skills/_deps.py`.
- **(b) Token re-fetch.** The tiktoken `cl100k_base` count of a
  Skills call exceeds the P95 of the rolling historical baseline
  (window = last 100 calls, seed-stratified). Excess implies the
  cached lazy view was invalidated and the full module was re-shipped
  through the boundary.

Either condition firing is sufficient. Both firing in the same step
still counts as a single break (no double-penalty), to keep the reward
signal stationary.

## Implementation

- **`src/services/lazy_load_monitor.py`** (≤150 LOC) exposes a single
  `LazyLoadMonitor` class with one public method:
  `is_broken(call_record) -> bool`. The class is constructed with the
  declared-deps map and the rolling P95 estimator, and is consumed by
  the reward function through a thin adapter so the RL code never
  touches `sys.modules` directly.
- **`tests/architecture/test_lazy_load_invariants.py`** owns the
  import-graph fixture. It parametrizes over every Skills submodule,
  snapshots `sys.modules`, imports the submodule, diffs, and asserts
  the diff is a subset of the declared deps. The same test file
  exercises the token branch with synthetic call records at, below,
  and above the P95 threshold.
- The monitor emits a **boolean signal** per environment step. The
  reward function reads that boolean and applies `P_skills` exactly
  once per `True`.

## Penalty Magnitude

`P_skills = -5.0`, set in `config/config.yaml` under
`rewards.skills_protection_penalty`. The value is intentionally
large relative to per-step shaping rewards so that a *single* break
dominates the episode return — lazy loading is a correctness
invariant, not a soft preference. The exact magnitude is tuned in the
α/β/γ ablation matrix (≥5 seeds per cell) and reported with the rest
of the sensitivity grid; the *sign and order of magnitude* are fixed
by this ADR, the *exact value* is empirical.

## Consequences

- The reward function gains one extra term, gated by a cheap
  monitor call (one set diff, one float comparison per step).
- Skills authors must keep `src/skills/_deps.py` honest; the import-
  graph test will catch undeclared transitive imports at CI time, not
  at training time.
- The P95 baseline is **seed-stratified**: each RL seed maintains its
  own rolling window so a noisier seed doesn't unfairly trip the
  token branch for a quieter one.
- The boolean signal is logged alongside cost rows (ADR-003 schema),
  so post-hoc analysis can attribute reward drops to import leaks vs.
  token re-fetches without re-running training.

## Alternatives Considered

- **Import-graph only.** Rejected: misses cache-miss regressions that
  don't change which modules are imported, only how much of them
  crosses the boundary.
- **Token-budget only.** Rejected: misses structural regressions where
  a single small helper silently pulls a heavy dependency at import
  time but the per-call token count looks normal.
- **Continuous penalty (scaled by excess).** Rejected: makes the
  reward non-stationary across seeds with different baselines and
  invites reward hacking by sitting just under the threshold.
- **Soft penalty (-0.5).** Rejected: too small to dominate shaping;
  the agent learns to pay the fee rather than fix the leak.
