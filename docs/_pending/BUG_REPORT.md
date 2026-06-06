# Bug Report

> **Phase 4 fill — populated from training rollout traces per ADR-007 instrumentation.**
> This file holds the structured post-mortem of the two reproducible bugs surfaced
> during PPO training. Each entry is tied back to a step ID in the rollout log so
> the symptom can be re-played deterministically (seed-pinned per ADR-010).

## Scope

- Source: live PPO rollouts (≥5 seeds {42, 7, 123, 314, 271}).
- Detection: reward-component spikes and invariant violations captured by ADR-007 hooks.
- Reward formula referenced below:
  `R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t`
  with defaults α=1.0, β=1.0, γ=0.5, P_skills=−5.0.

---

## Bug 1: <title>

- **Symptom**: <observed behavior — e.g., reward drops by >X% between adjacent steps; invariant Z violated>
- **Step ID**: <rollout step / episode index from ADR-007 trace>
- **Reward Δ**: <component-level delta: ΔModularity / ΔCohesion / Coupling_Penalty / P_skills>
- **Root cause**: <module + commit hash + minimal explanation>
- **Fix**: <patch summary, link to PR / commit, regression test path>

---

## Bug 2: <title>

- **Symptom**: <observed behavior>
- **Step ID**: <rollout step / episode index from ADR-007 trace>
- **Reward Δ**: <component-level delta>
- **Root cause**: <module + commit hash + minimal explanation>
- **Fix**: <patch summary, link to PR / commit, regression test path>

---

## Cross-references

- ADR-007 (rollout instrumentation): `docs/adr/ADR-007-*.md`
- Convergence windows (ADR-010): non-overlapping A=[t-200:t-100], B=[t-100:t], step=100.
- Trace artefacts: `results/traces/seed_<n>/rollout.jsonl` (Phase 4 generated).
