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

## Bug 1: Categorical(logits=all_-inf) NaN-explosion when action_mask is all-False

- **Severity**: HIGH — the entire PPO rollout collector wedges with no graceful
  recovery. The training process spins at ~100% CPU on the sampler thread until
  a wall-clock timeout kills it. Observed on 3 of 5 canonical seeds.

- **Symptom (reproducer)**: pre-R1 (commit `71f0213`, before the fix at `5dd14ca`),
  ``PolicyNet.get_action`` would feed an all-``-inf`` row into
  ``torch.distributions.Categorical``, which yields NaN probabilities and a
  hanging ``.sample()`` call. Minimal repro at the policy-net boundary:
  ```bash
  uv run python -c "
  import torch
  from torch.distributions import Categorical
  # Pre-R1 code path inside PolicyNet.get_action:
  logits = torch.zeros(1, 45057)
  mask   = torch.zeros(1, 45057, dtype=torch.bool)   # all-False action mask
  masked = logits.masked_fill(~mask, float('-inf'))  # → all -inf row
  dist   = Categorical(logits=masked)
  print(dist.probs)         # NaN, NaN, NaN, ...
  print(dist.sample())      # spins / hangs / explodes
  "
  ```

- **Step ID**: surfaced reproducibly during the Phase-3 5-seed isolated-subprocess
  retrain (commit `8f96e30`). Seeds 123, 314, 271 wedged at the very first PPO
  rollout step, at the policy-net `Categorical(logits=...).sample()` call —
  before any environment reward was emitted, so there is no ADR-007 reward-delta
  trace; the diagnostic signal is the wall-clock hang itself plus the all-False
  mask captured under the watchdog. Seeds 42 and 7 completed normally because
  their initial graph topologies kept SPLIT / MERGE / REWIRE legal at every
  rollout step they touched.

- **Reward Δ**: not applicable — the hang occurs in the actor before any
  ``(s,a,r,s')`` transition is appended to the rollout buffer. No
  ΔModularity / ΔCohesion / Coupling_Penalty / P_skills row is produced.

- **Root cause**: an ``action_mask`` edge case during high-entropy exploration on
  degenerate sub-graphs. The legal-action filters in
  ``src/env/action_mask.py`` reject every slot when:
  - SPLIT requires children ≥ k + 2 (Louvain k-way) — false on small components,
  - MERGE requires |V| ≥ 2 — false on single-node graphs,
  - REWIRE requires |E| ≥ 1 — false on edge-empty graphs,
  and the pre-R1 code path inside ``PolicyNet.get_action`` did **not** pin the
  NOOP slot True when the resulting row was all-False. The downstream
  ``masked_fill(~action_mask, float('-inf'))`` then produced an all-``-inf``
  logits row, which ``Categorical(logits=...)`` silently turns into NaN probs
  (softmax of all ``-inf`` is ``0/0``), and ``dist.sample()`` enters an
  infinite retry on NaN probability tensors. Pre-fix commit: `71f0213`.

- **Fix**: commit `5dd14ca`, in
  ``src/model/policy_net.py`` lines 108–117 (verified against the current
  source). The defensive guard runs **before** ``masked_fill``:
  ```python
  # src/model/policy_net.py:113-117
  safe_mask = action_mask
  empty_rows = (~action_mask).all(dim=-1)
  if bool(empty_rows.any()):
      safe_mask = action_mask.clone()
      safe_mask[..., _NOOP_IDX_DEFAULT] = (
          safe_mask[..., _NOOP_IDX_DEFAULT] | empty_rows
      )
  masked = logits.masked_fill(~safe_mask, float("-inf"))   # line 118
  ```
  This preserves the Huang & Ontañón (2022) §3 invalid-action-masking guarantee
  (NOOP is the always-legal escape slot per ACTION_DESIGN §2.4 and is already
  pinned True inside ``src/env/action_mask.py:139``), and is belt-and-suspenders
  for any future code path that constructs a mask without going through
  ``compute_mask``.

- **Regression tests**: ``tests/architecture/test_policy_net_categorical_safe.py``
  — 4 cases, all green:
  1. ``test_all_false_mask_does_not_hang_and_returns_noop`` — pins the
     failure-mode row; asserts ``action_idx == NOOP_INDEX`` and finite
     ``log_prob`` instead of a hang.
  2. ``test_single_true_mask_is_deterministic`` — guard is a no-op on
     healthy single-slot masks; that slot is sampled deterministically.
  3. ``test_normal_mask_distribution_is_finite_and_sums_to_one`` — softmax
     over legal slots is finite and normalised after the guard.
  4. ``test_mixed_batch_with_one_empty_row_still_samples_finitely`` — NaN
     from one empty row does not contaminate healthy rows in the same batch.

- **Affected seeds (pre-fix)**: 123, 314, 271 hung at the first rollout step
  of the ~5 000-step PPO budget. Post-fix the partition-memo + watchdog
  work in commit `d489306` (RC-1) closed the remaining daemon-thread
  contention path for seeds 123 / 314 — they are still tracked honestly
  as Phase-4 RC-INCOMPLETE under the architect lock (3 OK + 2 TIMEOUT
  is the canonical Wave-1 retrain outcome; see TRACE F10 / TODO T3.7 /
  EXPERIMENTS P3-E1).

- **Post-fix smoke**: NOOP-only and all-False masks both return
  ``action_idx = 45056`` with finite ``log_prob`` (= 0.0) in < 0.002 s on the
  reference machine. 114 passed + 1 skipped across
  ``tests/unit/model`` + ``tests/architecture`` + ``tests/unit/services``
  after the fix.

---

## Bug 2: <title>

> **Placeholder — TODO_ARCHITECT_VERIFY (Wave 3).** Wave-3 will investigate the
> second reproducible bug and fill this stub. Candidate sources under the
> Phase-4 honesty lock: the Louvain modularity-wedge regression caught by
> ``tests/architecture/test_modularity_wedge_regression.py`` (commit `44b313f`),
> or a reward-component delta isolated from the new ``results/traces/seed_<n>/
> rollout.jsonl`` ADR-007 traces. Selection criterion: pick the candidate whose
> reproducer is fully self-contained in source (no live PPO rollout required)
> so the entry, like Bug 1, can be replayed deterministically without a
> multi-hour training run.

- **Symptom**: <observed behavior>
- **Step ID**: <rollout step / episode index from ADR-007 trace>
- **Reward Δ**: <component-level delta>
- **Root cause**: <module + commit hash + minimal explanation>
- **Fix**: <patch summary, link to PR / commit, regression test path>

---

## Cross-references

- ADR-007 (rollout instrumentation): `docs/adr/ADR-007-reward-upgrade-MUST.md`
- Convergence windows (ADR-010): non-overlapping A=[t-200:t-100], B=[t-100:t], step=100.
- Trace artefacts: `results/traces/seed_<n>/rollout.jsonl` (Phase 4 generated).
- Regression test (Bug 1): `tests/architecture/test_policy_net_categorical_safe.py`
- Fix commit (Bug 1): `5dd14ca` — *fix(policy): guarantee NOOP slot in action
  mask — kills Categorical(-inf) hang on seeds 123/314/271*.
- Pre-fix commit (Bug 1): `71f0213` — Phase 3 baseline where the hang
  reproduces.
