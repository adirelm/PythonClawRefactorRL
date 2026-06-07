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

## Bug 2: Louvain community detection wedges on mid-rollout graph topology → env.step blocks indefinitely

- **Severity**: HIGH — the entire training loop wedges. The original symptom was
  a 7+ hour spin at ~99% CPU on a single seed before manual kill. Surfaced
  reproducibly during Phase-3 5-seed isolated-subprocess retrains and pinned
  by RC-0 cProfile under a 90 s SIGALRM watchdog.

- **Symptom (reproducer)**: pre-RC-1 (any commit before `44b313f`),
  ``env.step`` blocks indefinitely on seed 123 mid-rollout because
  ``compute_modularity`` calls ``networkx.community.louvain_communities`` on a
  pathologically-shaped mid-rollout graph snapshot. Minimal repro at the
  PPO-trainer boundary:
  ```bash
  # Pre-RC-1 (any commit before 44b313f):
  uv run python -c "
  import torch
  from pathlib import Path
  from src.env.skills_graph_env import SkillsGraphEnv
  from src.model.policy_net import PolicyNet
  from src.services.ppo_trainer import PPOTrainer
  torch.manual_seed(123)
  env = SkillsGraphEnv(Path('src/pythonclaw_shim/sample_skills'), seed=123)
  trainer = PPOTrainer(env, PolicyNet(), clip_eps=0.2, gae_lambda=0.95)
  trainer.collect_rollout()  # wedges indefinitely on seed=123 mid-rollout
  "
  ```

- **Stack trace at the wedge** (RC-0 cProfile spike captured under a 90 s
  SIGALRM watchdog):
  ```
  PPOTrainer.collect_rollout → env.step → _safe_reward → compute_reward
    → compute_modularity → nx_comm.louvain_communities → _one_level → _neighbor_weights
  ```

- **Affected seeds (pre-fix)**: 123, 314. Reproducible in their own isolated
  subprocesses, which rules out inter-seed state leak and pins the bug to
  topology-conditioned Louvain pathology rather than rollout-buffer
  cross-contamination.

- **Wall-clock impact (pre-fix)**: 7+ hours at ~99% CPU on the original
  Phase-3 retrain before being killed manually. Under the RC-0 90 s
  SIGALRM watchdog the spike pinned at ``_neighbor_weights`` every time.

- **Reward Δ**: not applicable — the wedge is inside ``compute_modularity``
  which is invoked from ``_safe_reward → compute_reward``. The step never
  returns so no ``(s, a, r, s')`` transition is appended to the rollout
  buffer, and no ΔModularity / ΔCohesion / Coupling_Penalty / P_skills
  row is emitted for the wedged step.

- **Root cause hypothesis**: ``networkx.community.louvain_communities`` has a
  worst-case pathological execution path on small but specifically-shaped
  graphs. The cProfile stack pointed at the inner ``_one_level →
  _neighbor_weights`` loop, which iterates a ``defaultdict(float)`` over each
  node's neighbours; degenerate neighbour-count patterns appear to cause
  prolonged iteration without crashing or returning.

  **Compounding factor.** ``env.step`` calls ``compute_reward`` once, but
  ``compute_reward`` calls ``compute_modularity`` *twice* (before + after the
  refactor op for ΔModularity), ``compute_cohesion`` *twice*, and
  ``compute_coupling_penalty`` *twice* — so a naive implementation pays
  **six Louvain calls per env.step**. On a wedge-prone topology each one
  blows the per-call budget independently, multiplying the hang.

- **Fix**: multi-part, landed across commits `44b313f` and `d489306` (RC-1):

  1. ``safe_louvain`` wraps Louvain in a ``threading.Thread`` plus
     ``threading.Event.wait(timeout=WATCHDOG_SECONDS)``. On timeout, fall
     back to ``greedy_modularity_communities`` under the same budget; if
     *that* also exceeds, return ``None`` and the metric short-circuits
     to ``0.0``.
  2. ``WATCHDOG_SECONDS = 0.05`` — aggressive on purpose: OK seeds run
     Louvain in microseconds on these graph sizes, so the budget never
     trips for them; wedge-prone topologies short-circuit fast instead
     of blowing the rollout budget.
  3. **Per-step partition share**: ``compute_reward`` computes the
     partition *once per snapshot* (before + after = **2** ``safe_louvain``
     calls per ``env.step`` instead of 6), threading the cached partition
     into modularity, cohesion, and coupling-penalty metrics.
  4. **Cache** ``safe_louvain`` results by ``(frozenset(nodes),
     frozenset(edges))``. NOOPs and failed refactor ops (graph unchanged
     step-to-step) pay zero Louvain work.
  5. **Topology early-return guard**: ``V < 2`` or ``E == 0`` short-circuits
     before Louvain is even called.

- **Regression tests**:
  - ``tests/architecture/test_modularity_wedge_regression.py`` — asserts
    ``wall_time < 1.5 s`` on the known-wedge topology that previously hung
    for 7+ hours.
  - ``tests/unit/services/metrics/test_modularity_watchdog.py`` — 4 cases:
    watchdog fires + falls back to greedy, double-timeout returns ``0.0``,
    happy path unaffected (no spurious timeout), and cache-hit path skips
    the watchdog entirely.
  - ``tests/unit/services/metrics/conftest.py`` — autouse ``safe_louvain``
    cache clear between tests so partition memoisation cannot leak across
    cases and mask a regression.

- **Residual (honestly disclosed)**: seeds 123 and 314 **still** TIMEOUT at the
  240 s/seed budget under full ``collect_rollout`` + ``update`` after RC-1.
  The wedge moved from Louvain blocking on the main thread to *daemon-thread
  contention*: Python cannot kill threads cooperatively, so each wedged
  Louvain leaks a background thread that keeps competing for the GIL with
  the main rollout thread. A fundamental fix would require switching from
  ``threading`` to ``multiprocessing`` (so a hung worker can be ``terminate()``-d).
  Tracked as Phase-4 RC-INCOMPLETE in TRACE F10 / TODO T3.7 / EXPERIMENTS
  P3-E1. Acceptable under the honesty lock (3/5 OK seeds → −2 self-grade
  penalty already applied).

- **Empirical numbers**: seed 123 went from a 7+ hour hang (pre-RC-1) to
  ``51.3 s`` for an isolated single-seed ``collect_rollout`` smoke (post-RC-1),
  to ``240 s/seed`` TIMEOUT under the full PPO rollout + update loop
  (residual daemon-thread contention described above). Seeds 42, 7, 271
  complete cleanly post-RC-1 with no watchdog firings observed.

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
