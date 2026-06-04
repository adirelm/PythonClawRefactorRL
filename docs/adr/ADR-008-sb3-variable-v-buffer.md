# ADR-008: SB3 Variable-|V| Replay Buffer Strategy

- **Status:** Proposed (spike-gated)
- **Date:** 2026-06-04
- **Deciders:** Architect (human), Implementer (Codex)
- **Related:** ADR-002 (GraphifyAdapter), ADR-004 (GraphSAGE primary
  path), STATE_DESIGN §4 (conditional padding), CLAUDE.md §CANONICAL
  "V_max policy" + §A4 (2h spike timebox), OQ-9 (variable-|V| obs into
  SB3), grade-strategy "week-3 discovery here is unrecoverable"

## Context

Our observation space is a code-graph whose node count |V| varies per
episode and per step (extract → refactor → re-extract). Stable-Baselines3
on-policy buffers assume a fixed `gym.spaces` shape. Naively flattening
GraphSAGE node embeddings into a fixed tensor erases the topology signal
the policy is supposed to learn from.

The grade-strategy round flagged this as the single highest-risk OQ: if we
hit this in week 3, the schedule cannot absorb a redesign. We therefore
gate the decision on a **2-hour timeboxed spike on Day 2** before any
trainer code is written against an assumed buffer shape. Per CANONICAL
V_max policy, `V_max = 512` is a **fallback cap**, not the primary path
(ADR-004 GraphSAGE accepts variable-|V| via PyG DataLoader; padding-to-512
applies only if SB3 RolloutBuffer cannot handle Dict obs).

## Decision

**Spike-then-fallback.** We do not preemptively commit to a buffer
architecture. On Phase 1, Day 2, we run a **2-hour timeboxed spike**
(`notebooks/00_buffer_spike.ipynb`) that probes three candidate paths
**in declared order**:

1. **(a) Dict obs + custom `features_extractor`** — preferred path,
   preserves ADR-004 GraphSAGE size-independence.
2. **(b) Pad-to-`V_max=512` + boolean action mask + flat MLP** —
   fallback path, gated by this ADR and STATE_DESIGN §4's CONDITIONAL
   `V_max` cap.
3. **(c) Custom PPO wrapper over SB3 primitives** — last-resort path,
   budgeted at +1 day beyond the spike.

The first path whose pass criteria (defined per-path below) are met
inside the 2-hour window wins. If (a) fails → fall back to (b); if (b)
also fails → escalate to (c). Outcome is recorded in a post-spike
amendment that flips this ADR from **Proposed (spike-gated)** to
**Accepted (path X)** and unblocks Phase 3 trainer work.

Until the amendment lands, **no trainer code may assume a buffer
shape**, and `V_max = 512` remains a CONDITIONAL fallback per
STATE_DESIGN §4 — neither retired nor in force.

## Decision Rationale

Preemptively committing to a buffer architecture before the spike is
worse than spike-gating, because the three candidate paths impose
**mutually incompatible** constraints on the rest of the stack:

- Path (a) requires `gym.spaces.Dict` observations and a custom
  `features_extractor`. ADR-004's GraphSAGE size-independence claim
  survives intact; `V_max` is retired.
- Path (b) requires a fixed-width flat tensor with a boolean action
  mask. ADR-004's size-independence claim is **broken** by the
  padding, and STATE_DESIGN §4's CONDITIONAL `V_max = 512` becomes
  the binding cap (with graphs |V| > 512 dropped from the corpus
  and disclosed in §2.4).
- Path (c) keeps GraphSAGE intact but forces us to maintain a
  hand-rolled PPO collect-rollouts loop, with a 1e-6 numerical
  equivalence bar against stock SB3.

The information value of running the spike — i.e. learning *which*
of these three regimes our stack actually lives in — strictly
dominates committing in advance, because:

1. **Cost of being wrong is asymmetric.** Choosing (b) preemptively
   and discovering in week 3 that (a) would have worked is recoverable
   only by re-doing all training runs (≥5 seeds × baselines).
   Choosing (a) preemptively and discovering it does not boot is the
   "week-3 discovery is unrecoverable" failure mode the grade-strategy
   round explicitly flagged.
2. **Cost of the spike is bounded.** 2 hours on Day 2, with a single
   notebook artifact and a written amendment. No production code is
   written against the spike's assumptions.
3. **Downstream contracts depend on the outcome.** ADR-004
   (GraphSAGE primary path) and STATE_DESIGN §4 (CONDITIONAL `V_max`)
   both have branches that activate or retire based on which spike
   path wins. Committing now would either invalidate ADR-004 (if we
   pick b) or invalidate STATE_DESIGN §4's conditional clause (if we
   pick a) — without evidence.

The spike is therefore the **cheapest** way to resolve the
architectural fork, and the only mechanism that keeps ADR-004 and
STATE_DESIGN §4 honest until the evidence is in.

## Spike (Day 2, 2-hour timebox per CLAUDE.md §A4)

Notebook: `notebooks/00_buffer_spike.ipynb`.

Three candidate paths are explored, **in order**, until one passes the
acceptance bar or the timebox expires.

### Path (a) — Custom `features_extractor` + Dict obs — **PREFERRED**

Subclass `BaseFeaturesExtractor`, accept `Dict({nodes, edges, mask})`
as the observation space, pool GraphSAGE node embeddings inside the
extractor. SB3 buffer stores Dict tensors natively; topology lives in
the extractor, not the space. Preserves ADR-004's size-independence
claim because the extractor consumes variable-|V| through PyG batching.

**Pass criteria (a) — all must hold:**

1. SB3 PPO trains 100 steps on Dict obs without raising an exception.
2. Advantage tensor shape matches `(n_envs, n_steps)` after rollout.
3. Gradients flow end-to-end (non-zero `.grad` on extractor params).
4. P95 step latency ≤ 50 ms on the spike laptop.

### Path (b) — Pad-to-V_max=512 + boolean action mask + MLP-on-flat — **FALLBACK**

Right-pad node features to `V_max = 512` with zeros, expose a boolean
action mask so PPO never selects a pad slot, encode with a flat MLP.
Closest to stock SB3; loses inductive GraphSAGE elegance.

**Pass criteria (b) — both must hold:**

1. Padding + masking does **not** change the policy distribution
   vs. an unpadded baseline by more than **5%** (KL or per-action
   probability delta) on a fixed-|V| toy graph corpus.
2. Boots a 100-step rollout without shape errors and clears the
   ≤ 50 ms P95 latency bar.

### Path (c) — Custom PPO wrapper over SB3 low-level primitives — **LAST RESORT**

Reuse `RolloutBuffer` primitives but write our own `collect_rollouts`
that handles variable-|V| natively. Highest engineering cost
(**~1 day**, exceeds the spike timebox — only entered post-spike if
both (a) and (b) fail).

**Pass criteria (c) — both must hold:**

1. Hand-rolled PPO matches SB3 PPO output within **1e-6** on identical
   seed, identical rollout, identical hyperparameters.
2. Clears the ≤ 50 ms P95 latency bar.

## Fallback selection rule

**If (a) fails → default to (b). If (b) also fails → escalate to (c).
Document outcome in an ADR-008 amendment post-spike.** Do not iterate
past the 2-hour timebox; partial wins do not count. Path (c) is
explicitly budgeted at +1 day beyond the spike if reached.

## Cross-links

- **ADR-004 (GraphSAGE, primary path).** Reconciles the size-independence
  claim: paths (a) and (c) preserve it; path (b) breaks it and forces
  the V_max cap.
- **STATE_DESIGN §0 + §4 (conditional padding).** V_max marked
  conditional there; this ADR is the gate that flips it from
  "potential" to "in force" (path b) or retires it (paths a / c).
- **CLAUDE.md §CANONICAL "V_max policy".** Single source of truth for
  the fallback rule; this ADR is the operational expansion.

## Trade-offs of the fallback (path b)

- ✅ Unblocks PPO training on stock SB3 immediately.
- ✅ Action masking still prevents illegal refactor targets.
- ❌ Loses inductive generalisation across graph sizes (the GraphSAGE
  selling point in §2.4).
- ❌ Caps the corpus to graphs with |V| ≤ 512; larger files are
  dropped from training with a logged warning.
- ⚠️ Must be disclosed in the §2.4 essay and the results section as
  a scope reduction, not silently absorbed.

## Consequences

- **Schedule.** Day 2 is reserved for the spike. Trainer scaffolding
  (Phase 3) cannot start until ADR-008 is amended with the spike
  outcome.
- **Interface.** Whichever path wins, the GraphifyAdapter (ADR-002)
  exposes obs in a single canonical shape. The policy never sees raw
  PythonClaw output.
- **Reversibility.** All three paths preserve the SB3 PPO trainer
  surface; only the `features_extractor` and `observation_space`
  change. We can swap between paths post-spike without rewriting the
  training loop, at the cost of re-running the seed sweep.

## Amendment policy

This ADR is **proposed**, not accepted. After the spike, append an
"Amendment — Spike Outcome" section that records:

1. Which path was attempted, in which order.
2. The minute mark at which each path passed or was abandoned.
3. The chosen path forward, with a one-line justification.
4. Any deviation from the fallback (e.g. V_max raised/lowered).

Only after that amendment is committed does ADR-008 move to **Accepted**
and unblock Phase 3 trainer work.
