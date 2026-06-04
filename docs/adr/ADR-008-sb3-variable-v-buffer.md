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
