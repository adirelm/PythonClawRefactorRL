# ADR-008: SB3 Variable-|V| Replay Buffer Strategy

- **Status:** Proposed (spike-gated)
- **Date:** 2026-06-04
- **Deciders:** Architect (human), Implementer (Codex)
- **Related:** OQ-9 (variable-|V| obs into SB3), grade-strategy "week-3
  discovery here is unrecoverable", ADR-002 (GraphifyAdapter)

## Context

Our observation space is a code-graph whose node count |V| varies per
episode and per step (extract → refactor → re-extract). Stable-Baselines3
on-policy buffers assume a fixed `gym.spaces` shape. Naively flattening
GraphSAGE node embeddings into a fixed tensor erases the topology signal
the policy is supposed to learn from.

The grade-strategy round flagged this as the single highest-risk OQ: if we
hit this in week 3, the schedule cannot absorb a redesign. We therefore
gate the decision on a **2-hour timeboxed spike on Day 2** before any
trainer code is written against an assumed buffer shape.

## Spike (Day 2, 2-hour timebox)

Notebook: `notebooks/00_buffer_spike.ipynb`.

Three options are explored, in order, until one passes the acceptance bar
or the timebox expires:

1. **Custom `features_extractor` + padded obs.** Subclass
   `BaseFeaturesExtractor`, accept `Dict({nodes, edges, mask})`, pool
   node embeddings inside the extractor. SB3 buffer stores padded
   tensors; topology lives in the extractor, not the space.
2. **Pad-to-V_max=512 + action mask.** Treat |V|<=512 as fixed,
   right-pad with zeros, expose an action mask so PPO never selects a
   pad slot. Closest to stock SB3; loses inductive GraphSAGE elegance.
3. **Drop to SB3 low-level building blocks.** Reuse `RolloutBuffer`
   primitives but write our own `collect_rollouts` that handles
   variable-|V| natively. Highest engineering cost, highest fidelity.

### Acceptance bar (per option)

- Boots a 100-step rollout on a toy graph corpus without shape errors.
- Round-trips obs through buffer -> extractor -> policy head -> loss.
- P95 step latency <= 50 ms on the spike laptop.

## Decision

Run the spike in the order above. **Pick the first option that clears the
acceptance bar inside the 2-hour window.** Do not iterate past the
timebox; partial wins do not count.

## Fallback (if all three options fail the spike)

Adopt **fixed-max-|V|=512 with padded MLP encoder + action masking**.
This is option (2) without the GraphSAGE message-passing layer: a flat
MLP over padded node features, masked at the action head.

Trade-offs of the fallback:

- ✅ Unblocks PPO training on stock SB3 immediately.
- ✅ Action masking still prevents illegal refactor targets.
- ❌ Loses inductive generalisation across graph sizes (the GraphSAGE
  selling point in §2.4).
- ❌ Caps the corpus to graphs with |V|<=512; larger files are dropped
  from training with a logged warning.
- ⚠️ Must be disclosed in the §2.4 essay and the results section as a
  scope reduction, not silently absorbed.

## Consequences

- **Schedule.** Day 2 is reserved for the spike. Trainer scaffolding
  (Phase 3) cannot start until ADR-008 is amended with the spike
  outcome.
- **Interface.** Whichever option wins, the GraphifyAdapter (ADR-002)
  exposes obs in a single canonical shape. The policy never sees raw
  PythonClaw output.
- **Reversibility.** All three options preserve the SB3 PPO trainer
  surface; only the `features_extractor` and `observation_space`
  change. We can swap between options post-spike without rewriting the
  training loop, at the cost of re-running the seed sweep.

## Amendment policy

This ADR is **proposed**, not accepted. After the spike, append an
"Amendment — Spike Outcome" section that records:

1. Which option was attempted, in which order.
2. The minute mark at which each option passed or was abandoned.
3. The chosen path forward, with a one-line justification.
4. Any deviation from the fallback (e.g. V_max raised/lowered).

Only after that amendment is committed does ADR-008 move to **Accepted**
and unblock Phase 3 trainer work.
