# ACTION_DESIGN.md — Action Space + Reward for PythonClaw Refactor RL

**Status:** draft, paired with STATE_DESIGN.md v1.
**Owner:** human (architect). AI implements against this contract only.
**Canonical authority:** ADR-007 (reward), ADR-002 (graph build),
CLAUDE.md §CANONICAL VALUES (single source of truth across docs).

## 1. Action space overview

Discrete, masked, parametric. At step `t` the env exposes a per-state
**legal action set** `L_t ⊆ A_max` and the policy emits logits over the
full `A_max` index space; illegal entries are zeroed by the mask before
softmax. There are **three structural categories** plus a NO-OP:

1. `split_module(node_id, partition_id)`
2. `merge_modules(node_a, node_b)`
3. `rewire(edge_id, new_target)`
4. `noop()`

`A_max` arithmetic (CLAUDE.md §CANONICAL "A_max arithmetic"):

- `V_max = 512` (ADR-008 fallback cap), `K = 8` split-points,
  `M = 16` merge-targets, `E_max = 4096` edges, `R = 8` rewire-targets.
- `A_split  = V_max · K  = 512 · 8  = 4096`
- `A_merge  = V_max · M  = 512 · 16 = 8192`
- `A_rewire = E_max · R  = 4096 · 8 = 32768`
- `A_noop   = 1`
- `A_max    = 4096 + 8192 + 32768 + 1 = **45057**`

Exact `A_max` is pinned in `config/config.yaml#action`; the policy head
sizes its final linear layer from that constant so checkpoints stay
compatible across runs with identical config.

## 2. Action semantics

### 2.1 `split_module(node_id, partition_id)`

- Preconditions: `kind[node_id] ∈ {class, module}` AND `LOC[node_id] ≥
  LOC_split_min` AND node is not in `frozen_set`.
- Effect: partition the symbols owned by `node_id` into two **disjoint**
  subsets `S_A, S_B` according to template `partition_id`. Two new nodes
  replace the original; inbound edges are rerouted to whichever subset
  contains the called symbol; outbound edges are duplicated only when
  both subsets call the target.
- Legal-mask rule: at most `K` legal entries per qualifying node;
  templates whose resulting `|S_A| / |S_B| < 0.1` are masked out
  to prevent degenerate splits.

### 2.2 `merge_modules(node_a, node_b)`

- Preconditions: `kind[node_a] == kind[node_b]`, both in the same layer
  (`layer_L*` triple matches), and `(a, b)` or `(b, a)` is a top-`M`
  neighbor edge in `A`.
- Effect: union the symbol sets; the merged node inherits the max of
  `cyclomatic`, sum of `LOC`, and a recomputed degree. Self-loops
  created by the merge are dropped.
- Legal-mask rule: masked when the merge would push composite complexity
  above `complexity_norm_cap`.

### 2.3 `rewire(edge_id, new_target)`

- Preconditions: edge exists in `edge_attrs[r]` for some relation `r ∈
  {call, import}` (inheritance edges are immutable in v0; data-flow is
  out of scope).
- Effect: redirect `(src, old_dst, r)` to `(src, new_target, r)`.
  `new_target` is drawn from the top-`R` candidates by feature cosine
  to `old_dst`, ensuring the rewire is semantically plausible.
- Legal-mask rule: masked when rewire would create a cycle in `import`
  (PythonClaw forbids circular imports as a hard constraint).

### 2.4 `noop()`

- Always legal. Used by the agent to defer commitment when entropy is
  high and to provide a fixed reference point for advantage estimates.

## 3. Action masking implementation

- Env produces `legal_mask: torch.BoolTensor` of shape `(A_max,)` once
  per step. Construction is `O(|V| + |E|)` and is memoized while the
  graph is unchanged within a step.
- Policy applies the masking rule **pre-softmax: illegal logits → −∞**
  (implemented as `logits = logits.masked_fill(~legal_mask, -1e9)` since
  finite −1e9 underflows to 0 after `softmax` in fp32 while staying
  numerically safe under fp16 mixed precision). This follows the
  invalid-action-masking formulation of **Huang & Ontañón (2022),
  "A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms"** — masking before the softmax (not after) preserves
  the unbiased policy-gradient estimator. Value head and entropy bonus
  see the same masked logits, so exploration credit is only granted
  over legal moves.
- During replay sampling, the stored `legal_mask` is replayed alongside
  the action; PPO importance ratios are clipped only on legal indices.

## 4. Reward (canonical — verbatim from ADR-007)

```
R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t
```

Defaults from `config/config.yaml#reward` (frozen per ADR-007):

| symbol     | value | role                                                        |
|------------|-------|-------------------------------------------------------------|
| α          | 1.0   | ΔModularity weight                                          |
| β          | 1.0   | ΔCohesion weight                                            |
| γ          | 0.5   | Coupling penalty weight                                     |
| P_skills_t | -5.0  | Lazy-load-break PENALTY (NEGATIVE — no positive bonus form) |

- `ΔModularity_t = Q_mod(G_t) − Q_mod(G_{t-1})` — graph-modularity delta
  (community-structure improvement, positive when modularity rises).
- `ΔCohesion_t = C(G_t) − C(G_{t-1})` — intra-module cohesion delta.
- `Coupling_Penalty_t` — cross-module edge weight (subtracted with γ).
- `P_skills_t` — NEGATIVE scalar applied when the lazy-load monitor
  (ADR-005) detects a break this step; 0 otherwise. No positive
  `skills_bonus` term exists.

Stale formulations explicitly forbidden (per ADR-007): `ΔReuse`,
`ΔQ_struct`, `ΔQ_runtime`, positive `skills_bonus`, `+P_skills`,
`λ · skills_bonus`. Any of these is a Phase-0 contract violation.

Weights `(α, β, γ, P_skills)` are MUST-ablated per ADR-007's 3×3×3×2
grid (54 cells × ≥5 seeds in fine pass).

**Reward shaping discipline (Ng, Harada & Russell, 1999 —
"Policy Invariance Under Reward Transformations: Theory and Application
to Reward Shaping").** Every shaping term in R_t is a **potential-based
difference** (ΔModularity, ΔCohesion are state-potential deltas;
Coupling_Penalty is a state-function subtracted with γ; P_skills fires
only on a transition event). This satisfies the Ng et al. invariance
theorem: the optimal policy of the shaped MDP equals the optimal policy
of the un-shaped MDP, so the four-term composite reward does not bias
the policy away from genuine modularity gain. Any future reward term
MUST be expressible as a potential-difference F(s,s') = γΦ(s') − Φ(s)
or it is rejected at ADR review.

## 5. Terminal + truncation

- Episode terminates on success criterion (config-pinned) or at
  `t = T_horizon = 256` (truncation). Truncated episodes still
  bootstrap the value target; success episodes do not.
- Early-terminate fires if `noop` is sampled `N_noop_max = 16`
  consecutive times — prevents stalling for entropy credit.

## 6. Test hooks

- Unit tests assert every action category round-trips through the env
  (`apply → undo` restores byte-identical `(A, X, edge_attrs)`).
- `tests/architecture/test_reward_formula.py` AST-parses `src/env/reward.py`
  and asserts the four canonical terms (ΔModularity, ΔCohesion,
  Coupling_Penalty, P_skills) with correct signs (γ subtracted; P_skills
  added as a negative term).
- Property test fuzzes `legal_mask` against 1000 random graphs and
  asserts no masked action ever mutates state when forced through.
