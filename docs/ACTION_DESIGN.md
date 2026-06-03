# ACTION_DESIGN.md — Action Space + Reward for PythonClaw Refactor RL

**Status:** draft, paired with STATE_DESIGN.md v1.
**Owner:** human (architect). AI implements against this contract only.

## 1. Action space overview

Discrete, masked, parametric. At step `t` the env exposes a per-state
**legal action set** `L_t ⊆ A_max` and the policy emits logits over the
full `A_max` index space; illegal entries are zeroed by the mask before
softmax. There are **three structural categories** plus a NO-OP:

1. `split_module(node_id, partition_id)`
2. `merge_modules(node_a, node_b)`
3. `rewire(edge_id, new_target)`
4. `noop()`

Per-state action count varies, hence "parametric": the policy network
always sees `A_max` logits but only `|L_t|` are valid. `A_max` is
budgeted at construction time from `V_max` and edge limits:

- `A_split = V_max · K_partitions` where `K_partitions = 8`
  (pre-baked partition templates: by-class, by-call-cluster,
  by-data-flow, etc.).
- `A_merge = V_max · M_candidates` where `M_candidates = 16` (top-M
  neighbors by adjacency weight, computed per node from `A`).
- `A_rewire = E_max · R_targets` where `E_max = 4096`, `R_targets = 8`
  (top-R alternative targets by feature cosine).
- `A_noop = 1`.
- `A_max = A_split + A_merge + A_rewire + A_noop = 4096·8 + 512·16 + 512·8 + 1 = 32 769 + …`.

Exact `A_max` is pinned in `config/action.yaml`; the policy head sizes
its final linear layer from that constant so checkpoints stay
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
- Legal-mask rule: at most `K_partitions` legal entries per qualifying
  node; templates whose resulting `|S_A| / |S_B| < 0.1` are masked out
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
  {calls, imports}` (data-flow / inheritance edges are immutable in v0).
- Effect: redirect `(src, old_dst, r)` to `(src, new_target, r)`.
  `new_target` is drawn from the top-`R` candidates by feature cosine
  to `old_dst`, ensuring the rewire is semantically plausible.
- Legal-mask rule: masked when rewire would create a cycle in `imports`
  (PythonClaw forbids circular imports as a hard constraint).

### 2.4 `noop()`

- Always legal. Used by the agent to defer commitment when entropy is
  high and to provide a fixed reference point for advantage estimates.

## 3. Action masking implementation

- Env produces `legal_mask: torch.BoolTensor` of shape `(A_max,)` once
  per step. Construction is `O(|V| + |E|)` and is memoized while the
  graph is unchanged within a step.
- Policy applies `logits = logits.masked_fill(~legal_mask, -1e9)` before
  `softmax`, then samples. Value head sees the same masked logits when
  computing entropy bonuses, so exploration credit is only granted over
  legal moves.
- During replay sampling, the stored `legal_mask` is replayed alongside
  the action; PPO importance ratios are clipped only on legal indices.

## 4. Reward sketch

Per-step reward decomposes into three weighted terms plus a skills
shaping bonus:

```
r_t = α · ΔQ_struct + β · ΔQ_runtime − γ · cost_t + λ · skills_bonus_t
```

- `ΔQ_struct`: drop in a static-quality scalar (weighted sum of cyclomatic,
  fan-out, lazy-load violations, betweenness concentration). Computed as
  `Q_struct(s_{t-1}) − Q_struct(s_t)` so improvements are positive.
- `ΔQ_runtime`: drop in P95 token count from the lazy-load-broken pytest
  walk over `sys.modules`. Captured only every `K_runtime` steps to keep
  the env step cost bounded; intermediate steps use the cached value.
- `cost_t`: edit cost — a fixed per-action penalty plus a churn term
  proportional to the number of symbols touched. NO-OP cost is zero by
  construction so the agent can stall without bleeding reward.
- `skills_bonus_t`: `+ε · 𝟙[touched_node ∈ P_skills]` — a small positive
  shaping signal when the agent acts on nodes flagged in the curriculum
  skill set. ε is held small enough that ablation (λ → 0) recovers a
  policy within the rolling-100 convergence band (see `config/reward.yaml`).

Weights `(α, β, γ, λ)` are MUST-ablated per the locked decision:
≥ 5 seeds per cell, full matrix in `results/ablation/`. Defaults
`α = 1.0, β = 0.5, γ = 0.1, λ = 0.05` are starting points only; final
values land after Phase 4 sweeps.

## 5. Terminal + truncation

- Episode terminates when `Q_struct(s_t) ≤ Q_struct_target` (success) or
  when `t = T_horizon = 256` (truncation). Truncated episodes still
  bootstrap the value target; success episodes do not.
- An additional early-terminate fires if `noop` is sampled `N_noop_max =
  16` consecutive times — this prevents the policy from collecting the
  entropy bonus by stalling forever.

## 6. Test hooks

- Unit tests assert that every action category round-trips through the
  env (`apply → undo` restores byte-identical `(A, X, edge_attrs)`).
- A property test fuzzes `legal_mask` against 1 000 random graphs and
  asserts no masked action ever mutates state when forced through.
- Reward tests pin the sign of each component on hand-crafted graphs so
  α/β/γ ablation can never silently flip the reward direction.
