# STATE_DESIGN.md — Observation Space for PythonClaw Refactor RL

**Status:** draft, locked behind ADR-002 (GraphifyAdapter).
**Owner:** human (architect). AI implements against this contract only.

## 1. State tuple

State `s_t = (A_t, X_t, edge_attrs_t)` at episode step `t`. All three are
derived from the same code graph snapshot produced by `GraphifyAdapter`.
Together they fully describe the agent's observation; nothing else from
the environment leaks into the policy network.

- `A_t`: weighted adjacency, `scipy.sparse.csr_matrix`, shape `(|V|, |V|)`.
  `A_t[i, j] = Σ_r w_r · 𝟙[(i, j) ∈ E_r]` where `r` ranges over relation
  types (`calls`, `imports`, `inherits`, `co_change`, `data_flow`). Weights
  `w_r` come from `config/state.yaml` and are frozen per experiment seed.
- `X_t`: dense node-feature matrix, `torch.FloatTensor`, shape `(|V|, 16)`.
  Column order is the contract in §3 and must not be reordered without an
  ADR bump.
- `edge_attrs_t`: `dict[str, scipy.sparse.csr_matrix]` keyed by relation
  type, plus a `counts` sub-dict `{r: nnz_r}` for fast reward shaping.

## 2. Adjacency `A` construction

1. Build a per-relation sparse matrix `A_r` from the typed edge list emitted
   by `GraphifyAdapter` (one `(src, dst, weight)` triple per edge).
2. Sum across relations: `A = Σ_r w_r · A_r`. Self-loops are dropped before
   summation; symmetry is **not** enforced (call graphs are directional).
3. Store in CSR for `O(nnz)` row slicing during GNN message passing.
4. The raw per-relation matrices stay in `edge_attrs` so the policy head
   can attend to relation type without re-deriving it from `A`.

## 3. Node features `X` — 16 columns

| idx | name              | dtype  | source                            | normalization        |
|-----|-------------------|--------|-----------------------------------|----------------------|
| 0   | LOC               | float  | tokenizer line count              | log1p, /max_LOC      |
| 1   | cyclomatic        | float  | `radon` cyclomatic complexity     | /20 clipped to [0,1] |
| 2   | degree_in         | float  | `A.sum(axis=0)[node]`             | /max_deg             |
| 3   | degree_out        | float  | `A.sum(axis=1)[node]`             | /max_deg             |
| 4   | betweenness_cached| float  | networkx, computed once/seed      | already in [0,1]     |
| 5   | layer_L1          | float  | one-hot: domain layer             | {0,1}                |
| 6   | layer_L2          | float  | one-hot: service layer            | {0,1}                |
| 7   | layer_L3          | float  | one-hot: infra/IO layer           | {0,1}                |
| 8   | lazy_load_flag    | float  | static analyzer (deferred import) | {0,1}                |
| 9   | kind_class        | float  | AST node kind one-hot             | {0,1}                |
| 10  | kind_module       | float  | AST node kind one-hot             | {0,1}                |
| 11  | kind_function     | float  | AST node kind one-hot             | {0,1}                |
| 12  | in_skills         | float  | `P_skills` membership lookup      | {0,1}                |
| 13  | complexity_norm   | float  | (cyclomatic · LOC) composite      | /max_composite       |
| 14  | age_episodes      | float  | episodes since node was last edited | /T_horizon         |
| 15  | reserved          | float  | always 0.0 (future-proofing)      | —                    |

Notes:
- One-hot triples (`layer_L*`, `kind_*`) are mutually exclusive; unknown
  layer/kind sets all three to 0, which is a legal sentinel.
- Betweenness is cached per `(repo_sha, seed)` to keep the per-step cost
  bounded; recompute happens only at episode reset, not per action.

## 4. Variable-|V| handling

- `V_max = 512`. Episodes that exceed this are rejected by the env before
  reset (logged as `oversized_repo`); no truncation policy ships in v0.
- For `|V| < V_max`:
  - Pad `X` with zero rows up to `(V_max, 16)`.
  - Pad `A` by allocating a `(V_max, V_max)` CSR and assigning the
    `|V| × |V|` block to the top-left; remaining rows/cols stay empty.
  - Emit a boolean `mask: torch.BoolTensor` of shape `(V_max,)` with
    `mask[i] = True` iff `i < |V|`. The policy head multiplies logits
    by `mask` before softmax so padded nodes never receive probability.
- The mask is part of the observation dict but **not** part of the
  `(A, X, edge_attrs)` tuple — it is environment-side metadata that the
  agent consumes alongside the state.

## 5. Serialization for replay buffer

- Stored record per step: `{A_coo, X_fp16, edge_counts, mask, node_ids}`.
  - `A_coo`: COO tuple `(row, col, data)` in `int32 / int32 / float16`.
  - `X_fp16`: `X` cast to fp16; safe because all features sit in [0, 1]
    after normalization and gradients live on the GPU-side fp32 copy.
  - `edge_counts`: `dict[str, int]` (the heavy per-relation matrices are
    reconstructed on demand from `A` + relation masks at sample time).
  - `mask`: packed bits, `V_max / 8` bytes.
  - `node_ids`: `int32` array mapping local index → stable graph id for
    cross-episode joins.
- Compression: `lz4` frame around the concatenated bytes; benched at
  ~3.1× ratio on the bootstrap corpus, well inside the buffer budget.
- Schema is versioned (`state_schema_version=1`); buffer loader refuses
  to mix versions to avoid silent drift after column edits.

## 6. Determinism + reproducibility

- All sparse builders take `(repo_sha, seed)` and are deterministic.
- Feature normalization constants (`max_LOC`, `max_deg`, `max_composite`)
  are computed once on the training corpus and pinned in
  `config/state.yaml`; eval runs reuse the same constants so train/eval
  feature distributions stay aligned.
- Unit tests assert `state(repo_sha, seed) == state(repo_sha, seed)`
  byte-for-byte across two cold processes — this is the regression
  fence against accidental nondeterminism creeping in from networkx.
