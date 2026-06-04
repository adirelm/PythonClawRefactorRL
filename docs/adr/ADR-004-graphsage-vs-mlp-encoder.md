# ADR-004: GraphSAGE vs MLP encoder for variable-|V| code-graph state

- **ID**: ADR-004
- **Status**: Accepted (primary path)
- **Date**: 2026-06-04
- **Related**: ADR-002 (GraphifyAdapter), ADR-008 (SB3 variable-|V| buffer spike), STATE_DESIGN §4

## Context

The refactoring RL agent observes a Python code graph produced by the
GraphifyAdapter (ADR-002). Across episodes the node count |V| varies:

- different source files have different AST/CFG sizes;
- refactoring actions (extract-method, inline-variable, rename) add
  or remove nodes mid-episode;
- the curriculum mixes small toy modules with larger real files.

A standard MLP head expects a fixed-width input. Flattening the
adjacency matrix and padding to a global max |V| has three problems
we do not want to absorb:

1. Padding inflates parameter count quadratically in max |V|, most
   of which is masked-out noise.
2. Flattening discards topology — the network has to relearn that
   adjacency entries are not independent features.
3. Masking logic has to be threaded through actor, critic, and
   rollout buffer; any bug there silently corrupts gradients.

We need an encoder that ingests a graph of arbitrary size and emits
a fixed-width state embedding for the actor-critic heads.

## Decision

Adopt a **2-layer GraphSAGE encoder** implemented via
**PyTorch Geometric** (`torch-geometric` / PyG) as the state encoder
sitting between GraphifyAdapter output and the actor-critic heads.

- Layer count: 2 (2-hop receptive field).
- Aggregator: mean (PyG default for `SAGEConv`).
- Readout: mean-pool over node embeddings for the graph embedding,
  plus per-node embeddings for action heads that score node-local
  refactors.

A 2-hop receptive field matches the local-refactor scope: most
refactoring actions affect a node and its immediate AST/CFG
neighbors. Deeper stacks risk over-smoothing without buying useful
context.

## Tradeoffs

| Aspect | GraphSAGE (2-layer, PyG) | MLP on padded adjacency |
|---|---|---|
| Variable \|V\| | Native (PyG `DataLoader` batches per-graph) | Pad + mask to max \|V\| |
| Topology preserved | Yes | No (flattened) |
| Parameter count | Independent of max \|V\| (see V_max note) | Grows with max \|V\|^2 |
| Inductive on unseen nodes | Yes (Hamilton et al. 2017) | No |
| Per-step forward cost | Higher than same-width MLP | Lower per FLOP |
| Extra runtime dependency | Yes (`torch-geometric`) | No |
| Sample efficiency on graph tasks | Better | Worse |

Note on alternatives: **GAT/GATv2** (attention aggregator) and **GIN**
(injective sum aggregator, Xu et al. 2019) were also considered as
size-independent encoders. Both are viable swaps for `SAGEConv` if a
later ablation shows mean-aggregation is the bottleneck; GraphSAGE is
chosen as the primary because of its explicit inductive sampling story
that matches our cross-file generalization requirement.

## V_max reconciliation (size-independence vs ADR-008 fallback)

The "parameter count independent of max |V|" claim above is a property
of the encoder itself: a 2-layer SAGEConv has weights of shape
`(in_dim, hidden) + (hidden, out)`, with no dependence on |V|. This
holds regardless of how observations reach the encoder.

What *can* depend on a `V_max` cap is the **rollout buffer layout**,
not the encoder. Per CANONICAL V_max policy and ADR-008:

- **Primary path (this ADR).** GraphSAGE accepts variable-|V| natively
  via the PyG `DataLoader` / `Batch.from_data_list` path. If the
  ADR-008 spike confirms SB3 `RolloutBuffer` can carry our `Dict` obs
  (outcome A: custom `features_extractor` + native variable-|V|), then
  `V_max` is **unused at runtime** — observations stream into the
  encoder at their natural size and `max_nodes_v = 512` is only a
  defensive ceiling for cost accounting.

- **Fallback path (ADR-008 outcomes B/C).** If `RolloutBuffer` requires
  fixed-shape tensors, observations are padded to `V_max = 512` plus a
  boolean node-mask. GraphSAGE still respects the original |V|:
  padded rows enter as zero-feature nodes with zero-weight edges in
  message-passing, the mask zeros them out before mean-pool readout,
  and per-node action logits over padded slots are masked to `-inf`
  before the softmax. The encoder's parameter count remains
  independent of `V_max`; only the per-step FLOP cost and the buffer
  tensor footprint scale with `V_max`.

STATE_DESIGN §0 and §4 mark `V_max` as **conditional** on this spike
outcome. ADR-008 is the gating decision; this ADR is the encoder
contract that holds under either outcome.

## Justification

GraphSAGE is designed for inductive node embedding on graphs of
varying size and unseen nodes, via sampled neighborhood aggregation
(Hamilton, Ying, Leskovec, NeurIPS 2017). That property is exactly
the variable-|V| requirement above. The same paper argues that
learned aggregators on local neighborhoods generalize better than
transductive whole-graph approaches when the graph distribution
shifts at test time — our setting, since the agent sees new code
files during evaluation.

We accept the extra `torch-geometric` dependency and the higher
per-step forward cost because the alternative (pad-and-mask MLP) is
strictly worse on the structural axes that matter: topology
preservation, parameter efficiency, and inductive generalization.

## Consequences

Positive:

- Handles variable |V| without bespoke pad/mask plumbing.
- Actor head scores node-local refactors directly from per-node
  embeddings.
- Future ablations against GCN or GAT swap only `SAGEConv`.

Negative:

- New runtime dependency: `torch-geometric`, plus its CPU/CUDA wheel
  matrix. Install caveats documented in `README.md` and pinned in
  `pyproject.toml`.
- Encoder forward becomes the dominant per-step cost; the
  training-time SLO must budget for it.
- Adds a graph-batching step (`Batch.from_data_list`) on the hot
  path that the rollout buffer must be aware of.

## Alternatives Considered

- **MLP on flattened, padded adjacency** — rejected (see Tradeoffs).
- **GCN (Kipf and Welling, 2017)** — viable but originally
  transductive; less natural for unseen graphs across episodes.
- **GAT / GATv2 (Veličković et al. 2018; Brody et al. 2022)** — viable
  size-independent alternative; attention aggregator may help when
  node-importance varies sharply across AST/CFG neighbors. Deferred on
  cost grounds and revisited as a `SAGEConv` swap if the 2-hop
  receptive field proves insufficient on larger files.
- **GIN (Xu et al. 2019)** — viable size-independent alternative;
  injective sum aggregator has stronger theoretical expressive power
  for graph isomorphism, but is empirically more sensitive to depth
  and batch-norm tuning than mean-aggregation SAGE on small graphs.
  Held as a fallback ablation, not the primary.
- **Hand-crafted graph features into an MLP** — rejected; reintroduces
  manual feature engineering the brief asks us to avoid.

## References

- Hamilton, W. L., Ying, R., and Leskovec, J. (2017). *Inductive
  Representation Learning on Large Graphs.* NeurIPS 2017.
  arXiv:1706.02216.
- Kipf, T. N. and Welling, M. (2017). *Semi-Supervised Classification
  with Graph Convolutional Networks.* ICLR 2017.
- Veličković, P. et al. (2018). *Graph Attention Networks.* ICLR 2018.
- Brody, S., Alon, U., and Yahav, E. (2022). *How Attentive are Graph
  Attention Networks?* ICLR 2022 (GATv2).
- Xu, K. et al. (2019). *How Powerful are Graph Neural Networks?*
  ICLR 2019 (GIN).
- PyTorch Geometric documentation — `torch_geometric.nn.SAGEConv`,
  `torch_geometric.loader.DataLoader`, `Batch.from_data_list`.
- ADR-002: GraphifyAdapter (upstream graph contract).
- ADR-008: SB3 variable-|V| buffer spike (gates V_max usage).
- STATE_DESIGN §4: observation-space contract and V_max policy.
