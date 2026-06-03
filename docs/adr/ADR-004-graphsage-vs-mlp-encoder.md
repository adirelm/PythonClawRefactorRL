# ADR-004: GraphSAGE vs MLP encoder for variable-|V| code-graph state

- **ID**: ADR-004
- **Status**: Accepted
- **Date**: 2026-06-04
- **Related**: ADR-002 (GraphifyAdapter)

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
| Variable \|V\| | Native | Pad + mask to max \|V\| |
| Topology preserved | Yes | No (flattened) |
| Parameter count | Independent of max \|V\| | Grows with max \|V\|^2 |
| Inductive on unseen nodes | Yes (Hamilton et al. 2017) | No |
| Per-step forward cost | Higher than same-width MLP | Lower per FLOP |
| Extra runtime dependency | Yes (`torch-geometric`) | No |
| Sample efficiency on graph tasks | Better | Worse |

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
- **Graph Transformer / GATv2** — deferred on cost grounds; revisit
  if the 2-hop receptive field proves insufficient on larger files.
- **Hand-crafted graph features into an MLP** — rejected; reintroduces
  manual feature engineering the brief asks us to avoid.

## References

- Hamilton, W. L., Ying, R., and Leskovec, J. (2017). *Inductive
  Representation Learning on Large Graphs.* NeurIPS 2017.
- Kipf, T. N. and Welling, M. (2017). *Semi-Supervised Classification
  with Graph Convolutional Networks.* ICLR 2017.
- PyTorch Geometric documentation — `torch_geometric.nn.SAGEConv`.
- ADR-002: GraphifyAdapter (upstream graph contract).
