# ADR-002: GRAPHIFY — Local NetworkX Implementation Behind an Adapter

- **Status:** Accepted
- **Date:** 2026-06-04
- **Deciders:** Architect (human, §1.4 contract)
- **Implementer:** AI agent against this ADR
- **Supersedes:** —
- **Contract Authority:** **this ADR is the single source of truth for the
  `GraphifyAdapter` Protocol, node/edge attribute schema, and `EdgeWeigher`
  Protocol.** PRD-GRAPHIFY, STATE_DESIGN, ACTION_DESIGN, and all
  `src/graphify/` code reference this ADR as truth. Any drift in method
  name, signature, or attribute set must land here first.
- **Related:** OQ-2 (GRAPHIFY binary availability), ADR-003
  (tiktoken cost-metric sidecar — `token_count` lives there, NOT on graph
  nodes), ADR-004 (GraphSAGE consumer of this graph), ADR-005
  (`lazy_load_flag` node attribute semantics)

> **Decoupling note.** ADR-001 covers the *PythonClaw* skills shim
> (`src/pythonclaw/`) — a separate concern from the *graph adapter*. The
> earlier framing that lumped both under "adapter shims" was a category
> error: ADR-001 is about Skills L1/L2/L3 wrappers, ADR-002 is about
> turning Python source into a weighted DiGraph. They share a pattern
> name only; they do not share a contract, a layer, or a test surface.

## Context

The ex04 brief §1.2 specifies that the refactor environment consumes a
**weighted** directed graph `G = (V, E)` where `V` is the set of code
symbols (functions, classes, modules) and `E` encodes call / import /
inheritance edges. The brief names GRAPHIFY as the reference extractor
but does not guarantee a published binary or an installable package —
this is logged as **OQ-2** in
`instructions/assignment-4/open_questions.md`.

Two failure modes follow from doing nothing:

1. **Binary path is gated.** If GRAPHIFY ships only as a private artifact,
   we cannot run reproducibly inside CI or on a grader's machine.
2. **Binary path produces no artifacts.** Even if it runs, the grader sees
   a black box: no math, no test surface, no code to review. The A2 and
   A3 rubrics rewarded *"we built it ourselves, here is the math"*; a
   shelled-out binary leaves nothing to grade.

## Decision

Define `GraphifyAdapter` as a `typing.Protocol`, then ship one concrete
implementation now and leave room for a second later.

```
src/graphify/
    __init__.py            # re-exports GraphifyAdapter, EdgeWeigher, get_default()
    adapter.py             # Protocol definitions (≤80 LOC)
    weigher.py             # EdgeWeigher Protocol + default impls (≤120 LOC)
    local_impl.py          # NetworkX implementation (≤150 LOC)
    # local_binary.py      # FUTURE — only if GRAPHIFY binary lands
```

### FR-1 — `GraphifyAdapter` Protocol (canonical signature)

This is the **single canonical signature**. Any other name (`.extract()`,
`.parse()`) anywhere in the codebase, docs, or tests is stale and must
be rewritten to match this block:

```python
from pathlib import Path
from typing import Protocol
import networkx as nx

class GraphifyAdapter(Protocol):
    def build(self, src_root: Path, *, seed: int = 0) -> nx.DiGraph: ...
    def load(self, pickle_path: Path) -> nx.DiGraph: ...
```

- `build()` walks `src_root` with the `ast` module and returns a fresh
  weighted `nx.DiGraph`. `seed` is forwarded to any non-deterministic
  weighting step (e.g. learned EdgeWeigher) so runs are reproducible.
- `load()` deserializes a previously pickled graph (used by the eval
  harness and by checkpoint replay).

### FR-2 — Node attribute schema (canonical)

Every node in the returned `nx.DiGraph` carries **exactly** this
attribute set:

| Attribute        | Type     | Domain / Notes |
|------------------|----------|----------------|
| `kind`           | `str`    | one of `{"function", "class", "module"}` |
| `LOC`            | `int`    | lines of code in the symbol body (≥1) |
| `cyclomatic`     | `int`    | McCabe cyclomatic complexity (≥1) |
| `layer`          | `str`    | architectural layer tag (e.g. `"domain"`, `"infra"`, `"ui"`, `"unknown"`) |
| `lazy_load_flag` | `bool`   | True iff symbol participates in a lazy-load contract (ADR-005 semantics) |

No other attributes belong on graph nodes. In particular, **`token_count`
does NOT live on the node** — it is a cost metric, governed by ADR-003,
and is emitted to a separate sidecar (`results/cost/tokens.parquet`).

### FR-3 — Edge attribute schema (canonical)

Every edge in the returned `nx.DiGraph` carries **exactly** this
attribute set:

| Attribute   | Type    | Domain / Notes |
|-------------|---------|----------------|
| `rel_type`  | `str`   | one of `{"call", "import", "inheritance"}` |
| `weight`    | `float` | strictly positive; produced by FR-4 `EdgeWeigher` |

`rel_type ∈ {"call", "import", "inheritance"}` is closed — any new
relation type requires an ADR amendment.

### FR-4 — `EdgeWeigher` Protocol (brief §1.2 "weighted graph" mandate)

The brief explicitly requires a **weighted** graph. Weighting strategy
is decoupled from extraction so we can A/B different weight schemes
without rebuilding the AST walker:

```python
class EdgeWeigher(Protocol):
    def weight(
        self,
        src: str,
        dst: str,
        rel_type: str,
        ctx: "WeighingContext",
    ) -> float: ...
```

Default ships in `src/graphify/weigher.py`:

- `UniformEdgeWeigher` → always returns `1.0` (baseline / ablation).
- `FrequencyEdgeWeigher` → for `rel_type == "call"`, returns the number
  of call-sites; for `import` / `inheritance`, returns `1.0`.

`LocalImpl.build()` accepts an optional `weigher: EdgeWeigher`
constructor argument and defaults to `FrequencyEdgeWeigher`. The chosen
weigher is recorded in graph-level metadata (`G.graph["weigher"] =
weigher.__class__.__name__`) so eval logs can reproduce it.

### FR-5 — `token_count` is NOT a graph node attribute

`token_count` (tiktoken `cl100k_base`) is a **cost metric**, not a
structural graph property. ADR-003 governs the cost-metric sidecar.
`local_impl.py` MAY compute token counts as a by-product of its AST
walk, but MUST emit them to `results/cost/tokens.parquet` keyed by
fully-qualified symbol name — not to `G.nodes[n]["token_count"]`. Tests
in `tests/architecture/test_graph_node_schema.py` assert the node
attribute set is exactly FR-2 (no extras).

### Implementation sketch

`LocalImpl` walks `src_root` with the `ast` module, builds nodes for
every top-level symbol, and emits edges for: (a) `Call` nodes inside a
function body (`rel_type="call"`), (b) `Import` / `ImportFrom` targets
(`rel_type="import"`), (c) class bases in `ClassDef`
(`rel_type="inheritance"`). Edge `weight` is delegated to the injected
`EdgeWeigher`. Node attributes follow FR-2 exactly.

A future `LocalBinaryImpl` would wrap the GRAPHIFY CLI behind the same
`build()` / `load()` Protocol; **no agent code, no reward-shaping code,
and no test fixture imports the concrete class** — they all depend on
the `GraphifyAdapter` Protocol. The 24h swap window from the project
swap-policy applies.

## Alternatives Considered

| # | Alternative | Why rejected |
|---|---|---|
| A1 | Shell out to GRAPHIFY binary as the only path | OQ-2 unresolved; zero artifacts for grader; not reproducible in CI |
| A2 | Use `pyan3` / `code2flow` directly, no adapter | Locks us to one tool's quirks; can't swap; obscures the §1.2 math |
| A3 | Hand-rolled regex / import scan, no `ast` | Loses scope resolution; misses dynamic calls; fails on decorators |
| A4 | **Adapter + local NetworkX `ast` impl now, binary impl later** | ✅ Chosen — testable, swappable, matches §1.2 math, scores on rubric |
| A5 | Adapter + multiple impls shipped simultaneously | Premature; doubles test surface before we know if the binary exists |
| A6 | Bake weighting into `build()` with no `EdgeWeigher` seam | Blocks weight-scheme ablation; brief §1.2 calls out "weighted" but is silent on scheme |
| A7 | Put `token_count` on graph nodes | Conflates structural graph with cost-metric pipeline; breaks ADR-003 separation; pollutes FR-2 schema |

## Consequences

**Positive.**
- The `build()` / `load()` Protocol is the only seam the RL agent sees;
  swapping implementations is a one-line change in `get_default()`.
- The NetworkX impl is fully covered by unit tests on tiny fixture
  repos — the 85% coverage gate from CLAUDE.md is reachable.
- §2.4 essay can cite the local impl directly ("here is `local_impl.py`
  line 42, here is the AST visitor for `Call` edges") instead of waving
  at a binary.
- Betweenness-centrality experiments (exactly 2 calls per seed — start
  and end — over ≥5 seeds, mean ± std + 95% CI per ADR-006) run on a
  graph we control end-to-end.
- `EdgeWeigher` Protocol lets us ship a uniform-weight ablation and a
  frequency-weight default without re-rolling the AST walker.

**Negative.**
- Local impl will miss some dynamic dispatch (e.g. `getattr`-based
  calls). Documented as a known limitation in `local_impl.py` docstring
  and §2.4 essay's "threats to validity" section.
- If GRAPHIFY publishes later with better recall, we must run a
  comparison study to justify keeping the local impl.

**Neutral.**
- ADR-001 (PythonClaw Skills shim) and ADR-002 (Graphify adapter) share
  a Protocol-based pattern but solve different problems. Treat them as
  independent contracts.

## V3 Traceability

| Source | Anchor | This ADR addresses |
|---|---|---|
| ex04 brief §1.2 | "weighted G = (V, E) where V = symbols, E = relations" | FR-1 `build()` returns `nx.DiGraph`; FR-4 `EdgeWeigher` produces `weight` |
| ex04 brief §1.4 | "AI is implementer, human is architect" | ADR is human-authored; agent codes against it |
| OQ-2 | "GRAPHIFY binary availability uncertain" | Alternatives A1, A4; future `local_binary.py` |
| Grade-strategy round | "we built it ourselves, here is the math" | Decision §, A1 rejection, §2.4 essay hook |
| Lecture 7 (PPO + GAE) | Agent acts on state derived from graph | Protocol returns `nx.DiGraph`, agent code unchanged across swaps |
| CLAUDE.md §1 | ≤150 LOC per .py | `local_impl.py` size budget noted in Decision § |
| CLAUDE.md §2 | TDD + 85% coverage | Adapter testable via fixture repos |
| ADR-003 | tiktoken `cl100k_base` cost-metric sidecar | FR-5: `token_count` NOT on graph nodes |
| ADR-005 | `lazy_load_flag` semantics | FR-2 node attribute |
| ADR-006 | Betweenness called exactly twice per seed | Graph stability under repeated metric calls |

## Open follow-ups

- **OQ-2 monitoring** — re-check GRAPHIFY availability at Phase 3 gate;
  if published, evaluate `local_binary.py` under the 24h swap window.
- **Comparison harness** — if both impls ever coexist, a parity harness
  must verify they emit graphs satisfying FR-1 through FR-5 identically
  on the fixture repos.
- **Weight-scheme ablation** — log results of `UniformEdgeWeigher` vs
  `FrequencyEdgeWeigher` in §2.4 essay once training converges.
