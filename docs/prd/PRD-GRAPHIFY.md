# PRD-GRAPHIFY — Static-Analysis-to-Graph Engine (Local Re-Implementation)

**Status:** Draft v0.1 · **Owner:** A4 Working Group · **Scope:** ADR-002 boundary
**Parent:** `docs/PRD.md` §3.1 F2 (GRAPHIFY adapter, brief §1.2)

## 1. Purpose

The course brief §1.2 specifies that the refactor target must be modelled as
a **directed weighted graph G = (V, E)** extracted from source code, where
nodes carry structural metadata and edges carry dependency-relation types.
GRAPHIFY is the component that produces that graph. GRAPHIFY is not
publicly distributed; we re-implement the brief §1.2 G=(V,E) extraction
locally to (a) own the artifact for grading, (b) make the adapter
swap-ready when/if the binary becomes available. The local re-impl sits
behind `GraphifyAdapter` (ADR-002) so an upstream binary can swap in
within a 24h window without code change in the RL trainer or reward
shaper. See `instructions/assignment-4/open_questions.md` OQ-2 for the
swap-readiness discussion.

## 2. Inputs

- Python source tree rooted at `src/pythonclaw_shim/skills/` (any depth).
- Optional include/exclude globs (default: include `*.py`, exclude `tests/`,
  `__pycache__/`, anything matching `*.pyi`).
- Seed (int) — only used for tie-breaking in deterministic node ordering.

## 3. Outputs

A `networkx.DiGraph` with:

**Node attributes** (one node per top-level function, class, and module):
- `kind` ∈ {`module`, `class`, `function`, `method`}
- `LOC` — physical lines of code (excludes blank + comment-only lines)
- `cyclomatic` — McCabe complexity via `radon.complexity.cc_visit`
- `layer` ∈ {`io`, `domain`, `infra`, `unknown`} — inferred from import
  patterns and module path heuristic
- `lazy_load_flag` ∈ {True, False} — True if imported only inside a function
  body or guarded by `if TYPE_CHECKING` / `try/except ImportError`

**Edge attributes:**
- `rel_type` ∈ {`call`, `import`, `inheritance`}
- `weight` ∈ ℝ⁺ — count of occurrences (calls: callsite count; imports: 1;
  inheritance: 1) scaled by §6 weighting strategy

The graph is serialised to `results/graphify/<run_id>/graph.gpickle` and a
side-car `graph.json` for human inspection.

## 4. Functional Requirements

- **FR-1 AST walk.** Use stdlib `ast` only (no Python execution of target
  code). One pass per file; visitor pattern subclass per node kind.
- **FR-2 Symbol resolution.** Build a flat `qualname → node_id` table
  during pass 1; resolve call/inheritance edges in pass 2 so forward
  references work.
- **FR-3 Dependency extraction.** Detect: direct calls (`ast.Call` with
  `Name`/`Attribute` func), method calls on `self`, class bases
  (`ClassDef.bases`), `import` / `from ... import` statements.
- **FR-4 Weighting strategy.** Default = raw count. Pluggable
  `EdgeWeigher` Protocol so the RL reward shaper (R-1.3) can pass a
  centrality-aware weigher without GRAPHIFY needing to know about RL.
- **FR-5 Determinism.** Same input tree + same seed ⇒ byte-identical
  `graph.gpickle` (verified by SHA-256 in `tests/test_determinism.py`).
- **FR-6 Performance budget.** ≤ 2s wall-clock on the 50-file shim
  corpus; ≤ 200ms on the 5-file smoke fixture.

## 5. Edge Cases (Explicit)

- **Dynamic imports** (`importlib.import_module(name)` where `name` is not
  a string literal): emit a `call` edge to `importlib.import_module` and
  log a `DynamicImportWarning`; do **not** fabricate a phantom node.
- **Conditional imports** (inside `if`, `try/except`, function body):
  flag node with `lazy_load_flag=True`; still emit the `import` edge.
- **Decorators**: treat `@foo` as a `call` edge from the decorated
  function/class node to `foo`. Stacked decorators ⇒ multiple edges.
- **`TYPE_CHECKING` guards** (PEP 484): emit the import edge but mark
  `lazy_load_flag=True` and `weight *= 0.1` (configurable).
- **Re-exports** (`from x import y as y` in `__init__.py`): collapse into
  a single edge from the package module to `x.y`; do not double-count.
- **Star imports** (`from x import *`): emit one edge with
  `weight = config.graphify.star_import_weight` (default 0.5) and warn.

## 6. Acceptance Criteria

- **AC-1** `pytest tests/graphify/` passes with ≥ 85% line coverage on
  `src/graphify/` (enforced via `--cov-fail-under=85`).
- **AC-2** Graph determinism: 5 consecutive runs on the same fixture
  produce identical SHA-256 of the serialised `gpickle`.
- **AC-3** On the lazy-load broken fixture (Phase-0 acceptance gate), the
  resulting graph correctly marks the offending import with
  `lazy_load_flag=True`; the pytest sys.modules walker (`docs/PRD.md` §4.2)
  agrees.
- **AC-4** ruff clean; no file in `src/graphify/` exceeds 150 LOC
  (CLAUDE.md hard rule).
- **AC-5** Documented public API: only `GraphifyAdapter.build(path) →
  DiGraph` and `GraphifyAdapter.load(pickle_path) → DiGraph` are stable;
  everything else is internal.

## 7. ADR-002 Boundary

```python
class GraphifyAdapter(Protocol):
    def build(self, src_root: Path, *, seed: int = 0) -> nx.DiGraph: ...
    def load(self, pickle_path: Path) -> nx.DiGraph: ...
```

The local re-impl lives in `src/graphify/local/`. When the published
binary lands, we add `src/graphify/upstream/UpstreamGraphifyAdapter` that
satisfies the same Protocol and flip a single config flag
(`graphify.backend: local | upstream`). No call-site in the RL trainer,
reward shaper, or notebooks may import from `src/graphify/local/`
directly — they go through `GraphifyAdapter`. This is enforced by an
`import-linter` contract in CI.

## 8. Out of Scope

- Cross-language graphs (Python only for A4).
- Runtime/dynamic call graphs (we are explicitly static-only — see brief
  §1.2 "static analysis").
- Visualisation (handled by §F8 NetworkX/pyvis screenshot pipeline).

## 9. References

- Course brief §1.2 (graph definition)
- `docs/PRD.md` §3.1 F2 (GRAPHIFY requirement row)
- ADR-002 (Adapter boundary rationale)
- CLAUDE.md §1, §4, §6 (file-size, config, ruff)
