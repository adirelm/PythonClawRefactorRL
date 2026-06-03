# ADR-002: GRAPHIFY — Local NetworkX Implementation Behind an Adapter

- **Status:** Accepted
- **Date:** 2026-06-04
- **Deciders:** Architect (human, §1.4 contract)
- **Implementer:** AI agent against this ADR
- **Supersedes:** —
- **Related:** ADR-001 (PythonClaw shim), OQ-2 (GRAPHIFY binary availability)

## Context

The ex04 brief §1.2 specifies that the refactor environment consumes a
directed graph `G = (V, E)` where `V` is the set of code symbols
(functions, classes, modules) and `E` encodes call/import/inheritance
edges. The brief names GRAPHIFY as the reference extractor but does not
guarantee a published binary or an installable package — this is logged
as **OQ-2** in `instructions/assignment-4/open_questions.md`.

Two failure modes follow from doing nothing:

1. **Binary path is gated.** If GRAPHIFY ships only as a private artifact,
   we cannot run reproducibly inside CI or on a grader's machine.
2. **Binary path produces no artifacts.** Even if it runs, the grader sees
   a black box: no math, no test surface, no code to review. The A2 and
   A3 rubrics rewarded *"we built it ourselves, here is the math"*; a
   shelled-out binary leaves nothing to grade.

## Decision

Define `GraphifyAdapter` as an abstract interface, then ship one concrete
implementation now and leave room for a second later.

```
src/graphify/
    __init__.py            # re-exports GraphifyAdapter, get_default()
    adapter.py             # abstract base class (≤80 LOC)
    local_impl.py          # NetworkX implementation (≤150 LOC)
    # local_binary.py      # FUTURE — only if GRAPHIFY binary lands
```

The adapter contract is a single method:

```python
class GraphifyAdapter(ABC):
    @abstractmethod
    def extract(self, source_dir: Path) -> nx.DiGraph: ...
```

`LocalImpl` walks `source_dir` with the `ast` module, builds nodes for
every top-level symbol, and emits edges for: (a) `Call` nodes inside a
function body, (b) `Import` / `ImportFrom` targets, (c) class bases in
`ClassDef`. Node attributes include `kind`, `loc`, `module_path`, and a
tiktoken (`cl100k_base`) token count — the headline tokenizer locked in
the grade-strategy round.

A future `LocalBinaryImpl` would wrap the GRAPHIFY CLI behind the same
`extract()` signature; **no agent code, no reward shaping code, and no
test fixture imports the concrete class** — they all depend on
`GraphifyAdapter`. The 24h swap window from ADR-001 applies.

## Alternatives Considered

| # | Alternative | Why rejected |
|---|---|---|
| A1 | Shell out to GRAPHIFY binary as the only path | OQ-2 unresolved; zero artifacts for grader; not reproducible in CI |
| A2 | Use `pyan3` / `code2flow` directly, no adapter | Locks us to one tool's quirks; can't swap; obscures the §1.2 math |
| A3 | Hand-rolled regex/import scan, no `ast` | Loses scope resolution; misses dynamic calls; fails on decorators |
| A4 | **Adapter + local NetworkX `ast` impl now, binary impl later** | ✅ Chosen — testable, swappable, matches §1.2 math, scores on rubric |
| A5 | Adapter + multiple impls shipped simultaneously | Premature; doubles test surface before we know if the binary exists |

## Consequences

**Positive.**
- The `extract()` contract is the only seam the RL agent sees; swapping
  implementations is a one-line change in `get_default()`.
- The NetworkX impl is fully covered by unit tests on tiny fixture
  repos — the 85% coverage gate from CLAUDE.md is reachable.
- §2.4 essay can cite the local impl directly ("here is `local_impl.py`
  line 42, here is the AST visitor for `Call` edges") instead of waving
  at a binary.
- Betweenness-centrality experiments (≥5 seeds, mean±std+95% CI per the
  locked decisions) run on a graph we control end-to-end.

**Negative.**
- Local impl will miss some dynamic dispatch (e.g. `getattr`-based
  calls). Documented as a known limitation in `local_impl.py` docstring
  and §2.4 essay's "threats to validity" section.
- If GRAPHIFY publishes later with better recall, we must run a
  comparison study to justify keeping the local impl. ADR-003 will
  cover that swap if it triggers.

**Neutral.**
- Two ADRs (001, 002) now describe adapter shims. This is a pattern,
  not a smell — both isolate the project from upstream availability
  risk.

## V3 Traceability

| Source | Anchor | This ADR addresses |
|---|---|---|
| ex04 brief §1.2 | "G = (V, E) where V = symbols, E = relations" | Decision §, `extract()` contract returns `nx.DiGraph` |
| ex04 brief §1.4 | "AI is implementer, human is architect" | ADR is human-authored; agent codes against it |
| OQ-2 | "GRAPHIFY binary availability uncertain" | Alternatives A1, A4; ADR-003 trigger |
| Grade-strategy round | "we built it ourselves, here is the math" | Decision §, A1 rejection, §2.4 essay hook |
| Lecture 7 (PPO+GAE) | Agent acts on state derived from graph | Adapter returns `nx.DiGraph`, agent code unchanged across swaps |
| CLAUDE.md §1 | ≤150 LOC per .py | `local_impl.py` size budget noted in Decision § |
| CLAUDE.md §2 | TDD + 85% coverage | Adapter testable via fixture repos |
| Locked decision: tiktoken `cl100k_base` | Headline tokenizer | Node attribute `token_count` produced by local impl |

## Open follow-ups

- **OQ-2 monitoring** — re-check GRAPHIFY availability at Phase 3 gate;
  if published, evaluate `local_binary.py` under the 24h swap window.
- **Comparison harness** — if both impls ever coexist, the harness from
  ADR-001 §"swap window" applies symmetrically.
