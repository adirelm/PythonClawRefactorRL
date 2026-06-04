# ADR-011 — SkillsAdapter Protocol (Runtime Seam for PythonClaw Skills)

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-04 |
| **Decider** | solo developer (architect role per CLAUDE.md §1.4) |
| **Supersedes** | — |
| **Superseded by** | — |
| **Related** | ADR-001 (shim boundary), PRD-SKILLS (Skills component PRD), brief §1.1 (Skill three-layer model), brief §2.1 (Skills mandate) |

## 1. Context

The brief (§1.1) defines a Skill as a three-layer object — L1 metadata,
L2 instructions, L3 resources — with a lazy-load invariant: discovery
touches only L1, and L2/L3 are loaded on first attribute access.
**PRD-SKILLS** spells out the contract; **ADR-001** decides the
vendor-vs-shim policy and the 24h swap window.

What ADR-001 deliberately leaves *out of scope* is the **runtime seam**
between agent code (RL policy, environment, reward pipeline) and the
backing Skills package. Earlier drafts of ADR-001 confused that seam
with the `GraphifyAdapter` (ADR-002). That was a category error:

```
ADR-002 GraphifyAdapter   — extracts G = (V, E) from a source tree
                            (returns nx.DiGraph; no Skills awareness)

ADR-011 SkillsAdapter     — exposes Skill enumeration, metadata, lazy
                            L2/L3 access (no graph awareness)
```

The two adapters live at orthogonal seams. `GraphifyAdapter` is read by
the state featuriser; `SkillsAdapter` is read by the action handler and
the lazy-load monitor. Confusing them would make `GraphifyAdapter`
responsible for runtime behaviour it has no business knowing about.

This ADR defines the missing seam.

## 2. Decision — the `SkillsAdapter` Protocol

We define a typing.Protocol that agent code depends on. The Protocol
exposes a minimal `Skill` data class, a `SkillRegistry` for enumeration
and lookup (matching the PRD-SKILLS §3 surface verbatim), and
lazy-property access for L2/L3. The standout behavioural invariant —
reading `skill.metadata` and `skill.estimated_tokens(2)` MUST NOT
trigger L2/L3 load (PRD-SKILLS §4) — is what every field below is
shaped to preserve.

```python
# src/skills/adapter.py
from pathlib import Path
from typing import Protocol, runtime_checkable

class Skill(Protocol):
    name: str               # stable identifier (e.g. "refactor.split-fn")
    version: str            # semver string; used in the sort key (PRD-SKILLS A4)
    metadata: dict          # L1 payload — cheap, always available
    layer: int              # currently loaded depth: 1, 2, or 3

    @property
    def instructions(self) -> str: ...        # L2 — lazy
    @property
    def resources(self) -> "Resources": ...   # L3 — lazy

    def estimated_tokens(self, layer: int) -> int: ...
    # cl100k_base count for the requested layer; MUST work pre-load
    # (layer=1 and layer=2 callable on a freshly discovered handle
    # without triggering L2/L3 import — PRD-SKILLS §4 invariant).

@runtime_checkable
class SkillRegistry(Protocol):
    # PRD-SKILLS §3 surface — what agent code (A1, featuriser) imports.
    def discover(self, path: Path) -> list[Skill]: ...   # L1 only
    def get(self, name: str) -> Skill: ...               # raises SkillNotFound
    def __len__(self) -> int: ...

    # Lazy-load monitor seam (PRD-SKILLS §5) — explicit load entry
    # points that bypass the `Skill` properties, so the monitor can
    # force a layer load without the property side-effect tripping
    # the very check it is performing.
    def load_metadata(self, name: str) -> dict: ...      # L1 on demand
    def load_instructions(self, name: str) -> str: ...   # L2 on demand
    def load_resources(self, name: str) -> "Resources": ...  # L3 on demand

@runtime_checkable
class SkillsAdapter(Protocol):
    registry: SkillRegistry
```

Key shape decisions:

- **`Skill.name` + `Skill.version`** match PRD-SKILLS §3 verbatim
  (earlier drafts of this ADR used `id`; that was a paraphrase and
  diverged from the PRD surface that agent code actually imports).
  The pair also feeds the `(name, version)` sort key required by
  PRD-SKILLS A4 for deterministic rollout order.
- **`Skill.metadata: dict`** is the L1 payload itself, not a flag.
  PRD-SKILLS §4 names this attribute as the canonical L1 access path
  (`reading skill.metadata ... MUST NOT cause L2 or L3 payloads to be
  imported`), so the Protocol exposes it directly. This is the
  standout behavioural invariant the PRD calls out, and §2 above pins
  the field shape that makes it testable.
- **`Skill.layer`** advances `1 → 2 → 3` as the lazy-properties are
  touched. The lazy-load monitor (PRD-SKILLS §5) reads this field to
  decide whether the `P_skills` penalty (see CLAUDE.md "Reward
  equation": `P_skills = -5.0` when the lazy-load monitor detects a
  break) should fire.
- **`estimated_tokens(layer: int) -> int`** is a **method**, not an
  attribute — taking the layer index as an argument. This matches
  PRD-SKILLS §3 (`estimated_tokens(self, layer: int) -> int`) and
  PRD-SKILLS §4 (which invokes `skill.estimated_tokens(2)` as the
  canonical pre-load token query). Counts use `cl100k_base` (the
  tokenizer pinned by ADR-003) and MUST be answerable for L1/L2/L3
  without forcing the queried layer to load. This is what guarantees
  the L1 token budget from PRD-SKILLS §5 step 4 can hold.
- **`SkillRegistry.discover` / `.get` / `__len__`** live on the
  registry because that is where PRD-SKILLS §3 puts them — agent code
  (A1) and the state featuriser import them from the registry, not
  from a separate adapter façade. The `SkillsAdapter` Protocol is the
  thin wrapper that holds a `registry` field; callers reach the PRD
  surface via `adapter.registry.discover(...)`.
- **`load_metadata` / `load_instructions` / `load_resources`** are
  explicit registry methods *in addition to* the `Skill` lazy
  properties. The property form is for ergonomic agent code
  (`skill.instructions`); the registry form is for the lazy-load
  monitor, which needs to invoke the L2/L3 load without going through
  the property (otherwise the act of testing would itself trip the
  monitor it is testing).
- **`@runtime_checkable`** is set on `SkillsAdapter` and
  `SkillRegistry` so that `isinstance(obj, SkillsAdapter)` works in
  the factory (§3) — useful both for type guards and for the parity
  test (§4) which verifies *both* concrete adapters satisfy the
  Protocol.

> Cross-reference: every field above is the seam side of a contract
> defined in **PRD-SKILLS §3** (API surface) and **PRD-SKILLS §4** (the
> standout behavioural invariant: `skill.metadata` and
> `skill.estimated_tokens(2)` MUST NOT load L2/L3). If PRD-SKILLS §3 or
> §4 drift, §2 here drifts with them — see §8 Review Trigger.

## 3. Two Implementations + Factory Swap Mechanism

```
src/skills/
    __init__.py              # re-exports factory get_skills_adapter()
    adapter.py               # SkillsAdapter / SkillRegistry / Skill Protocols (≤80 LOC)
    shim_skills_adapter.py   # ShimSkillsAdapter (today) — backs onto src/pythonclaw_shim/
    real_skills_adapter.py   # RealSkillsAdapter (future) — backs onto vendor/pythonclaw/
    factory.py               # get_skills_adapter(cfg) → SkillsAdapter (≤60 LOC)
```

### 3.1 `ShimSkillsAdapter`

Backs onto the vendored `src/pythonclaw_shim/` package. Implements every
Protocol method by delegating to the shim's `SkillRegistry`. This is the
adapter the assignment ships with on Day 1 and runs against until OQ-1
resolves.

### 3.2 `RealSkillsAdapter`

Backs onto `vendor/pythonclaw/` (cloned at the confirmed SHA per
ADR-001 §6.1 step 1). Same Protocol surface, different backend. Lands
only when the URL is confirmed and the swap sequence in ADR-001 §6.1
is executed.

### 3.3 Factory + config flag

```python
# src/skills/factory.py
from pathlib import Path
from src.utils.config import load_config
from src.skills.adapter import SkillsAdapter
from src.skills.shim_skills_adapter import ShimSkillsAdapter
from src.skills.real_skills_adapter import RealSkillsAdapter

def get_skills_adapter(cfg_path: Path | None = None) -> SkillsAdapter:
    cfg = load_config(cfg_path)
    backend = cfg["skills"]["backend"]   # "shim" | "real"
    if backend == "shim":
        return ShimSkillsAdapter()
    if backend == "real":
        return RealSkillsAdapter()
    raise ValueError(f"unknown skills.backend: {backend!r}")
```

The factory reads `config/config.yaml#skills.backend`. This is the
*one line* that flips during the ADR-001 §6.1 swap (step 3). Agent
code calls `get_skills_adapter()` once at construction time and never
imports either concrete class directly.

## 4. Parity Test — `tests/test_skills_adapter_parity.py`

A new test asserts that `ShimSkillsAdapter` and `RealSkillsAdapter`
agree on the externally observable surface. This is the gate that
ADR-001 §6.1 step 5 hangs on.

```python
# tests/test_skills_adapter_parity.py (sketch — owned by this ADR)
import pytest
from src.skills.shim_skills_adapter import ShimSkillsAdapter
from src.skills.real_skills_adapter import RealSkillsAdapter
from src.skills.adapter import SkillsAdapter

FIXTURE_ROOT = ...   # bundled tiny skills tree, deterministic

@pytest.fixture
def shim() -> SkillsAdapter:
    return ShimSkillsAdapter()

@pytest.fixture
def real() -> SkillsAdapter:
    return RealSkillsAdapter()

def test_both_satisfy_protocol(shim, real):
    assert isinstance(shim, SkillsAdapter)
    assert isinstance(real, SkillsAdapter)

def test_enumeration_cardinality_matches(shim, real):
    assert len(shim.registry.discover(FIXTURE_ROOT)) == \
           len(real.registry.discover(FIXTURE_ROOT))

def test_enumeration_order_matches(shim, real):
    shim_names = [(s.name, s.version) for s in shim.registry.discover(FIXTURE_ROOT)]
    real_names = [(s.name, s.version) for s in real.registry.discover(FIXTURE_ROOT)]
    assert shim_names == real_names   # sorted by (name, version) per PRD-SKILLS A4

def test_metadata_structure_matches(shim, real):
    for name, _ in [(s.name, s.version) for s in shim.registry.discover(FIXTURE_ROOT)]:
        s_meta = shim.registry.get(name).metadata   # L1 dict, no lazy property load
        r_meta = real.registry.get(name).metadata
        assert set(s_meta) == set(r_meta), f"metadata keyset diff for {name}"
```

The parity test runs in CI against the bundled fixture tree. Before the
swap, only the shim-only assertions run (the `real` fixture skips); on
swap day, both fixtures resolve and the full matrix runs. A diff
anywhere is a hard fail and stops the ADR-001 §6.1 sequence at step 5.

What the parity test deliberately does *not* assert:

- Equality of L2/L3 payload bytes (the shim is by design a minimal
  stand-in; payload content can differ).
- Equality of `estimated_tokens` magnitudes (token counts will diverge
  once the real backend loads real instructions/resources).

The parity test pins **structure**, not **content** — exactly what the
swap window needs to verify that the seam is intact.

## 5. Alternatives Considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| A | Have agent code import `pythonclaw_shim` directly | Rejected | The whole point of ADR-001's swap window is that nothing outside the adapter knows which backend is live. Direct imports defeat the seam. |
| B | Reuse `GraphifyAdapter` (ADR-002) for the Skills runtime | Rejected | Category error — graph extraction and Skills runtime are orthogonal. This was the bug being fixed when ADR-011 was created. |
| C | Use a concrete base class instead of `typing.Protocol` | Rejected | `Protocol` keeps the dependency direction inverted (concrete classes depend on the Protocol, not vice versa). With `@runtime_checkable` we still get `isinstance` for the factory. |
| D | Lazy-load through a global mutable registry | Rejected | Hidden global state would make the parity test unstable across test runs and break the lazy-load monitor's `sys.modules` delta check (PRD-SKILLS §5 step 3). |
| E | Skip the Protocol; just define `ShimSkillsAdapter` + a typedef | Rejected | A Protocol gives a single typed seam the lazy-load monitor and the action handler can both depend on without coupling to a concrete class. |

## 6. Consequences

**Positive.**
- Agent code is fully insulated from the shim-vs-real choice — the
  swap is a one-line config change at runtime, not a code change.
- Parity test gives a binary go/no-go signal for the swap, satisfying
  ADR-001 §6.1 step 5.
- Lazy-load monitor (PRD-SKILLS §5) has a clean hook: `Skill.layer`
  + `registry.load_*` methods expose exactly what the monitor needs to
  observe.
- `GraphifyAdapter` no longer carries Skills responsibilities — it can
  stay focused on returning `nx.DiGraph` per ADR-002.

**Negative.**
- One more file/Protocol to maintain in `src/skills/`. Bounded — each
  file stays ≤150 LOC per CLAUDE.md hard rule.
- Parity test cannot run end-to-end until `RealSkillsAdapter` lands;
  until then, only `test_both_satisfy_protocol` runs in single-backend
  mode against the shim.

**Neutral.**
- Three ADRs (001 shim-vs-vendor, 002 graph adapter, 011 skills
  adapter) describe seams. This is the right number for a system that
  has to swap one upstream dependency safely.

## 7. Traceability

| Source | Anchor | This ADR addresses |
|---|---|---|
| brief §1.1 | three-layer Skill model, lazy-load requirement | `Skill.layer`, lazy properties, monitor hook |
| brief §2.1 | Skills mandate in deliverables | `SkillsAdapter` is the canonical entry point |
| PRD-SKILLS §3 | minimal API surface (`Skill`, `SkillRegistry`) | §2 Protocol matches verbatim — `name/version/metadata/instructions/resources/estimated_tokens(layer)` on `Skill`; `discover/get/__len__` on `SkillRegistry` |
| PRD-SKILLS §4 | standout invariant — `skill.metadata` + `skill.estimated_tokens(2)` MUST NOT load L2/L3 | §2 lazy-property + `estimated_tokens(layer:int)` method shape is what makes the invariant testable |
| PRD-SKILLS §5 | broken-lazy-load detection method | `registry.load_metadata/load_instructions/load_resources` are the monitor's seam (bypass the `Skill` properties so the test does not trigger the very condition it checks) |
| ADR-001 §2 | shim boundary delegation | this ADR owns the runtime seam ADR-001 delegates |
| ADR-001 §6.1 | swap mechanics | §3 factory + §4 parity test are the verifier ADR-001 step 5 cites |
| ADR-002 | `GraphifyAdapter` orthogonality | §1 boundary diagram; §5 alt B rejection |
| ADR-003 | tiktoken `cl100k_base` | `Skill.estimated_tokens` uses it |
| CLAUDE.md §1 | ≤150 LOC per .py | §3 file budget noted (adapter.py ≤80, factory.py ≤60) |
| CLAUDE.md §2 | TDD + 85% coverage | §4 parity test is part of the gate |
| CLAUDE.md "Reward equation" | `P_skills = -5.0` lazy-load penalty | `Skill.layer` is the monitor signal that triggers it |

## 8. Review Trigger

Re-open this ADR when **any** of the following occur:

- ADR-001 §6.1 swap executes (→ confirm parity test runs full matrix,
  update §4 narrative from "only shim today" to "both backends live").
- PRD-SKILLS §3 surface changes (→ re-sync §2 Protocol shape).
- Lazy-load monitor finds a case where `Skill.layer` is insufficient
  (→ extend Protocol with a per-layer-load timestamp or counter).
- `GraphifyAdapter` ever needs to consume Skill metadata (→ STOP; this
  is the category error this ADR exists to prevent; revisit with
  architect sign-off per CLAUDE.md §1.4 before any code lands).
