# PRD-SKILLS — PythonClaw Skills Module Shim

> Component PRD for the Skills module. Realises `docs/PRD.md` §3 / F1.
> Status: DRAFT. Author lens: external researcher reading brief §1.1 cold.

## 1. Purpose

Brief §1.1 defines a Skill as a three-layer object discovered and progressively
loaded by the host:

| Layer | Contents | Loaded when |
|-------|---------------------------------|------------------------------|
| L1    | Metadata (name, version, tags)  | discovery — always           |
| L2    | Instructions (prompt body)      | the agent selects the skill  |
| L3    | Resources (code, files, models) | a tool call needs them       |

The brief is explicit that L2 and L3 must be **lazy**: instantiating a `Skill`
handle and reading `skill.metadata` MUST NOT import the resource payload or
pay its token cost. Refactor candidates that flatten the layers (eager import
of L3 at discovery) are by definition *broken*, and the assignment treats
detecting that regression as a first-class evaluation signal.

Skills are the unit of action our RL agent picks at every step, so the
contract below is load-bearing for the entire reward pipeline.

## 2. Why a SHIM (not the real thing yet)

Per **OQ-1** (open question, `docs/PRD.md`): the upstream PythonClaw package URL
on the brief is unverified at submission time — pip resolution returned 404
on two candidate names, and the brief PDF does not pin a hash. **ADR-001**
draws the boundary: we depend on a `pythonclaw_shim` package we own, expose
the surface in §3 below, and keep a **24h swap window** in which the real
package can replace the shim *without any caller-side code change*.

This is not a stub for convenience. It is a deliberate seam so that the
grader can: (a) run the full pipeline today with the shim, and (b) re-run it
with the vendored upstream when the URL is confirmed, and observe identical
public behaviour.

## 3. Shim Contract (minimal API surface)

```python
class Skill:
    name: str
    version: str
    metadata: dict          # L1 — cheap, always available
    @property
    def instructions(self) -> str: ...   # L2 — loaded on first access
    @property
    def resources(self) -> Resources: ...# L3 — loaded on first access
    def estimated_tokens(self, layer: int) -> int: ...

class SkillRegistry:
    def discover(self, path: Path) -> list[Skill]: ...     # L1 only
    def get(self, name: str) -> Skill: ...
    def __len__(self) -> int: ...
```

Notes on the surface:

- `discover()` walks a directory tree and returns handles populated with L1
  only. It MUST NOT touch L2/L3 files on disk.
- `instructions` and `resources` are properties so caller code reads like
  attribute access, but the shim controls *when* the read happens.
- `estimated_tokens(layer)` is the headline metric for §2.4 of the report and
  is read by the RL state featuriser; it must work pre-load.
- The host-side caller bridge — the `SkillsAdapter` Protocol that exposes
  this surface to the RL env and featuriser — is owned by **ADR-011**
  ("SkillsAdapter"), forthcoming and authored alongside the ADR-001 fix.
  PRD-SKILLS defines *what* the shim guarantees; ADR-011 defines *how*
  the rest of the system talks to it.

## 4. Lazy-load semantics (the invariant we ship)

The single behavioural invariant the shim guarantees:

> reading `skill.metadata` and `skill.estimated_tokens(2)` MUST NOT cause
> L2 or L3 payloads to be imported, decoded, or counted.

(Verbatim canonical invariant — do not paraphrase. Applies to any freshly
discovered `skill` handle returned by `SkillRegistry.discover()`.)

Practically this means the shim stores L2/L3 as file paths + byte offsets at
discovery time, and only opens the files inside the property getter. A
module-level cache memoises after first access (so repeated reads are free).

## 5. Operational test for "broken lazy-loading" (OQ-5, ADR-005)

The full operational semantics — what counts as "broken", what we measure,
and how we grade — are owned by **ADR-005** ("Operational Semantics of
Broken Lazy-Loading"). This section restates the test so the PRD is
self-contained; ADR-005 is the source of truth for any drift.

Detection is a real test, not a vibe:

1. Snapshot `set(sys.modules)` before `registry.discover(...)`.
2. Run discovery over the full skills tree.
3. Re-snapshot `sys.modules`. The delta MUST NOT contain any module from
   the L3 resource namespace (e.g. `pythonclaw_shim.resources.*`).
4. Compute the cl100k_base token count of all *touched* file bytes. The
   **P95** across ≥5 seeds MUST be ≤ the L1-only budget declared in
   `config/skills.yaml` (currently 2,048 tokens / discovery sweep).
5. Then access `.instructions` on one skill; re-run step 3-4 and assert
   the L2 namespace IS now present and tokens rose by ≥ the L2 floor.

This double check (`sys.modules` delta AND token P95) is the canary the
assignment uses to grade whether a candidate refactor preserved the layered
loading contract. A refactor that "looks clean" but eagerly imports
resources fails step 3 even if functional tests pass.

## 6. Acceptance criteria

A1. `SkillRegistry.discover()` returns ≥1 skill for the bundled fixture
    directory and 0 modules from the resource namespace appear in
    `sys.modules` after the call.

A2. Reading `.metadata` on every discovered skill stays under the
    L1 token budget at P95 across 5 seeds (seeds 0..4, deterministic).

A3. Progressive load works end-to-end: L1 → `.instructions` (L2) →
    `.resources` (L3) each move the `sys.modules` delta forward by exactly
    the expected namespace prefix.

A4. Determinism: with a fixed seed, `discover()` returns skills in a stable
    order (sorted by `(name, version)`) so RL rollouts are reproducible.

A5. The shim raises `SkillNotFound` (not bare `KeyError`) on `registry.get`
    for an unknown name, so the RL env can map it to a terminal penalty.

## 7. 24h swap window

When the upstream PythonClaw URL is confirmed:

1. Vendor the upstream package under `vendor/pythonclaw/` (pinned commit).
2. Re-export the §3 surface from `pythonclaw_shim/__init__.py` so imports
   stay `from pythonclaw_shim import Skill, SkillRegistry`.
3. Re-run the test in §5. If it still passes, the swap is invisible to
   callers; if it fails, upstream itself broke lazy-loading and we file
   that as a finding (it is exactly the regression the assignment hunts).
4. ADR-001 gets an "implemented" timestamp; no other doc changes.

## 8. References

- Brief §1.1 — three-layer Skill model, lazy-load requirement.
- `docs/PRD.md` §3 / F1 — Skills module as a required feature of the refactor
  target; sets the boundary this PRD fills in.
- ADR-001 — shim-vs-vendor decision and 24h swap window.
- ADR-005 — operational semantics of "broken lazy-loading" (owns the §5 test).
- ADR-011 — `SkillsAdapter` Protocol (forthcoming, paired with ADR-001 fix).
- OQ-1 — unverified upstream URL.
- OQ-5 — "broken lazy-loading" detection methodology.
