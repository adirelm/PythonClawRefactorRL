# ADR-001 — PythonClaw Shim Boundary

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-04 |
| **Decider** | solo developer (architect role per CLAUDE.md §1.4) |
| **Supersedes** | — |
| **Superseded by** | — (24h swap window; see §6.1) |

## 1. Context

At planning time on **2026-06-03**, the canonical PythonClaw GitHub URL and
pinned commit SHA were **unconfirmed** (logged as Open Question **OQ-1** in
the planning notes). The assignment brief gives a **7-day hard deadline**;
losing Day 1 to wait on an external dependency is not affordable.

A grade-strategy round was run with three independent voices (architect,
risk-officer, deadline-officer). All three converged on the same
recommendation: **bootstrap immediately with a vendored shim**, rather than
wait for the real PythonClaw fork. The grading rubric explicitly rewards
visible iteration cadence; an empty `git log` on Day 3 against a
collaborator-lecturer is the worst possible signal.

## 2. Decision

We vendor a minimal, deterministic stand-in for PythonClaw's Skills module
at:

```
src/pythonclaw_shim/
```

The shim implements exactly the surface area the agent code calls into
(skill registry, deterministic skill outcomes, callable signatures), and
**nothing else**.

**Boundary delegation.** This ADR sets the *vendor-vs-shim* policy and the
24h swap window only. The *runtime seam* that isolates agent code from the
shim package is **out of scope here** and is owned by **ADR-011
(SkillsAdapter)**. Agent code never imports `pythonclaw_shim` directly —
it depends on the `SkillsAdapter` Protocol defined in ADR-011, and a
factory binds that Protocol to either `ShimSkillsAdapter` (today) or
`RealSkillsAdapter` (post-swap).

Earlier drafts of this ADR mistakenly described the shim as "gated behind
`GraphifyAdapter`". That was a category error: `GraphifyAdapter`
(ADR-002) extracts the code-symbol graph `G = (V, E)` and has no
relationship to the Skills runtime. The two adapters live at orthogonal
seams and must not be conflated. See **ADR-011 §1** for the corrected
boundary diagram.

When the real PythonClaw URL + SHA land, the swap is the mechanical
sequence described in §6.1 below — the `SkillsAdapter` factory flips
from `ShimSkillsAdapter` to `RealSkillsAdapter`, parity tests
(`tests/test_skills_adapter_parity.py`, defined in ADR-011 §4) confirm
behavioural equivalence, and `src/pythonclaw_shim/` is deleted in the
same commit.

## 3. Justification

1. **Empty Day-3 git history is the worst grading signal.** The lecturer
   is a read-only collaborator on the repo (per user memory) and will see
   commit cadence directly. Visible iteration beats hidden waiting.
2. **The shim is intentionally small (≤300 LOC total across all shim
   files).** Each file still respects the global ≤150-LOC rule. Small =
   easy to throw away. We are explicitly **not** investing in shim
   features beyond what the adapter contract requires.
3. **ADR-011's `SkillsAdapter` Protocol fully isolates the swap.** Agent
   code, training loop, and evaluation harness depend on the Protocol,
   not on the shim package. The shim is replaceable without touching the
   RL stack. (ADR-002's `GraphifyAdapter` is a separate seam covering
   graph extraction — it is *not* the swap point for Skills.)

## 4. Consequences

- **Positive:** Day 1 unblocked; commit history begins on schedule;
  the `SkillsAdapter` Protocol (ADR-011) gets exercised against a real
  (if synthetic) backend before the fork lands, which de-risks the swap.
- **Negative:** Short-term double-implementation cost (shim + fork). Bound
  by the ≤300-LOC ceiling so the cost is capped.
- **Swap window:** **24 hours soft** from the moment the real PythonClaw
  URL + SHA are confirmed. Mechanics are spelled out in §6.1. If the swap
  is not done within 24h of confirmation, this ADR is amended (not
  superseded) to record *why* and to log the eventual swap commit SHA in
  the amendment slot below.
- **Amendment slot:** *(to be filled on swap)* — Swap commit SHA: `____`
  — Date: `____` — Verifier: `tests/test_skills_adapter_parity.py` green.

## 5. Alternatives Considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **Wait** for real PythonClaw URL + SHA before any code lands | Rejected | Blocks Day 1, kills commit-cadence signal, no contingency if URL slips past Day 3. |
| b | Use a **completely different demo codebase** (skip PythonClaw entirely) | Rejected | Loses alignment with the brief's named target system; would force a §2.4 essay rewrite and break the V3 §5 traceability chain. |
| c | Mock PythonClaw **inside test fixtures only** (no shim module) | Rejected | Would leak mocking concerns into agent code and violate the adapter-seam isolation that ADR-011 (`SkillsAdapter`) depends on. |

## 6. Swap Window — When the Real PythonClaw URL Arrives

### 6.1 Mechanics (deterministic, ordered)

The 24h soft window from §4 is operationalised by the following ordered
sequence. Each step must succeed before the next begins; any failure
freezes the swap and the amendment slot in §4 is filled with the failure
reason instead of a swap SHA.

1. **Clone the real PythonClaw fork at the confirmed SHA** into
   `vendor/pythonclaw/` (pinned via submodule or copy-with-SHA-recorded;
   the call is OQ-1's resolution).
2. **Implement `RealSkillsAdapter`** in
   `src/skills/real_skills_adapter.py` against the `SkillsAdapter`
   Protocol from ADR-011 §2. The implementation reads from
   `vendor/pythonclaw/` instead of `src/pythonclaw_shim/`. No other agent
   code changes.
3. **Flip the factory flag** in `config/config.yaml#skills.backend` from
   `shim` to `real`. The factory in ADR-011 §3 reads this flag and
   returns the `RealSkillsAdapter` instance.
4. **Re-run the baseline metrics** (`uv run pytest tests/ --cov` and
   the `≥5-seed` betweenness-centrality + reward-curve harnesses) so
   that pre-swap and post-swap numbers are directly comparable. Save
   both PNGs side by side in `results/swap_baseline/`.
5. **Run the parity test** (`tests/test_skills_adapter_parity.py`,
   specified in ADR-011 §4). Skill enumeration cardinality, ordering,
   and metadata structure must match between `ShimSkillsAdapter` and
   `RealSkillsAdapter` on the bundled fixture tree. A diff anywhere is
   a hard fail and stops the swap.
6. **Delete `src/pythonclaw_shim/`** in the same commit that flips the
   factory flag. The shim is no longer reachable from any import.
7. **Commit** with subject `Phase 0 fix — swap PythonClaw shim → real
   adapter` and record the commit SHA in §4's amendment slot.
8. **Re-tag** ADR-001 status from `Accepted` to `Implemented` with the
   swap date.

### 6.2 Why these steps, in this order

Steps 1–3 land the real backend *behind the existing Protocol* before
anything in the RL stack notices. Steps 4–5 are the verification gate:
without identical baselines and a green parity test, the swap is not
safe. Step 6 removes the shim only after verification — never before —
so a rollback (re-flip the flag) remains possible up to step 5.

## 7. V3 §5 Traceability

This decision is traced against the V3 plan's §5 (Risk + Contingency)
table. The row references below are exact:

| V3 §5 Row | Concern | How ADR-001 addresses it |
|---|---|---|
| §5.R1 | OQ-1: PythonClaw URL/SHA unconfirmed | Shim removes the blocker; 24h swap window (§6.1) bounds the deviation. |
| §5.R2 | 7-day deadline cannot absorb Day-1 slip | Bootstrap-now keeps the critical path on Day 1. |
| §5.R3 | Grading rubric rewards visible iteration | Commit cadence begins on Day 1, not Day 3+. |
| §5.R4 | Backend swap must be low-risk | ADR-011 `SkillsAdapter` Protocol + parity tests make the swap mechanical. |
| §5.C1 | Contingency if real fork never materialises | Shim is a complete (if minimal) backend; submission is not blocked on the fork. |
| §5.C2 | Contingency if real fork API drifts from shim | ADR-011 Protocol is the single source of truth; drift is caught at the seam, not in agent code. |

## 8. Review Trigger

Re-open this ADR when **any** of the following occur:

- Real PythonClaw URL + SHA are confirmed (→ execute swap per §6.1, fill amendment slot).
- 24h soft window elapses without a swap (→ amend with reason).
- Shim file count or LOC budget is breached (→ re-justify or split).
- ADR-011 `SkillsAdapter` Protocol shape changes (→ re-verify parity tests still pin the seam).
