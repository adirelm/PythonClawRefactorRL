# ADR-001 — PythonClaw Shim Boundary

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-04 |
| **Decider** | solo developer (architect role per CLAUDE.md §1.4) |
| **Supersedes** | — |
| **Superseded by** | — (24h swap window; see *Consequences*) |

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
**nothing else**. It is gated behind `GraphifyAdapter` (see ADR-002) so
that the agent code never imports the shim directly — it talks to the
adapter, the adapter talks to either the shim or the real fork.

When the real PythonClaw URL + SHA land, we swap the import target inside
`GraphifyAdapter` (one config line) and delete `src/pythonclaw_shim/`.
Tests pin behavioural parity at the adapter seam, so the swap is a
mechanical, low-risk operation.

## 3. Justification

1. **Empty Day-3 git history is the worst grading signal.** The lecturer
   is a read-only collaborator on the repo (per user memory) and will see
   commit cadence directly. Visible iteration beats hidden waiting.
2. **The shim is intentionally small (≤300 LOC total across all shim
   files).** Each file still respects the global ≤150-LOC rule. Small =
   easy to throw away. We are explicitly **not** investing in shim
   features beyond what the adapter contract requires.
3. **ADR-002's adapter pattern fully isolates the swap.** Agent code,
   training loop, and evaluation harness are blind to which backend is
   live. The shim is replaceable without touching the RL stack.

## 4. Consequences

- **Positive:** Day 1 unblocked; commit history begins on schedule;
  adapter contract gets exercised against a real (if synthetic) backend
  before the fork lands, which de-risks the swap.
- **Negative:** Short-term double-implementation cost (shim + fork). Bound
  by the ≤300-LOC ceiling so the cost is capped.
- **Swap window:** **24 hours soft** from the moment the real PythonClaw
  URL + SHA are confirmed. If the swap is not done within 24h of
  confirmation, this ADR is amended (not superseded) to record *why* and
  to log the eventual swap commit SHA inline below.
- **Amendment slot:** *(to be filled on swap)* — Swap commit SHA: `____`
  — Date: `____` — Verifier: behavioural-parity test suite green.

## 5. Alternatives Considered

| # | Alternative | Verdict | Why rejected |
|---|---|---|---|
| a | **Wait** for real PythonClaw URL + SHA before any code lands | Rejected | Blocks Day 1, kills commit-cadence signal, no contingency if URL slips past Day 3. |
| b | Use a **completely different demo codebase** (skip PythonClaw entirely) | Rejected | Loses alignment with the brief's named target system; would force a §2.4 essay rewrite and break the V3 §5 traceability chain. |
| c | Mock PythonClaw **inside test fixtures only** (no shim module) | Rejected | Would leak mocking concerns into agent code and violate the adapter-seam isolation that ADR-002 depends on. |

## 6. V3 §5 Traceability

This decision is traced against the V3 plan's §5 (Risk + Contingency)
table. The row references below are exact:

| V3 §5 Row | Concern | How ADR-001 addresses it |
|---|---|---|
| §5.R1 | OQ-1: PythonClaw URL/SHA unconfirmed | Shim removes the blocker; 24h swap window bounds the deviation. |
| §5.R2 | 7-day deadline cannot absorb Day-1 slip | Bootstrap-now keeps the critical path on Day 1. |
| §5.R3 | Grading rubric rewards visible iteration | Commit cadence begins on Day 1, not Day 3+. |
| §5.R4 | Backend swap must be low-risk | ADR-002 adapter seam + parity tests make the swap mechanical. |
| §5.C1 | Contingency if real fork never materialises | Shim is a complete (if minimal) backend; submission is not blocked on the fork. |
| §5.C2 | Contingency if real fork API drifts from shim | Adapter contract is the single source of truth; drift is caught at the seam, not in agent code. |

## 7. Review Trigger

Re-open this ADR when **any** of the following occur:

- Real PythonClaw URL + SHA are confirmed (→ execute swap, fill amendment slot).
- 24h soft window elapses without a swap (→ amend with reason).
- Shim file count or LOC budget is breached (→ re-justify or split).
- ADR-002 adapter contract changes shape (→ re-verify parity tests still pin the seam).
