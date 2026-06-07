# PROMPTS.md — A4 Architect ↔ Implementer Trail (CLAUDE.md §1.4)

Per CLAUDE.md §1.4: the architect (solo developer) decides scope, architecture,
acceptance criteria; the AI (Claude) implements against approved specs.

## §1 — How to read
Each "Pass-N" section is a multi-agent workflow round. Decisions stay in §4
(Decision log); workflow registry in §5.

## §2 — Phase 0 (Planning + Review + Closure)

### Pass 1 — Grade-strategy 3-voice analysis
3 voices (Claude historical + Codex risk-locker + grader-sim) chose
BOOTSTRAP_NOW path + locked 10 architect-decisions (OQ1..OQ10).
Output: locked vendor PythonClaw shim, local GraphifyAdapter, tiktoken
cost metric, 5-seed multi-seed discipline, etc.

### Pass 2 — 20-agent deep-planning bundle
Foundation (1 agent) + 19 parallel doc agents (5 PRDs, 10 ADRs, PLAN,
TODO, STATE/ACTION designs, TRACE skeleton). 4 commits landed.

### Pass 3 — 10-agent planning review
4 Claude + 3 Codex + 3 grader-sim audited the planning bundle.
Result: NEEDS_REVISION, 14 critical + 30+ minor findings (reward drift,
GraphifyAdapter signature, Gymnasium-vs-brief tension).

### Pass 4 — 20-agent review closure
File-disjoint fix workflow with 6 canonical values embedded. Closed
all 14 critical; verifier wrote STATE/ACTION rewrites + commit landed
as 0ca9176. 5 Codex-targeted items were missed (Codex companion unavailable).

### Pass 5 — 5-agent Codex-gap cleanup
Claude-default fix workflow for the 5 Codex-missed items (PPO/GAE math,
ADR-004/008/010). Closed all gaps; commit 70464d7.

### Pass 6 — 20-agent final QA
4-group audit (content correctness, architectural integrity, doc quality,
submission package). Verdict: BLOCK_PUBLIC with 15 HIGH findings + 24 MEDIUM
+ 14 LOW. Dominant: betweenness 2-vs-3 drift (8 axes converged), banned λ-form
reward in PRD §1.3, 404 links in README, dangling ADR filenames.

### Pass 7 — 20-agent BLOCK→GO closure (THIS PASS)
Closing all 15 HIGH + 10 high-value MEDIUM findings before going public.

## §3 — Phase 1+ passes
(Pending — fill as we go.)

## §4 — Decision log
Every row here is a §1.4 *Human-decided* column entry.

| Date | Decision |
|---|---|
| 2026-06-03 | A3 self-grade target = 93 |
| 2026-06-04 | A4 self-grade target = 88-92 (honest framing per A1 lesson) — **SUPERSEDED 2026-06-08**: claim no numeric self-grade (brief does not request one); keep honest limitations only (QUALITY.md / PRD §7) |
| 2026-06-04 | A4 path: BOOTSTRAP_NOW with vendored PythonClaw shim (ADR-001) |
| 2026-06-04 | A4 algorithm: PPO+GAE via Stable-Baselines3 (brief §2.3 allows) |
| 2026-06-04 | A4 env: Custom Training Loop (brief §2.2 bans Gymnasium) |
| 2026-06-04 | A4 encoder: GraphSAGE primary, padding fallback (ADR-004/008) |
| 2026-06-04 | A4 reward: R_t = α·ΔMod + β·ΔCoh − γ·Coupling + P_skills (ADR-007) |
| 2026-06-04 | A4 betweenness: EXACTLY 2 calls per seed (brief §2.2; ADR-006) |
| 2026-06-04 | A4 convergence: ADR-010 dual-criterion (non-overlapping + entropy-slope) |
| 2026-06-04 | A4 seeds: 5 minimum {42, 7, 123, 314, 271} |
| 2026-06-04 | A4 §2.4 essay: 2500-3000 words, GRAPHIFY × AI agents topic (brief-verbatim) |

## §5 — Workflow registry
| Pass | Task ID | Agents | Tokens (est) |
|---|---|---|---|
| 1 | wq8vucq5m | 4 | ~100k |
| 2 | wzo3gzvwn | 21 | ~705k |
| 3 | wczb9xyxu | 11 | ~910k |
| 4 | wn7x1gig6 | 21 | ~810k |
| 5 | wn0veja33 | 6 | ~220k |
| 6 | whxf95rar | 21 | ~965k |
| 7 | (current) | 20 | (running) |
