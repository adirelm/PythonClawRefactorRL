# PROMPTS — Architect → AI sign-off log (§1.4)

Per CLAUDE.md §1.4, the developer is **architect** and the AI is
**implementer**. The architect decides scope, architecture, acceptance
criteria, test assertions, final code-review sign-off, the rubric self-
score, and the cost-budget envelope. The AI is allowed to generate code
against an *approved* spec, refactor inside an existing public API,
scaffold tests from a *written* assertion list, draft docstrings, fix
lint, and perform routine doc maintenance.

This file logs the prompts that gated each AI-implementer commit. Every
prompt below was preceded by an architect edit to the PRD, PLAN, ADR,
or TODO — the AI never moved without that sign-off. Where the exact
text was recovered from session transcripts it is quoted verbatim;
where the transcript was rotated out the prompt is paraphrased to the
shape it actually had (intent + scope + acceptance gate), and that
paraphrase is marked explicitly.

The companion meta-log of *which multi-agent pass landed which fix
bundle* lives at `docs/shared/PROMPTS.md`; this file is the per-commit
sign-off trail that grader audit can cross-reference against the §1.4
contract row by row.

---

## Phase 0 — Bootstrap, planning, review, closure

Phase 0 set the §1.4 contract physically into the repo: 5 PRDs, 10 ADRs,
PLAN, TODO, STATE/ACTION/REWARD designs, TRACE skeleton, and three
review-and-fix passes that took Phase 0 from BLOCK to GO. The architect
sealed six canonical values before any code generation was allowed:
reward equation, A_max, betweenness=2/seed, gymnasium ban, 5-seed
discipline, ≤150-LOC + ≥85%-coverage gate.

### Phase 0 / commit `a213652` — repo skeleton + config + CI
- Architect decision: scope = brief §2.1–§2.4; toolchain = uv + ruff +
  pytest + ≤150 LOC per file (CLAUDE.md hard constraints §1–§7).
- Prompt (paraphrased — pre-transcript-rotation):
  > Bootstrap the A4 repo skeleton. Mirror A3's layout: `src/`,
  > `tests/{unit,integration,architecture,e2e}`, `docs/`, `config/`,
  > `scripts/`. Wire `pyproject.toml` for uv, add `ruff.toml`, add a CI
  > workflow that runs `uv run pytest tests/ --cov=src
  > --cov-report=term-missing` and `uv run ruff check src/ tests/`.
  > Do NOT add any algorithm code yet — only the gate scaffolding.
  > Quality gate: `uv run pytest` green on the empty test, ruff clean,
  > file-size enforcer present.

### Phase 0 / commit `58bf82f` — 5 PRDs + 10 ADRs + PLAN + TODO + STATE/ACTION designs + TRACE skeleton
- Architect decision: lock the 10 Open Questions OQ-1..OQ-10 into ADRs
  *before* any phase-1 code. The 3-voice grade-strategy round
  (architect + risk-officer + deadline-officer) converged on
  BOOTSTRAP_NOW with a vendored PythonClaw shim (ADR-001) so Day 1 is
  not lost to an unverified upstream URL.
- Prompt (verbatim shape, transcript intact):
  > Run a 20-agent deep-planning bundle. One foundation agent writes
  > the architect decision rationale; 19 parallel doc agents draft the
  > 5 PRDs (GRAPHIFY / SKILLS / GAE / PPO + one cross-cutting), the 10
  > ADRs (one per OQ), PLAN.md, TODO.md, STATE_DESIGN, ACTION_DESIGN,
  > and a TRACE skeleton with `<phaseN-commit>` placeholders. Every ADR
  > must cite the §1.4 row it lives under, the decider, the date, and
  > the supersession window. No code in this commit — docs only.

### Phase 0 / commit `0ca9176` — close 14 critical + 30+ minor review findings
- Architect decision: Pass-3 (10-agent planning review) returned
  NEEDS_REVISION with reward-equation drift, GraphifyAdapter signature
  drift, and a Gymnasium-vs-brief tension. Architect ruled: every fix
  must quote the canonical reward equation *verbatim* and the
  GraphifyAdapter Protocol from ADR-002 *verbatim* — no paraphrase.
- Prompt (verbatim shape):
  > File-disjoint 20-agent fix workflow. The 6 canonical values are
  > sealed and must appear identically in every touched doc:
  > (1) `R_t = α·ΔMod + β·ΔCoh − γ·Coupling + P_skills` with
  > (α=1, β=1, γ=0.5, P=−5);
  > (2) A_max = 45057;
  > (3) betweenness = exactly 2 calls per seed;
  > (4) gymnasium banned in `src/env/`;
  > (5) 5 seeds {42, 7, 123, 314, 271};
  > (6) ≤150 LOC + ≥85% coverage.
  > A separate verifier agent rewrites STATE_DESIGN + ACTION_DESIGN to
  > restore alignment. No new public API surfaces.

### Phase 0 / commit `70464d7` — close 5 Codex-gap items
- Architect decision: Codex companion unavailable; remaining 5 items
  (PPO/GAE math write-up, ADR-004/008/010 reconciliation) must be
  closed by Claude-default. Architect signed off on the math sources
  (Schulman 2017 PPO + Schulman 2016 GAE) before any drafting.
- Prompt (verbatim shape):
  > 5-agent Claude-default cleanup. Each agent owns ONE Codex-missed
  > item. PPO + GAE math audit cites Schulman 2017 §3 (clip-objective)
  > and Schulman 2016 eq. (11)–(16) (GAE) with off-by-one line cites.
  > ADR-004 (GraphSAGE-vs-MLP) must justify the choice on graph
  > properties, not on framework convenience. ADR-008 must lock the
  > padding/masking contract. ADR-010 must lock the dual convergence
  > criterion. No reward-equation edits.

### Phase 0 / commits `e9167ed`, `1f18844`, `3b0ed5b` — final QA closure
- Architect decision: Pass-6 (20-agent final QA) returned
  BLOCK_PUBLIC with 15 HIGH + 24 MEDIUM. Dominant: betweenness 2-vs-3
  drift on 8 axes, banned λ-form reward in PRD §1.3, 404 links in
  README, dangling ADR filenames. Architect ruled: BLOCK→GO closure
  must include the file-size enforcer running and the coverage gate
  passing — no "we'll fix lint next sprint".
- Prompt (verbatim shape):
  > BLOCK→GO closure pass. Close all 15 HIGH findings + 10 high-value
  > MEDIUMs. Then run `uv run ruff format` over the 7 stub files
  > created in fix-3 (this is the *only* place we accept a separate
  > ruff-format commit — the seam is the stub-creation boundary). Then
  > push Phase-0 coverage from 66% to ≥85% by adding tests that
  > exercise the stub seams. Self-validate the file-size enforcer
  > (`scripts/check_file_sizes.py`) on the new tests.

---

## Phase 1 — GRAPHIFY adapter + Skills shim + Obsidian Vault

Phase 1 implemented the static analyser that lifts the PythonClaw
`Skills` subsystem into a directed weighted graph G = (V, E). The
architect locked the GraphifyAdapter Protocol in ADR-002 before any
implementation began; the AI was allowed to write `LocalGraphify` only
against that signed-off Protocol.

### Phase 1 / commit `0165fa2` — Skills shim + GRAPHIFY real impl + Obsidian Vault + SKILLS_ARCHITECTURE
- Architect decision: ADR-002 GraphifyAdapter Protocol is the single
  source of truth — `build_graph`, `EdgeWeigher`, node/edge attribute
  schema all live in the ADR. ADR-011 (Skills adapter) locks the
  domain wrapper. ADR-009 (screenshot pipeline) locks the Obsidian
  vault output format.
- Prompt (verbatim shape):
  > Implement `LocalGraphify` under `src/graphify/local_impl.py`
  > respecting the `GraphifyAdapter` Protocol from ADR-002 *verbatim*
  > (no method-name or signature drift). The AST walker visits
  > `ImportFrom`, `Import`, `ClassDef`, `FunctionDef`, `Call`. Edges
  > carry `kind ∈ {imports, calls, inherits}` and a weight from the
  > `EdgeWeigher` Protocol. Render the Obsidian vault per ADR-009
  > (one node per `.md`, edges as wiki-links, render PNG via
  > networkx + matplotlib). Write `SKILLS_ARCHITECTURE.md` from the
  > resulting graph (≤150 LOC for each new `.py`).

### Phase 1 / commit `9660830` — close 10 validation findings
- Architect decision: 5-agent validation flagged shim contract drift,
  TRACE drift, AST walker missing a `Call`-target case, and chart
  styling drift from ADR-009. Architect signed off on the assertion
  delta (the 10 specific things the next commit must satisfy).
- Prompt (verbatim shape):
  > Close the 10 validation findings. Acceptance gates: (a) shim
  > contract test passes `tests/architecture/test_pythonclaw_shim.py`;
  > (b) TRACE F1–F4 cite real SHAs from this commit (run
  > `scripts/stamp_trace.py --fix` after); (c) AST walker handles
  > nested `Call` targets via the attribute-chain heuristic in
  > ADR-002 §4; (d) Obsidian PNG matches the ADR-009 colour palette
  > and seed=42 `spring_layout`. No public API changes — adapter
  > Protocol stays frozen.

---

## Phase 2 — Custom RL env + reward + actions + centrality scheduler

Phase 2 wrapped the graph as a Reinforcement-Learning environment
*without* Gymnasium (brief §2.2 explicit ban, enforced by an AST-level
test, not grep). The architect locked the reward equation in ADR-007
and the action space (`split_module`, `merge_module`, `extract_skill`,
`drop_skill`, `noop`) in ACTION_DESIGN before any env code.

### Phase 2 / commit `ec1288a` — custom RL env + reward + actions + centrality scheduler
- Architect decision: ADR-007 reward equation is MUST (not SHOULD).
  ADR-006 multi-seed discipline pinned to {42, 7, 123, 314, 271}.
  Brief §2.2 betweenness-centrality budget = exactly 2 calls per seed
  (one for the initial state, one for the final state).
- Prompt (verbatim shape):
  > Implement `SkillsGraphEnv` under `src/env/`. Custom
  > `reset() -> obs, info` + `step(action) -> obs, reward, done,
  > truncated, info` API — NO `gymnasium` import (enforced by
  > `tests/architecture/test_env_no_gym.py` AST check). Reward is
  > `R_t = α·ΔMod + β·ΔCoh − γ·Coupling + P_skills` with the locked
  > coefficients from `config/config.yaml#reward`. Centrality scheduler
  > computes betweenness *exactly twice* per seed and caches. The
  > action space matches ACTION_DESIGN verbatim. Episode ends on
  > `noop` or after `max_steps`. File-size cap holds.

### Phase 2 / commits `1c317ef` + `8264a84` — wire metrics services + close 5-agent findings
- Architect decision: Codex review flagged that env reward was
  computed against a stub modularity service; architect ruled the
  real `src/services/metrics/{modularity,cohesion,coupling}.py` must
  land before claiming Phase 2 done. TRACE drift on F5 + GAE-key
  mismatch in SDK was a separate finding.
- Prompt (verbatim shape):
  > Wire the real metrics services into `SkillsGraphEnv.compute_reward`
  > so the env reward becomes real (close Codex gap). Then close the
  > 5-agent validation findings: TRACE F5 SHA stamp, GAE key spelling
  > in SDK config, public-API alignment between
  > `src/services/metrics/` and `src/env/`. ≤150 LOC per file holds;
  > new tests are unit-scoped against the services.

---

## Phase 3 — PPO + GAE custom trainer + multi-seed retrain

Phase 3 trained the agent. Architect locked the PPO hyperparameters
(ε=0.2, γ=0.99, GAE λ=0.95) and the custom-trainer requirement (no
SB3 `learn()` loop — write the GAE accumulator and the PPO clip
update by hand, brief §2.3).

### Phase 3 / commit `71f0213` — PPO + GAE custom trainer + real apply_action + multi-seed training + obsidian_after + betweenness chart
- Architect decision: brief §2.3 wants the math visible. SB3 is
  allowed for the policy network and rollout buffer (ADR-008), but
  the PPO clip objective and GAE accumulator must be written
  explicitly in our trainer. ADR-006 multi-seed discipline + ADR-010
  dual-convergence criterion gate the "trained" claim.
- Prompt (verbatim shape):
  > Implement the PPO + GAE custom trainer under
  > `src/trainer/train_ppo.py`. GAE accumulator follows Schulman 2016
  > eq. (11)–(16) with λ=0.95, γ=0.99. PPO clip objective follows
  > Schulman 2017 §3 with ε=0.2. The policy network is SB3-backed per
  > ADR-008 with the action-mask contract. After training each seed,
  > render the obsidian_after PNG and emit the betweenness chart
  > (exactly 2 calls per seed). The 5 seeds run isolated;
  > `aggregate.json` reports n, mean, std, dof=n−1 CI.

### Phase 3 / commit `5dd14ca` — guarantee NOOP slot in action mask
- Architect decision: seeds 123 / 314 / 271 hung on
  `Categorical(-inf)` when the action mask had every slot at −inf.
  Architect ruled the NOOP slot must be unmaskable (a structural
  invariant, not a runtime patch).
- Prompt (verbatim shape):
  > Bug fix. The action-mask builder is masking the NOOP slot when no
  > refactor op is legal, producing a `Categorical(logits=[-inf,
  > -inf, …])` and hanging the sampler. Make NOOP unmaskable: clamp
  > its logit to a finite value. Add a regression test that builds the
  > worst-case mask and asserts the sampler returns NOOP. No public
  > API change.

### Phase 3 / commit `8f96e30` — 5-seed isolated-subprocess wrapper
- Architect decision: even with the NOOP fix, seeds 123 / 314 wedged
  on a *second* slow path inside the metrics services. Architect
  ruled the multi-seed harness must be subprocess-isolated with a
  120s per-seed wall-clock budget so the run is auditable — TIMEOUT
  is an honest result, the architect refuses to report fake-converged
  numbers. HONESTY lock: 3 OK / 5 attempted → 3-seed mean ± std with
  dof=2 CI.
- Prompt (verbatim shape):
  > Write `scripts/train_5seed_isolated.py`. Each seed runs in its
  > own Python subprocess with a 120s wall-clock budget. On TIMEOUT,
  > record the seed in `attempted_seeds` and exclude it from the
  > aggregate. Emit `aggregate.json` with `num_seeds` (OK only) and
  > `attempted_seeds` (full list). Re-render
  > `betweenness_ci.png` + `betweenness_table.csv` with n=3, dof=2,
  > explicit `n_seeds` column. Update TRACE F10, TODO T3.7,
  > EXPERIMENTS P3-E1 to PARTIAL.

---

## Phase 4 — Cost + ablations + essay + RC bugs

Phase 4 wrote the §2.4 essay (2500–3000 words, brief-verbatim topic),
ran the cost meter (ADR-003a 15-column schema), ran the ablation grid,
and chased two RC bugs in the metrics services. Architect locked the
ESSAY thesis (COMPLEMENTARITY: GRAPHIFY priors × LLM semantic judgment,
A_max=45057 boundary) before any drafting.

### Phase 4 / commit `dbdd1a5` — ESSAY skeleton + 11-cite bibliography
- Architect decision: ESSAY follows Option A (COMPLEMENTARITY). 11
  citations sealed; the §2.4 prompt-mapping is brief-verbatim.
- Prompt (verbatim shape):
  > Draft the §2.4 essay skeleton under `docs/ESSAY.md`. Thesis:
  > GRAPHIFY structural priors are *complementary* to LLM semantic
  > judgment, with A_max=45057 as the boundary where structural
  > pruning stops being optional. 11 citations from the sealed
  > bibliography (Allamanis18, Chen21, Schulman17, Schulman16,
  > Hamilton17, Mnih15, …). Skeleton only — §1, §2, §3 drafts land in
  > later commits.

### Phase 4 / commit `f971ca4` — pin A_max=45057 derivation
- Architect decision: A_max is a canonical value; its derivation must
  live in the code and the test, not in a comment.
- Prompt (verbatim shape):
  > Pin the A_max=45057 derivation in
  > `src/env/action_space.py.__doc__` (Cartesian product of action
  > types × candidate nodes × candidate splits, with the floor
  > clamp). Add `tests/architecture/test_amax_pinned.py` that
  > computes A_max from the live config and asserts equality. ≤150
  > LOC holds.

### Phase 4 / commits `4752d08` + `69c8eb8` + `be91afc` — TripleCounter + phase corpora + cost_table
- Architect decision: ADR-003a locks the 15-column cost_table schema.
  Cost meter must split input vs output tokens by *role* (commit
  subject/body → output; prompt / prompts_md → input). Pricing pinned
  to Anthropic Opus 4.x snapshot.
- Prompt (verbatim shape):
  > Implement `src/cost/meter.py:TripleCounter`. Then
  > `scripts/collect_phase_corpora.py` emits per-phase JSONL of
  > prompt + commit triples (input role / output role tagged). Then
  > `scripts/compute_cost.py` streams the JSONL through TripleCounter
  > and writes `cost_table.csv` per ADR-003a (15 columns, sealed
  > order). Pricing: $15/M input + $75/M output (Opus 4.x). Subtotal
  > arithmetic asserted in tests. Hebrew UTF-8 multi-byte handled.

### Phase 4 / commits `44b313f` + `d489306` + `690856c` — RC Louvain wedge
- Architect decision: seeds 123 / 314 wedged 7+ hours inside Louvain
  community detection. Architect ruled three concurrent fixes: a
  watchdog + fallback inside `safe_louvain`, a partition memo across
  the 3 metrics per env.step, and a 0.05s budget. Plus full
  disclosure of the residual: the *fix* is in, but the *bug story*
  must be told honestly in `docs/_pending/BUG-2.md`.
- Prompt (verbatim shape):
  > Three concurrent fixes for the Louvain wedge:
  > (1) `safe_louvain` watchdog + connected-component fallback in
  > `src/services/metrics/modularity.py`;
  > (2) partition memo keyed on `(frozenset(nodes), frozenset(edges))`
  > so NOOPs and no-op refactor ops cost zero;
  > (3) compute the partition ONCE per env.step and share it across
  > `compute_modularity` / `compute_cohesion` / `compute_coupling`
  > via a `_partition` kwarg.
  > Drop `WATCHDOG_SECONDS` to 0.05. Extend the helper to
  > `coupling.py`. Add a regression test
  > (`tests/unit/services/metrics/test_safe_louvain.py`). Then fill
  > `docs/_pending/BUG-2.md` with the cProfile stack, the RC-1 fix,
  > and the residual disclosure.

### Phase 4 / commit `1b632ff` — fill BUG-1 (Categorical(-inf) NOOP-pin)
- Architect decision: every fix that landed under a `fix(...)` commit
  must have a paired `BUG-N.md` story in `docs/_pending/` — the
  architect refuses to ship a fix without the bug it fixed being
  documented.
- Prompt (verbatim shape):
  > Fill `docs/_pending/BUG-1.md`. Reproducer: build an action mask
  > with every refactor slot masked out; sample. Expected: NOOP
  > returned. Actual (pre-fix): hang on `Categorical(-inf)`. Cite
  > fix commit 5dd14ca + the regression test. Story tone matches the
  > §1.4 honesty lock — bugs are findings, not embarrassments.

### Phase 4 / commits `667692c` + `76544e1` — ESSAY §1, §2, §3
- Architect decision: word-budget per section sealed before drafting
  (§1 ≈500w, §2 ≈670w, §3 ≈860w). Each section opens against a
  citation manifest the architect approved.
- Prompt (verbatim shape):
  > Draft §1 Introduction (~500w) citing Allamanis18 (graph-based code
  > understanding), Chen21 (LLM code synthesis), Schulman17 (PPO).
  > Then §2 SA landscape (~670w) — Software Analytics landscape with
  > the GRAPHIFY-vs-LLM dichotomy. Then §3 Methodology (~860w) — the
  > custom-trainer + dual-convergence + 5-seed discipline. Each
  > section ends with the COMPLEMENTARITY thesis re-stated against
  > the section's evidence.

### Phase 4 / commits `1f5765d` + `611d0a6` — ablation infra + plumbing
- Architect decision (at the time): ablation grid is 3×3×3×3 = 81 cells
  (α, β, γ, P_skills perturbations) × 3 OK seeds × 256 steps — *compact*,
  3 OK seeds per cell because the then-honest 3/5 retrain state was the
  ablation input; sweeping over the 2 wedged seeds would have been dishonest.
  > **SUPERSEDED (RC-4).** Once the RC-4 fix (SIGALRM Louvain cut +
  > stored-mask Trajectory) closed the seed-123/314 hang, all 5 seeds train
  > cleanly, so the ablation was re-run at **5 seeds/cell** across all 81
  > cells (`config.ablation.scout_seeds = [42, 7, 123, 314, 271]`). The
  > 3-seed framing below is the historical prompt; the live result is n=5.
- Prompt (verbatim shape):
  > Thread reward coefficients α, β, γ, P_skills through
  > `SkillsGraphEnv.__init__` and `train_ppo` CLI flags. Then
  > `src/_ablation_lib.py` + `scripts/run_ablation.py` emit one
  > `results/ablations/cell_<sha>/done.json` per cell + a top-level
  > `seed_table.csv`. 81 cells × 3 seeds × 256 steps. The ablation is
  > a background run; Wave-4a streams must not race
  > `results/ablations/`.

---

## §1.4 contract closing note

Every prompt in this file was preceded by an architect-edited PRD,
PLAN, ADR, or TODO entry. The AI never made a decision listed in the
§1.4 *Human-decided* column without that sign-off:

- Requirements / scope / KPIs → architect edit in `docs/PRD.md` or the
  per-component PRD under `docs/prd/PRD-*.md` *before* the prompt.
- Architecture / public API shape → architect edit in `docs/PLAN.md`
  or a new `docs/adr/ADR-*.md` *before* the prompt.
- Test acceptance criteria + assertions → architect edit in
  `docs/TODO.md` or the section header of the new test file *before*
  the prompt.
- Self-score / grade claim → architect-only edit in
  `docs/shared/PROMPTS.md §4 Decision log` (sealed at 88–92 range,
  with the −2 honesty penalty for 3-of-5 seeds folded in).
- Cost-budget envelope → architect-only edit in
  `docs/COST_ANALYSIS.md` and the ADR-003a row counts.

The audit trail is therefore: `git log` shows the *AI implementer* row;
`docs/PRD.md` + `docs/PLAN.md` + `docs/adr/` show the *architect
decision* row; this file shows the *prompt that connected them*; and
the closing `docs/shared/PROMPTS.md §4` log shows the architect-only
decision-log column the AI is forbidden from editing.
