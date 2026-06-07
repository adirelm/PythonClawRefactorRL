# Bug Report

> **Brief §3 deliverable.** Two **architectural bugs in the PythonClaw `Skills`
> module** that the GRAPHIFY reverse-engineering + RL refactoring engagement
> exposed and isolated. Both are *structural* smells (not behavioural crashes —
> see PRD §6.2 L4): they are properties of the skill dependency graph that the
> agent's modularity / coupling / betweenness analysis surfaces directly.
>
> An **appendix** documents two engineering defects found and fixed in the RL
> training pipeline itself during the build — these are not Skills-module bugs,
> but they are recorded for rigor and because they shaped the experiment trail.

## Method (how the bugs were exposed)

1. `src/graphify/local_impl.py` parses the 10-skill `sample_skills` corpus into a
   `networkx.DiGraph` whose edges are the `depends_on` relations declared in each
   skill's `*.metadata.json` (L1).
2. The environment derives **modularity** (Newman-Girvan Q), per-node **fan-in /
   fan-out coupling**, and **betweenness centrality** (brief §2.2, computed twice
   per seed across 5 seeds — `results/data/betweenness_table.csv`).
3. The two bugs below fall straight out of those structural measures — no
   behavioural test of PythonClaw was needed, consistent with the brief's
   "structural bug discovery via reverse engineering" framing.

---

## Bug 1 (architectural): Orphan skills — `json_validator` and `web_search` are disconnected from the dependency graph

- **Severity**: MEDIUM — dead structural weight; depresses module modularity and
  is unreachable by any dependency-driven refactor.

- **Finding**: In the reverse-engineered dependency graph, `json_validator` and
  `web_search` both have **in-degree 0 AND out-degree 0** — they neither depend on
  any skill nor are depended upon by any skill. Every other skill participates in
  at least one `depends_on` edge.

- **Evidence**:
  - Dependency-graph computation over `src/pythonclaw_shim/sample_skills/*.metadata.json`:
    edge set is 10 edges across 8 skills; `json_validator` and `web_search` appear
    in zero edges.
  - `results/data/betweenness_table.csv`: `skill.web_search.L{1,2,3}` and
    `skill.json_validator.L1` all report `mean_before = 0.0` betweenness across all
    5 seeds — they lie on **no** dependency path.
  - **Documentation contradiction**: `sample_skills/README.md` lists
    `web_search` and `json_validator` under **"Roots"**. A root is a node with
    *outgoing* edges to its dependencies; these two have none. They are **isolated
    nodes**, not roots — the README mischaracterises them, which is itself the
    architectural-intent bug (a skill declared but never wired in).

- **Impact**:
  - Modularity (Q): each orphan is forced into its own singleton community,
    lowering the achievable community structure of the module.
  - RL refactor reachability: the `MERGE` action requires a top-M cosine-similar
    *neighbour* (`src/env/action_mask.py::_topm_similar`); with no edges, the
    orphans are never productive MERGE partners, so the agent cannot fold them in.
  - Maintenance: an orphan skill is either dead code (should be removed) or
    missing wiring (a consumer edge was never declared) — both are latent defects.

- **Recommended refactor**: either (a) remove the orphans if genuinely unused, or
  (b) declare the missing `depends_on` edge that a real consumer needs (e.g. a
  validation step in `code_review` plausibly should `depends_on: [json_validator]`).

---

## Bug 2 (architectural): Coupling hotspot — `python_execution` is a single point of structural fragility

- **Severity**: HIGH — highest afferent coupling and the only non-zero-betweenness
  node; a change here ripples to four dependents.

- **Finding**: `python_execution` has **fan-in 4** — it is depended upon by
  `code_review`, `diagram_creator`, `refactoring_planner`, and `test_generator`
  (the next-highest fan-in is `file_search` at 3). It is also the **only** node
  with non-zero betweenness centrality in the graph.

- **Evidence**:
  - Fan-in tally over the metadata graph: `python_execution = 4`, `file_search = 3`,
    `code_review = 2`, `documentation_writer = 1`, all others 0.
  - `results/data/betweenness_table.csv`: `skill.python_execution.L1` is the sole
    node with `mean_before > 0` (≈ 0.00246 before refactor, rising to ≈ 0.00286
    after) — it lies on the most dependency paths of any node, across all 5 seeds.

- **Why it is a bug (Stable Dependencies Principle, R. Martin)**: a node that four
  others depend on should be **maximally stable** (instability I = fan-out /
  (fan-in + fan-out) → 0). `python_execution` instead carries an outgoing edge
  (`→ file_search`), giving it I = 1 / (4 + 1) = **0.2** — non-zero. A change in
  `file_search` therefore propagates *through* `python_execution` to all four of
  its dependents. The most-depended-upon skill is not the most stable, which is
  the SDP violation.

- **Impact**:
  - The agent's betweenness analysis (brief §2.2) flags `python_execution` as the
    highest-value refactor target; a SPLIT that separates its file-search-dependent
    behaviour from its dependency-free core would raise module stability.
  - Until then it is the module's fragility hub: the blast radius of any change is
    five skills.

- **Recommended refactor**: SPLIT `python_execution` into a dependency-free core
  (depended upon by the four consumers) and a thin `file_search`-coupled adapter,
  so the hub becomes maximally stable (I = 0).

---

## Appendix — Engineering defects found & fixed in the RL training pipeline

> These are **not** Skills-module architectural bugs. They are defects in *our own*
> training harness, surfaced and fixed during the build. Recorded for rigor and
> because they gate the 5-seed result reported in EXPERIMENTS P3-E1.

### A1 — `Categorical(logits=all_-inf)` NaN-explosion on an all-False action mask

- **Symptom**: pre-fix (`71f0213`), `PolicyNet.get_action` fed an all-`-inf` row
  into `torch.distributions.Categorical` on degenerate sub-graphs (every SPLIT /
  MERGE / REWIRE illegal), producing NaN probabilities and a hanging `.sample()`.
- **Root cause**: the NOOP slot was not pinned True before `masked_fill` when the
  mask was otherwise all-False.
- **Fix** (`5dd14ca`, `src/model/policy_net.py:113-118`): force the NOOP slot True
  on any all-False row before masking — preserving the Huang & Ontañón (2022)
  invalid-action-masking guarantee.
- **Regression test**: `tests/architecture/test_policy_net_categorical_safe.py` (4 cases, green).

### A2 — Louvain community detection wedged on degenerate mid-rollout topologies

- **Symptom**: pre-fix, `compute_modularity → networkx.louvain_communities` blocked
  for 10-20 s (originally hours) on specific mid-rollout graph snapshots produced by
  the agent's SPLIT/MERGE actions; seeds 123 and 314 timed out.
- **Root-cause evolution**:
  - RC-1 (`44b313f`/`d489306`): per-snapshot partition sharing (6 → 2 Louvain calls
    per `env.step`) + structural-key cache + a daemon-thread watchdog. This reduced
    but did not eliminate the wedge — **orphaned daemon threads accumulated and
    contended for the GIL**, so seeds 123/314 still timed out at 3/5.
  - **RC-4 (current fix)**: replaced the daemon-thread watchdog with a `signal.SIGALRM`
    1-second hard cut that raises in the *calling* thread (no threads, no GIL
    accumulation), plus stored action masks in `Trajectory` to eliminate ~1024
    `compute_mask` recomputations per PPO update.
- **Outcome (RESOLVED)**: **all 5 seeds {42, 7, 123, 314, 271} now complete** within
  the per-seed budget (~10 s each). Mean final reward −0.461 ± 0.186 (n=5).
  `results/training/aggregate.json` reports `num_seeds=5`. The −2 honesty penalty
  pre-committed for a 3/5 outcome (PRD §7) is therefore **lifted** per the project's
  own rule (`5/5 → done`).
- **Regression tests**: `tests/architecture/test_modularity_wedge_regression.py`,
  `tests/unit/services/metrics/test_modularity_watchdog.py`.

---

## Cross-references

- Dependency graph source: `src/pythonclaw_shim/sample_skills/*.metadata.json`
- Betweenness evidence: `results/data/betweenness_table.csv` (n=5), `results/figures/betweenness_ci.png`
- Skills architecture deep-dive: `docs/SKILLS_ARCHITECTURE.md`
- Reward / instrumentation ADR: `docs/adr/ADR-007-reward-upgrade-MUST.md`
- Fix commits: A1 `5dd14ca`; A2 RC-1 `44b313f`/`d489306`, RC-4 (current branch)
