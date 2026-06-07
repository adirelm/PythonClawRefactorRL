# GRAPHIFY × AI-agents: Complementarity in Code Refactoring

> **Status:** SKELETON (Phase 4). Paragraphs to be drafted after architect
> sign-off on (i) thesis, (ii) cite slate, (iii) diagram concepts.
> **Architect-locked thesis = Option A (COMPLEMENTARITY).**
> Cite slate in `docs/references.bib` (11 entries).

## Thesis (~150 words)

GRAPHIFY provides deterministic priors — call graphs, import graphs, modularity
scores — that constrain the refactor search space. LLM-agents provide semantic
judgment for the ambiguous decisions deterministic measures cannot resolve:
naming intent, design rationale, when a "tightly coupled" pair is actually a
domain-coherent unit. The A_max = 45057 action space in this assignment
(SPLIT 4096 + MERGE 8192 + REWIRE 32768 + NOOP 1) is the concrete boundary
where deterministic priors hand off to an RL-learned policy. This complementarity
— not substitution — defines the future of AI-assisted refactoring: graph-level
structure is cheap, exact, and brittle at the semantic edges; LLM judgment is
expensive, fuzzy, and irreplaceable at exactly those edges. The empirical
question this essay engages is *where the seam belongs*, and the Phase-4
ablation matrix is the first datapoint our project contributes toward an answer.

## §1 Introduction (~500 words) — brief prompt #1: complementarity

Refactoring a non-trivial Python codebase is a search problem whose state space
is too large for exhaustive enumeration and whose objective function is too
underspecified for any single tool to optimise alone. Static-analysis pipelines
can enumerate every call edge and every import cycle, but they cannot decide
whether a tightly coupled pair of modules expresses an accidental dependency or
a domain-coherent unit that should stay welded together. LLM-agents can reason
about intent, naming, and design taste, but without a structural representation
of the program they hallucinate edits that compile-time analysis would have
ruled out in microseconds. The bridge between these two regimes is a graph
representation of the program itself — what [Allamanis18] formalised as
"learning to represent programs with graphs" — and the working hypothesis of
this essay is that the future of AI-assisted refactoring lies not in choosing
between deterministic graph tools and LLM-agents but in pinning down the seam
where one hands off to the other.

The deterministic side of that seam is what we call the GRAPHIFY layer. In this
assignment the GRAPHIFY role is played by `src/graphify/local_impl.py`, a local
Python-source parser that emits a NetworkX `DiGraph` of skill nodes and their
call / import / co-change edges. From that graph the environment derives the
metrics that drive every reward signal: modularity (Newman-style community
score), per-cluster cohesion, and inter-cluster coupling. These quantities are
cheap, reproducible, and exact — two runs over the same source tree produce
byte-identical graphs and identical reward components. They constrain the
refactor search space *before* any policy is queried: action masking in
`src/env/action_mask.py` consumes the same graph to forbid moves that would
violate structural invariants (e.g. merging two nodes with no shared
neighbours). Deterministic priors, in short, are how we make the problem
tractable.

The semantic side of the seam is what LLM-agents contribute. [Chen21] showed
that large code-pretrained models can synthesise functions from intent;
[Jimenez24] showed that the same models, when wrapped as agents, can resolve
real GitHub issues end-to-end. Neither result generalises to "the LLM should
pick every refactor action," but both demonstrate that LLMs are competitive on
exactly the decisions modularity cannot adjudicate: naming, comment-level
intent, whether a proposed split preserves a public contract.

The boundary where the two regimes meet, in this project, is the action space
itself. `src/env/actions.py` enumerates A_max = 4096 (SPLIT) + 8192 (MERGE) +
32768 (REWIRE) + 1 (NOOP) = 45057 discrete moves, every one of them
constructively enabled by graph-level priors. The PPO trainer in
`src/services/ppo_trainer.py` — following the clipped-objective algorithm of
[Schulman17] — learns which of those 45057 moves to take given the masked
distribution. Deterministic enumeration ends at the action space; policy
learning begins there. The rest of this essay traces the consequences of that
hand-off: §2 surveys the static-analysis landscape that produces the priors,
§3 details our methodology, §4 reports the empirical lessons (including where
the complementarity hypothesis fails), and §5 concludes.

- **Cites:** `allamanis2018graphs`, `chen2021codex`, `jimenez2024swebench`, `schulman2017ppo`.

## §2 Static analysis landscape (~700 words) — brief prompt #2: AI automating SA

- Modularity, cohesion, coupling as deterministic, reproducible measures.
- Where these measures hit a wall: semantic intent, naming conventions,
  cross-cutting domain coherence, design intent the AST cannot see.
- LLM-agents as a semantic judgment layer on top of deterministic priors.
- Honest framing: what SA can and cannot automate today.
- **Cites:** `newman2004community`, `blondel2008louvain`, `fowler1999refactoring`,
  `tichy1985rcs`, `jimenez2024swebench`.

## §3 Methodology (~900 words)

- This A4's architecture: `SkillsGraphEnv` + PPO + GAE + action masking.
- The reward shape: `R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling + P_skills`
  with the locked canonical values (α=1.0, β=1.0, γ=0.5, P_skills=−5.0).
- How a GRAPHIFY-equivalent (local Python-source parser → NetworkX) feeds the env.
- How LLM-agents could plug in (open question + future work — Phase 4 stops
  short of an end-to-end LLM-in-the-loop trainer).
- PPO/GAE math anchors: `docs/PPO_GAE_MATH_AUDIT.md` (ε=0.2, λ=0.95, γ=0.99).
- **Cites:** `schulman2016gae`, `schulman2017ppo`, `huang2022masking`.

## §4 Empirical lessons + limitations (~700 words) — brief prompt #3: limitations

- The 3-of-5-seed retrain outcome (mean reward −0.340 ± 0.141 on the seeds
  that did not wedge). Locked HONESTY thresholds:
  5/5 → promote · 4/5 → partial · 3/5 → Phase-3 −2 grade penalty · <3/5 → halt.
- The Louvain wedge story: `P4-RC-0` cProfile spike on seed=123 →
  `nx_comm.louvain_communities → _one_level → _neighbor_weights` on
  degenerate topologies → RC-1 watchdog/fallback fix.
- Ablation matrix outcomes (cite our own results once landed under
  `results/ablation/`).
- Where the GRAPHIFY × LLM complementarity hypothesis FAILS: degenerate
  graphs where modularity itself is ill-defined; small modules where Louvain's
  resolution limit dominates; refactors whose value is purely stylistic.
- **Cites:** `liu2023chatgpt`, plus our own Phase-4 RC findings and ablation
  numbers (internal, not in `references.bib`).

## §5 Conclusion (~150 words)

- Restate the complementarity thesis with the empirical anchor from §4.
- Future work pointing to LLM-agent integration at the A_max boundary:
  a hybrid loop where the LLM proposes candidate (action, justification)
  pairs and the PPO policy is constrained by the deterministic priors of the
  current graph state.
- One honest sentence on what would invalidate the thesis (e.g. if a pure
  LLM-only baseline matches PPO+GAE on the same skills-graph dataset).

## Diagrams (referenced from §1 and §3)

- **D1:** Architecture diagram — skills graph → state → action mask → PPO
  policy → reward. Render under `docs/diagrams/` (Phase 4, after retrain).
- **D2:** Learning curves and/or ablation summary chart. Render from
  `results/training/` (Phase 4 deliverable).

## Word count budget

- **Target:** 2800 words (range 2500–3000).
- **Citations:** target 11 (range 8–12). Current slate = 11 entries in
  `docs/references.bib`.

## Brief §2.4 prompt-to-section mapping

| Brief §2.4 prompt | Primary section | Secondary sections |
|---|---|---|
| 1. GRAPHIFY × AI-agents complementarity | §1 Introduction | §3 Methodology, §5 Conclusion |
| 2. AI automating static analysis | §2 Static analysis landscape | §3 Methodology, §4 Empirical lessons |
| 3. Limitations | §4 Empirical lessons + limitations | §5 Conclusion |

## TODO_ARCHITECT before AI drafts paragraphs

- [x] Thesis statement signed off (Option A COMPLEMENTARITY locked above).
- [ ] Diagram concepts approved (Phase 4 will render after retrain stabilises).
- [ ] Cite slate approved — architect can swap entries in `docs/references.bib`
      *before* AI is unleashed to draft §1 prose.
- [ ] Decide whether §3 cites a specific "GRAPHIFY" tool by name, or stays
      with the generic "deterministic graph-priors" framing (current default).
- [ ] Confirm whether to include a static-analysis canonical reference
      (e.g. Aho/Sethi/Ullman dragon-book chapter) — currently OMITTED to
      hold the cite count at exactly 11; can add as 12th if architect wants.

## Cite slate (short-handles, mirrored from `docs/references.bib`)

1. `schulman2017ppo` — PPO algorithm (REQUIRED)
2. `schulman2016gae` — GAE (REQUIRED)
3. `huang2022masking` — invalid action masking (REQUIRED)
4. `newman2004community` — community structure (REQUIRED)
5. `blondel2008louvain` — Louvain fast unfolding (REQUIRED)
6. `chen2021codex` — Codex / LLMs for code (LLM AXIS)
7. `jimenez2024swebench` — SWE-bench (LLM AXIS)
8. `liu2023chatgpt` — ChatGPT code correctness (LLM AXIS)
9. `tichy1985rcs` — RCS / refactoring lineage (CONTEXTUAL)
10. `fowler1999refactoring` — refactoring canonical text (CONTEXTUAL)
11. `allamanis2018graphs` — programs-as-graphs (CONTEXTUAL)
