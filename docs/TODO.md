# TODO — phased task list (A4: PythonClaw Refactor + RL on Code Graph)

Task tracker for Assignment 4. See `docs/PRD.md` for scope and `docs/PLAN.md`
for architecture context. Every row below states **who** owns it (always
the solo developer in the architect role — see §1), **what** the DoD is
(§2), and **which phase + acceptance criterion** it lives under (§3).

## §1 Ownership statement (CLAUDE.md §1.4)

Solo project. Every task below is owned end-to-end by the solo developer
in the **architect** role (scope, state/action design, reward shaping,
acceptance criteria, theory cross-references, sign-off). The AI acts as
**implementer** against an approved PRD/PLAN edit — never as
decision-maker. No per-task hand-off — ownership is stated once here
rather than per row.

The architect-decided commitments that frame the whole tracker:

- **ADR-001 (PythonClaw shim).** A4 ships a thin local shim today and
  swaps to the upstream PythonClaw API within 24h of release. The shim
  contract — public method names, return types, tokenisation — is
  human-decided. Replacement is AI-delegated.
- **ADR-002 (GRAPHIFY adapter).** GRAPHIFY runs as a local
  re-implementation behind `GraphifyAdapter`. The adapter interface is
  human-decided; the in-tree implementation is AI-delegated.
- **Token policy.** Headline numbers use `tiktoken cl100k_base`. The
  appendix reports chars + bytes for comparability. This is a
  human-decided reporting convention.
- **Betweenness centrality protocol.** Computed once per seed across
  ≥5 seeds; report mean ± std + 95% CI. Single-shot numbers are
  forbidden.
- **Convergence criterion (dual).** Rolling-100 episode reward stable
  within ±2% **AND** policy entropy below the configured floor for ≥50
  consecutive episodes. Either alone is insufficient.
- **Ablation matrix.** α (ΔModularity weight, default 1.0), β
  (ΔCohesion weight, default 1.0), γ (Coupling-penalty weight, default
  0.5), and `P_skills` (lazy-load-break penalty, default −5.0) are
  MUST-ablate. Full grid × ≥5 seeds/cell. Canonical reward equation
  (brief §2.2): `R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t`.
- **§2.4 essay.** 2500–3000 words, 4 sections, 8–12 citations, 2
  diagrams. Architect writes the outline + thesis; AI drafts paragraphs
  the architect then edits.

## §2 Definition of Done (DoD)

Every build task is **done** only when **all five** hold:

1. **Behaviour implemented** in `src/` matching the PRD requirement id
   (column `req`). The function/class signature lines up with whatever
   the PRD or ADR named — renames are architect-level changes and need
   a PRD edit first.
2. **Test asserting the behaviour** in `tests/` — the test must fail
   without the implementation and pass with it (TDD red→green→refactor).
   A single happy-path test is not enough when the behaviour has a
   conditional branch; cover both sides.
3. **Quality gates green** locally and in CI:
   - `uv run ruff check src/ tests/ main.py` → zero violations
   - every `.py` ≤150 LOC (enforced by the pre-commit hook from T00-02)
   - `uv run pytest --cov=src` ≥85% statement coverage (enforced once
     Phase 1 code lands; Phase 0 is docs+scaffold so coverage is N/A)
   - no PII matches against the deny-list (see CLAUDE.md)
4. **Evidence pointer** recorded in the commit message body — file path
   / test id / chart path / notebook cell / screenshot path / commit
   hash. Bare "done" is not acceptable. The evidence pointer is what
   `docs/TRACE.md` (added in Phase 4) will index.
5. **Commit lands** under the matching phase, subject line matching
   regex `^(Phase \d+|Phase 0 bootstrap|chore: bootstrap)` and trailer
   `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.

**Honesty stance.** This TODO contains **no numeric self-grade
prediction**. The self-grade declared on the cover sheet
(`adrl-001-ex04.pdf`) is the only place a numeric self-score lives.
Internal docs describe what was built and its honest limitations, not
what grade it will earn. Per-task evidence pointers describe what runs
and how it was measured — never "looks good" or "works well".

**Failure modes that do not pass DoD** (record these honestly in TRACE
rather than masking them):

- A test that passes only because the assertion was weakened mid-task.
- A coverage number ≥85% achieved by deleting hard-to-test branches.
- A chart whose caption omits seed, sample size, or mean ± std.
- An ablation cell with <5 seeds or no confidence interval.
- A convergence claim resting on only one of the two dual criteria.

## §3 Build phases — status + evidence-pointer template

| # | Phase | Status | Evidence pointer template |
|---|---|---|---|
| **0** | **Bootstrap** (repo skeleton, pyproject, uv, ruff, pytest, CI, ≤150-LOC guardrail, docs/ scaffold, ADR-001/002 stubs) | ✅ | `pyproject.toml`, `.github/workflows/ci.yml`, `docs/PRD.md`, `docs/PLAN.md`, `docs/adr/ADR-001-pythonclaw-shim-boundary.md`, `docs/adr/ADR-002-graphify-adapter.md`, commit Phase 0 bootstrap |
| 1 | §2.1 GRAPHIFY + Obsidian (graph builder, adapter, NetworkX + pyvis screenshots, Obsidian hero shots) | 🟡 in-progress (T1.1–T1.6 ✅, others pending) | `src/graphify/`, `src/graphify/adapter.py`, `results/graphify_output.gpickle`, `results/vault/`, `results/figures/` (Phase 2+: dedicated `results/graphs/` + `results/obsidian/` dirs pending), commit Phase 1 |
| 2 | §2.2 Environment (state/action design, refactor env, reward = α·ΔModularity + β·ΔCohesion − γ·Coupling_Penalty + P_skills, masking) | ⬜ | `src/env/refactor_env.py`, `src/env/action_mask.py`, `docs/STATE_DESIGN.md`, `docs/ACTION_DESIGN.md`, commit Phase 2 |
| 3 | §2.3 PPO + GAE (policy net, GAE advantage, clipped surrogate, SB3 spike + fallback) | ⬜ | `src/training/ppo_trainer.py`, `src/training/gae.py`, `src/policy/policy_net.py`, `docs/adr/ADR-008-sb3-variable-v-buffer.md`, commit Phase 3 |
| 4 | §2.4 Cost + ablations + essay (tiktoken cost panel, α/β/γ/P_skills ablation matrix ≥5 seeds, ΔReward summary, 2500–3000 word essay) | ⬜ | `docs/COST_ANALYSIS.md` (D8), `results/ablations/`, `docs/ANALYSIS.md` (D7 ΔReward), `docs/ESSAY.md`, `notebooks/analysis.ipynb`, commit Phase 4 |

Phase gates (all must be green before a phase is marked ✅):
ruff zero violations · every `.py` ≤150 LOC · coverage ≥85% · uv-only ·
SDK is single business-logic entry · notebook is a *consumer* of the
SDK (no parallel implementation) · betweenness reported with mean ± std
+ 95% CI across ≥5 seeds · convergence asserted via dual criterion ·
no PII matches in tree.

## §3.0 Phase 0 — Bootstrap (✅ landed)

Status: ✅ committed by the foundation agent. Listed for traceability.

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T00-01 | Initialise repo skeleton (`pyproject.toml`, `uv.lock`, ruff config, pytest config, `.gitignore`, `.env-example`, CI workflow) | Initialising repo skeleton | 0 | +200 | TR-bootstrap | `uv run pytest` and `uv run ruff check` both exit 0 on empty suite |
| T00-02 | Add `tool.coverage` `fail_under=85`, ruff line-length, and ≤150-LOC pre-commit hook script | Adding coverage gate and 150-LOC pre-commit hook | 0 | +40 | (gate) | Hook rejects a deliberately bloated test file in CI dry-run |
| T00-03 | Scaffold `docs/` (PRD, PLAN, TODO, README, adr/, shared/, prd/, diagrams/, assets/) | Scaffolding docs/ tree | 0 | +0 | (docs) | All 6 top-level docs files exist; `adr/` contains ADR-001 + ADR-002 stubs |
| T00-04 | Draft `docs/adr/ADR-001-pythonclaw-shim-boundary.md` (24h swap window, public-method contract) | Drafting ADR-001 (PythonClaw shim) | 0 | +120 docs | TR-shim | ADR has Context / Decision / Consequences / Swap-plan sections |
| T00-05 | Draft `docs/adr/ADR-002-graphify-adapter.md` (in-tree re-impl + adapter interface) | Drafting ADR-002 (GRAPHIFY adapter) | 0 | +120 docs | TR-graphify | ADR names adapter interface methods and rollback path |
| T00-06 | CLAUDE.md inheritance from A1 (≤150 LOC, ruff, pytest ≥85%, uv-only, commit-subject regex, PII deny-list) | Importing CLAUDE.md inheritance | 0 | +0 | (gates) | `grep -E "150\|ruff\|85%\|uv\|Phase " CLAUDE.md` returns ≥6 matches |

## §3.1 Phase 1 — §2.1 GRAPHIFY + Obsidian (🟡 in-progress; T1.1–T1.6 landed)

Goal: build the code graph (nodes = files/modules/symbols, edges =
imports/calls/inheritance), wrap it behind `GraphifyAdapter`, and produce
the screenshot deliverables (programmatic NetworkX + pyvis charts and
Obsidian hero shots).

### §3.1.0 Phase 1 workflow-level tasks (T1.1–T1.6 ✅ landed)

The Phase 1 build workflow tracks these six top-level deliverables, all of
which landed in this workflow run. The fine-grained T01-NN rows below
remain the per-acceptance-test surface; T1.x rows are the workflow-level
bundle that gates Phase 1 closure.

| id | content | status | evidence pointer |
|----|---------|--------|------------------|
| T1.1 | Vendor PythonClaw shim (+ adapter Protocol per ADR-001) | ✅ | `src/pythonclaw_shim/skill.py`, `src/pythonclaw_shim/registry.py` (Phase 2+: `src/adapters/pythonclaw_adapter.py` facade pending), commit `0165fa2` |
| T1.2 | Run GRAPHIFY on Skills/ (local_impl real, not stub) | ✅ | `src/graphify/local_impl.py`, `results/graphify_output.gpickle` (Phase 2+: `results/graphs/skills_graph.json` JSON export pending), commit `0165fa2` |
| T1.3 | Build Obsidian Vault (vault writer from GRAPHIFY output) | ✅ | `scripts/build_vault.py`, `results/vault/` (Phase 2+: dedicated `src/obsidian/vault_writer.py` module + `results/obsidian/vault/` relocation pending), commit `0165fa2` |
| T1.4 | Capture obsidian_before.png (pre-refactor hero shot) | ✅ | `scripts/capture_obsidian_stub.py`, `results/figures/obsidian_before.png` (Phase 2+: `results/obsidian/obsidian_before.png` relocation pending), commit `0165fa2` |
| T1.5 | Author `docs/SKILLS_ARCHITECTURE.md` (F15) | ✅ | `docs/SKILLS_ARCHITECTURE.md`, commit `0165fa2` |
| T1.6 | Run lazy-load invariant tests (ADR-005 semantics) | ✅ | `tests/architecture/test_lazy_load_broken.py`, commit `0165fa2` |


| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T01-01 | Implement `src/graphify/parser.py` — Python AST → node/edge stream (imports, calls, class inheritance) | Implementing AST parser | 1 | +140 | §2.1-parse | `test_parser_extracts_imports_calls_inheritance` |
| T01-02 | Implement `src/graphify/graph_builder.py` — assemble `networkx.DiGraph` from parser stream, attach node attrs (loc, complexity, kind) | Implementing graph builder | 1 | +120 | §2.1-build | `test_graph_builder_emits_directed_graph_with_attrs` |
| T01-03 | Implement `src/adapters/graphify_adapter.py` per ADR-002 — public methods `build_graph(path)`, `as_networkx()`, `as_pyvis_html()` | Implementing GraphifyAdapter facade | 1 | +90 | §2.1-adapter,ADR-002 | `test_graphify_adapter_methods_present` + `test_adapter_returns_networkx_digraph` |
| T01-04 | Implement `src/graphify/betweenness.py` — wrap `networkx.betweenness_centrality` with seeded sampling; emit per-seed CSV | Implementing seeded betweenness | 1 | +90 | §2.1-betweenness | `test_betweenness_seeded_reproducible` + CSV exists for ≥5 seeds |
| T01-05 | Authoring `scripts/render_networkx.py` — render graph PNG via NetworkX/matplotlib (deterministic layout seed) | Authoring NetworkX renderer | 1 | +100 | §2.1-render | `results/graphs/networkx_seed{0..4}.png` exists |
| T01-06 | Authoring `scripts/render_pyvis.py` — render interactive HTML via pyvis | Authoring pyvis renderer | 1 | +90 | §2.1-render | `results/graphs/pyvis.html` opens and shows graph |
| T01-07 | Capture Obsidian hero shots (3 screenshots: vault overview, focused subgraph, refactor-target node) | Capturing Obsidian hero shots | 1 | +0 | §2.1-obsidian | `results/obsidian/hero_{1,2,3}.png` exist; each ≥1280×720 |
| T01-08 | Report betweenness mean ± std + 95% CI across 5 seeds → `results/betweenness_summary.csv` + notebook cell | Reporting betweenness stats (5 seeds) | 1 | +60 | §2.1-stats | `test_betweenness_summary_has_mean_std_ci` |
| T01-09 | Author `docs/adr/ADR-001-pythonclaw-shim-boundary.md` swap test — `test_shim_swap_smoke` that exercises the public surface | Authoring PythonClaw shim swap test | 1 | +60 test | ADR-001 | Test passes against shim today; swap target documented |
| T01-10 | Add `src/graphify/skills.py` — extract per-file skill vector (test-coverage hint, type-hint density, docstring presence, complexity bin) used by lazy-load monitor that triggers the `P_skills_t` penalty in the reward (brief §2.2) | Adding skill-vector extractor | 1 | +90 | §2.1-skills,§2.2-reward | `test_skill_vector_shape_and_bounds` + values in [0, 1] |
| T01-11 | Add `docs/diagrams/graph_overview.svg` — high-level architecture diagram (parser → builder → adapter → env) referenced from PRD + essay | Authoring graph-overview diagram | 1 | +0 | §2.1-diagram | SVG exists; referenced from PRD §Architecture and Essay §Method |
| T01-12 | Author `docs/SKILLS_ARCHITECTURE.md` (F15) — L1/L2/L3 theoretical deep-dive (skill hierarchy → composition → reuse) with ≥2 concrete usage examples per brief §2.1 mandate | Authoring SKILLS_ARCHITECTURE.md (F15) | 1 | +250 docs | F15,§2.1-skills | File exists; L1/L2/L3 sections present; ≥2 worked examples; cross-referenced from PRD §Skills |

## §3.2 Phase 2 — §2.2 Environment (⬜ pending)

Goal: define the RL environment that wraps the code graph — state vector,
action space, reward `R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t`
(brief §2.2 verbatim — P_skills is a NEGATIVE penalty applied on lazy-load
break), terminal conditions, and the action-masking service. No Gymnasium
import in `src/env/` (brief §2.2 mandate "ללא סביבת Gymnasium").

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T02-01 | Author `docs/STATE_DESIGN.md` — justify each state feature (token count, graph diameter, betweenness of refactor target, skill-coverage vector) | Authoring STATE_DESIGN.md | 2 | +120 docs | §2.2-state | Each feature has rationale + rejected-features list |
| T02-02 | Author `docs/ACTION_DESIGN.md` — discrete action set (extract-function, inline, rename, split-module, merge-module, noop) | Authoring ACTION_DESIGN.md | 2 | +90 docs | §2.2-action | Action space documented; one-hot dim recorded |
| T02-03 | Implement `src/env/refactor_env.py` — `reset()`, `step(a)`, reward eq, terminal (token-budget exhausted / skill-coverage met / step-limit) | Implementing RefactorEnv | 2 | +140 | §2.2-env | `test_env_reset_returns_state_vector` + `test_step_returns_obs_reward_done_info` |
| T02-04 | Implement `src/env/reward.py` — canonical `R_t = α·ΔModularity_t + β·ΔCohesion_t − γ·Coupling_Penalty_t + P_skills_t` (brief §2.2) with α=1.0, β=1.0, γ=0.5, P_skills=−5.0 read from `config/config.yaml` | Implementing canonical reward function | 2 | +80 | §2.2-reward | `test_reward_decomposition_matches_config_weights` + `tests/architecture/test_reward_formula.py` asserts exact term structure |
| T02-05 | Implement `src/env/action_mask.py` — mask illegal refactors (e.g. extract-function on non-function node) via logits → −∞ | Implementing action mask | 2 | +90 | §2.2-mask | `test_masked_logits_yield_zero_probability` |
| T02-06 | Implement `src/env/tokens.py` — wrap `tiktoken cl100k_base`; expose `count_tokens(text)` + chars/bytes appendix counters | Implementing tiktoken counter | 2 | +70 | §2.2-tokens | `test_tiktoken_cl100k_count_matches_reference` + appendix counters present |
| T02-07 | Add `tests/test_lazy_load.py` — walk `sys.modules` after `import src`, assert no heavy deps loaded (torch, sb3) before first call; assert P95 import-token cost under threshold | Adding lazy-load + token P95 test | 2 | +90 test | §2.2-lazy | Test passes; threshold in `config/config.yaml` |
| T02-08 | Implement `src/env/episode_logger.py` — log per-step (s, a, r, mask, tokens) JSONL for replay/debug | Implementing episode logger | 2 | +80 | §2.2-log | `test_episode_logger_emits_jsonl_per_step` |
| T02-09 | Implement `src/utils/seeding.py` — `set_global_seed(seed)` seeding `random`, `numpy`, `torch` (CPU+CUDA), `PYTHONHASHSEED`; `cudnn.deterministic=True`, `benchmark=False`; `tests/test_reproducibility.py` asserts two consecutive forward passes are bitwise-equal | Authoring seeding utility + reproducibility test | 2 | +100 + 60 test | §2.2-repro | `test_reproducibility_two_consecutive_forwards` passes; CUDA caveats documented in README |
| T02-10 | Add `tests/test_env_smoke.py` — full episode rollout with a random policy from `reset()` to `done=True`, asserting reward, mask, and JSONL log all line up | Adding env smoke test | 2 | +70 test | §2.2-env | Random-policy rollout terminates within step-limit and writes a non-empty JSONL log |
| T02-11 | Add `tests/architecture/test_env_no_gym.py` — AST-level walk of `src/env/` asserting **no** `import gymnasium` (brief §2.2 "ללא סביבת Gymnasium") — grep is insufficient because it misses aliased imports | Adding no-gym architecture test | 2 | +60 test | §2.2-no-gym | Test fails if any `src/env/**/*.py` AST contains `Import(name='gymnasium')` or `ImportFrom(module='gymnasium')` |
| T02-12 | Add `tests/architecture/test_reward_formula.py` — parse `src/env/reward.py` AST + assert four canonical terms (`ΔModularity`, `ΔCohesion`, `Coupling_Penalty`, `P_skills`) are all present and combined with the right signs (γ subtracted; P_skills added as negative) | Adding reward-formula architecture test | 2 | +80 test | §2.2-reward | Test fails on any stale formulation (Δtokens, Δdistance, Δskill_coverage, ΔReuse, ΔQ_struct, ΔQ_runtime) |
| T02-13 | Add `tests/architecture/test_betweenness_call_count.py` — patch `networkx.betweenness_centrality` and assert it is invoked **exactly twice per seed** (training-start + training-end) across ≥5 seeds per brief §2.2 + ADR-006 | Adding betweenness-call-count architecture test | 2 | +80 test | §2.2-betweenness,ADR-006 | Test fails if call count per seed ≠ 2 |
| T02-14 | Add `tests/architecture/test_config_refs.py` — walk docs/ and src/ for any reference to the legacy split-config filenames (state/action/reward `.yaml`); assert all references use `config/config.yaml#<block>` notation (CLAUDE.md §4 single-source-of-truth); the test file itself owns the literal stale strings in its `STALE_REFS` tuple | Adding config-refs architecture test | 2 | +60 test | CLAUDE.md§4 | Test fails on any stale split-config filename |

## §3.3 Phase 3 — §2.3 PPO + GAE (⬜ pending)

Goal: implement PPO with GAE on the refactor env. Spike Stable-Baselines3
for 2h; if SB3 buffer doesn't match our action-mask shape, fall back to
custom padding + masking.

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T03-01 | Implement `src/policy/policy_net.py` — actor-critic torso, Categorical head over action set, value head V_φ(s) | Implementing actor-critic net | 3 | +130 | §2.3-policy | `test_policy_outputs_logits_and_value` |
| T03-02 | Implement `src/training/gae.py` — GAE-λ advantage Â_t per Schulman et al. 2016 | Implementing GAE advantage | 3 | +90 | §2.3-gae | `test_gae_matches_hand_computed_example` |
| T03-03 | Implement `src/training/ppo_loss.py` — clipped surrogate L^CLIP + value loss + entropy bonus | Implementing PPO clipped loss | 3 | +110 | §2.3-loss | `test_ppo_clipped_loss_formula` + `test_entropy_bonus_present` |
| T03-04 | 2h SB3 spike — wire `stable_baselines3.PPO` on env, evaluate buffer compatibility with action mask | Spiking SB3 for 2h | 3 | +60 | §2.3-spike | `docs/adr/ADR-008-sb3-variable-v-buffer.md` records outcome (chose SB3 / chose fallback) |
| T03-05 | Implement custom PPO trainer fallback `src/training/ppo_trainer.py` (used if SB3 spike fails) — rollout buffer with padding + mask | Implementing custom PPO trainer | 3 | +140 | §2.3-trainer | `test_ppo_trainer_runs_one_update_cycle` |
| T03-06 | Implement `src/training/convergence.py` — dual criterion (rolling-100 reward ±2% × 50 episodes AND entropy < floor) | Implementing dual convergence check | 3 | +90 | §2.3-conv | `test_dual_convergence_requires_both_conditions` |
| T03-07 | Implement `src/sdk.py` — single facade exposing `build_graph()`, `train_ppo()`, `evaluate()`, `get_metrics()`, `save_run()` | Implementing SDK facade | 3 | +140 | §2.3-sdk | `test_sdk_is_single_entry_point` (CLI, notebook import only SDK) |
| T03-08 | Run PPO end-to-end on a single seed; emit `results/ppo_reward_curve.png` + `results/ppo_entropy_curve.png` | Running first end-to-end PPO seed | 3 | +60 | §2.3-run | Both charts exist; caption cites seed + episode count |
| T03-09 | Implement `src/cli/menu.py` + `main.py` — terminal entry (`uv run main.py` boots in ≤20 LOC `main.py`); menu lists build-graph / train-ppo / evaluate / ablate / show-cost | Implementing CLI menu | 3 | +130 | §2.3-cli | `uv run main.py` boots; menu lines match the 5 entries above |
| T03-10 | Author `docs/THEORY.md` — PPO clipped surrogate, GAE-λ derivation, value-loss + entropy bonus, with cross-reference table to `src/training/` modules (PPO: Schulman et al. 2017 arXiv:1707.06347; GAE: Schulman et al. 2016 arXiv:1506.02438) | Authoring THEORY.md (PPO + GAE) | 3 | +220 docs | §2.3-theory | All 4 equations render in LaTeX; cross-ref table names exact `src/` module per equation; both arXiv IDs cited |
| T03-11 | Render `results/learning_curves/reward_vs_episode.png` (D6) — mean ± 95% CI over ≥5 seeds of episode reward across training; caption names seeds, episode count, and rolling window | Rendering learning-curve PNG (D6) | 3 | +90 | D6,§2.3-run | PNG exists; caption includes seed list, episode count, mean ± 95% CI band; aggregated across ≥5 seeds |
| T03-12 | Author `tests/test_learning_curve.py` (F16) — assert `results/learning_curves/reward_vs_episode.png` exists after a short training run AND assert ΔReward (final − initial mean) is computed and stored numerically (not just plotted) | Authoring learning-curve test (F16) | 3 | +80 test | F16,D6,D7 | Test fails if PNG missing OR if ΔReward numeric not extractable from results artifact |

## §3.4 Phase 4 — §2.4 Cost + ablations + essay (⬜ pending)

Goal: token-cost panel via tiktoken, full α/β/γ/P_skills ablation matrix
(≥5 seeds/cell), and the 2500–3000 word essay with 8–12 citations + 2
diagrams.

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T04-01 | Author `docs/COST_ANALYSIS.md` (D8) — tiktoken cl100k headline + chars/bytes appendix, prompts × tokens × $ table, AI-tooling cost section, **cost envelope** (architect-decided spend cap with running total vs envelope) | Authoring COST_ANALYSIS.md (D8) | 4 | +180 docs | D8,§2.4-cost | Headline number is tiktoken; appendix has chars + bytes; ≥1 cost table; cost envelope section names cap + actual spend |
| T04-02 | Build ablation harness `scripts/run_ablations.py` — sweep α ∈ {…}, β ∈ {…}, γ ∈ {…}, P_skills ∈ {…} × 5 seeds/cell | Building ablation harness | 4 | +130 | §2.4-ablate | Harness writes one row per (cell × seed) to `results/ablations/raw.csv` |
| T04-03 | Run full ablation matrix; aggregate `results/ablations/raw.csv` → `results/ablations/summary.csv` with mean ± std + 95% CI per cell | Running full ablation matrix | 4 | +0 | §2.4-ablate | `summary.csv` has ≥1 row per cell with mean, std, ci_lo, ci_hi |
| T04-04 | Render ablation charts — heatmap per (α, β) cell, line plot per γ, bar chart per P_skills | Rendering ablation charts | 4 | +110 | §2.4-charts | `results/ablations/heatmap_ab.png`, `line_gamma.png`, `bar_pskills.png` exist |
| T04-04b | Author `docs/ANALYSIS.md` §ΔReward (D7) — final − initial mean reward as mean ± std + 95% CI across ≥5 seeds; consume `results/learning_curves/reward_vs_episode.png` (D6) and the per-seed reward arrays; cross-link to ESSAY §Results | Authoring ΔReward section (D7) | 4 | +120 docs | D7,§2.4-essay | ANALYSIS.md exists with ΔReward block; numeric value reported with mean ± std + 95% CI; ≥5 seeds |
| T04-05 | Author `docs/ESSAY.md` — 2500–3000 words, 4 sections (motivation, method, results, limitations), 8–12 citations, 2 diagrams; Results section cites D6 learning curve + D7 ΔReward | Authoring §2.4 essay | 4 | +900 docs | §2.4-essay | Word count in [2500, 3000]; ≥8 citations; 2 diagrams referenced; D6 + D7 cross-linked |
| T04-06 | Author `notebooks/analysis.ipynb` §Cost + §Ablations + §Essay-summary — consumes SDK only | Authoring notebook §Cost/Ablations | 4 | +200 nb | §2.4-nb | Notebook imports SDK only (no parallel impl); LaTeX blocks precede plots |
| T04-07 | Author `docs/shared/PROMPTS.md` — verbatim prompts used (architect → implementer trail per §1.4) with human-judgment annotations | Authoring PROMPTS.md | 4 | +200 docs | §1.4 | Every prompt mapped to a commit hash; decisions annotated |
| T04-08 | Run final gate sweep — ruff clean, all `.py` ≤150 LOC, coverage ≥85%, notebook executes top-to-bottom, no PII matches | Running final gate sweep | 4 | 0 | (gates) | `make check` (or scripted equivalent) exits 0; `grep -E "REDACTED-NAME\|REDACTED-HANDLE\|REDACTED\|REDACTED-ID\|GoogleDrive-REDACTED-HANDLE"` returns zero matches |
| T04-09 | Export submission PDF `adrl-001-ex04.pdf` (cover sheet is the **only** numeric self-grade location) | Exporting submission PDF | 4 | +0 | (submission) | PDF exists at repo root; cover sheet has self-grade; no other doc has numeric self-grade |
| T04-10 | Invite `rmisegal` as read-only collaborator on the A4 GitHub repo; record invitation ID in `docs/SUBMISSION.md` | Inviting rmisegal as read-only collaborator | 4 | +0 | (submission) | Invitation sent with **Read** role; invitation ID recorded |
| T04-11 | Tag submission commit `assignment-4`, push branch | Tagging submission and pushing branch | 4 | +0 | (submission) | Tag pushed |

## §3.5 Cross-phase invariants (recheck at every phase boundary)

These are not single tasks — they are properties the codebase must hold
at the end of every phase. The phase is only ✅ when every invariant
below is satisfied.

| invariant | how it is checked | first phase enforced |
|---|---|---|
| Every `.py` ≤150 LOC | pre-commit hook (T00-02) + CI | 0 |
| Ruff clean | `uv run ruff check src/ tests/ main.py` exits 0 | 0 |
| Coverage ≥85% | `uv run pytest --cov=src --cov-report=term-missing` | 1 |
| SDK is single business-logic entry | `test_sdk_is_single_entry_point` (T03-07); CLI + notebook import only SDK | 3 |
| Notebook is a consumer, not a parallel impl | `grep -E "^class \|^def " notebooks/analysis.ipynb` returns only thin wrappers | 4 |
| Betweenness reported with mean ± std + 95% CI across ≥5 seeds | `test_betweenness_summary_has_mean_std_ci` (T01-08) | 1 |
| Betweenness called exactly twice per seed (start + end) | `tests/architecture/test_betweenness_call_count.py` (T02-13) | 2 |
| No Gymnasium import anywhere in `src/env/` | `tests/architecture/test_env_no_gym.py` (T02-11) AST check | 2 |
| Reward formula matches canonical brief §2.2 (`α·ΔModularity + β·ΔCohesion − γ·Coupling_Penalty + P_skills`) | `tests/architecture/test_reward_formula.py` (T02-12) AST check | 2 |
| Config refs use `config/config.yaml#<block>` (no `state.yaml`/`action.yaml`/`reward.yaml`) | `tests/architecture/test_config_refs.py` (T02-14) | 2 |
| Convergence asserted via dual criterion | `test_dual_convergence_requires_both_conditions` (T03-06) | 3 |
| Learning curve PNG (D6) exists + ΔReward (D7) numeric | `tests/test_learning_curve.py` (T03-12) | 3 |
| Ablation cells have ≥5 seeds and CI | `summary.csv` schema check in T04-03 | 4 |
| Cost envelope (D8) recorded in `docs/COST_ANALYSIS.md` | T04-01 acceptance check | 4 |
| No PII matches against deny-list | `grep -E "REDACTED-NAME\|REDACTED-HANDLE\|REDACTED\|REDACTED-ID\|GoogleDrive-REDACTED-HANDLE"` returns zero matches in tree (the literal pattern lives only in this TODO row and in CLAUDE.md) | 0 |
| Commit subject matches `^(Phase \d+\|Phase 0 bootstrap\|chore: bootstrap)` | git log walk in T04-08 final sweep | 0 |

## §3.6 Phase 0.1 — Review closure (✅ landed)

Status: ✅ landed by this multi-pass review-closure workflow. The pass
swept master-PRD drift across PRD / TRACE / PLAN / TODO / STATE_DESIGN /
ACTION_DESIGN and pinned the canonical values (reward equation,
GraphifyAdapter signature, betweenness call discipline, V_max fallback
policy, Gymnasium ban, single-config-file invariant, A_max arithmetic,
Schulman arXiv IDs, brief §-id discipline, new F15/F16/D6/D7/D8
artefacts). Listed here for traceability — the actual artefact tasks
live in Phases 1–4 above.

| id | content | activeForm | phase | LOC-Δ | req | acceptance |
|----|---------|------------|-------|-------|-----|------------|
| T001-01 | Sweep PRD / TRACE / PLAN / TODO / STATE_DESIGN / ACTION_DESIGN against the master canonical-values block (reward eq, GraphifyAdapter sig, betweenness call count, V_max policy, Gymnasium ban, single-config-file, A_max=45057, Schulman IDs, brief §-id discipline) | Sweeping canonical-values drift | 0.1 | docs-only | (governance) | Every doc cites the canonical reward equation `R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling_Penalty + P_skills`; no stale Δtokens / Δdistance / Δskill_coverage / ΔReuse / ΔQ_struct / ΔQ_runtime references remain |
| T001-02 | Register new artefact ids F15 (SKILLS_ARCHITECTURE.md), F16 (test_learning_curve.py), D6 (reward_vs_episode.png), D7 (ΔReward in ANALYSIS.md), D8 (COST_ANALYSIS.md cost envelope) into TRACE.md and TODO.md | Registering new F/D ids | 0.1 | docs-only | F15,F16,D6,D7,D8 | All five ids appear in TRACE.md row index AND in TODO.md tasks T01-12, T02-11..14, T03-11, T03-12, T04-01, T04-04b |
| T001-03 | Reconcile V_max wording — STATE_DESIGN labels V_max as CONDITIONAL (post-ADR-008 spike); ADR-008 status matches ("spike-gated" or "accepted-with-fallback" — pick one and propagate) | Reconciling V_max policy across docs | 0.1 | docs-only | ADR-004,ADR-008 | STATE_DESIGN + ADR-008 use identical status string for V_max |
