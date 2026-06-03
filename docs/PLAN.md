# PLAN — architecture & design (PythonClawRefactorRL)

Design/architecture doc (§2.2) for Assignment 4 — *PPO + GAE over a
GRAPHIFY skills-graph refactor environment* with a `pythonclaw` shim
behind ADR-001. Single source-of-truth aggregating layering, ADRs,
interface contracts, the parallel-processing decision, and the 4-phase
build sequence. Mirrors A3's PLAN structure so grader audit transfers.

> **Scope note (honesty).** Refactor-as-RL over a synthetic skills
> graph induced from one mid-sized Python codebase (the PythonClaw
> repo) — not a population study. Four caveats: GRAPHIFY is a local
> re-impl behind an adapter, not a port; lazy-load-broken is over-
> approximated by a dependency-cycle proxy; reward weights α/β/γ are
> calibrated, not derived; SB3 buffer assumes fixed |V|, handled via
> padding/masking per ADR-008.

## §1 — Architecture (C4-Container, mermaid)

**Context.** One user (or grader) drives PythonClawRefactorRL; the
only external system is a local Python codebase snapshot that GRAPHIFY
ingests. No network, no DB, no inference server, no auth surface.

**Containers / layers** (External → SDK → Services → Domain → Infrastructure):

```mermaid
C4Container
    title PythonClawRefactorRL — C4 Container View (§1)

    Person(user, "User / Grader", "Drives training, ablation, evaluation via CLI, GUI, or notebook")

    System_Boundary(app, "PythonClawRefactorRL (single-host uv app)") {
      Container(cli,   "CLI (cli/menu.py)",                 "Python · stdlib",          "Terminal menu — depends ONLY on SDK")
      Container(gui,   "GUI (gui/app.py + pages/)",         "Streamlit ≥1.40 · plotly", "Multi-page dashboard (theme · state · components · charts · callbacks · data_helpers · explanations) — depends ONLY on SDK; live PPO curves via st.empty()")
      Container(nb,    "Analysis Notebook",                 "Jupyter · imports SDK",    "Ablation matrix, PPO+GAE curves, centrality plots, screenshots gallery")
      Container(sdk,   "RefactorSDK (sdk/sdk.py)",          "Python facade",            "SINGLE business-logic entry point — CLAUDE.md §3")
      Container(svc,   "Services",                          "ppo · gae · centrality · evaluation", "PPOService, GAEComputer, CentralityProbe, RolloutCollector, AblationRunner, ConvergenceMonitor")
      Container(env,   "Environment",                       "SkillsGraphEnv (Gymnasium)", "reset/step over GRAPHIFY graph · 4-action refactor space · variable |V| with padding to max_nodes_v=512")
      Container(model, "Models",                            "torch.nn",                 "GraphSAGE/MLP node encoder · ActorCriticHead (PPO)")
      Container(graph, "GRAPHIFY (local)",                  "networkx · ast",           "Code → skills graph adapter (GraphifyAdapter) · NodeFeaturizer · EdgeBuilder")
      Container(claw,  "pythonclaw shim",                   "stub behind ADR-001",      "24h swap window: thin import-shim emitting NotImplementedError for unbuilt paths; real claw bound later if available")
      Container(data,  "Data layer",                        "filesystem",               "RepoSnapshot, GraphCache (pickle), token counter (tiktoken cl100k_base + chars/bytes)")
      ContainerDb(store, "Local artifacts",                 "filesystem",               "results/vault/* (Obsidian) · results/figures/* · results/ablation/* · checkpoints")
    }

    System_Ext(repo, "Local Python codebase", "PythonClaw snapshot (one-time read from disk; no network)")

    Rel(user, cli,  "uv run main.py")
    Rel(user, gui,  "uv run streamlit run src/refactorrl/gui/app.py")
    Rel(user, nb,   "uv run jupyter")
    Rel(cli, sdk,   "calls")
    Rel(gui, sdk,   "calls")
    Rel(nb,  sdk,   "imports & calls (consumer only)")
    Rel(sdk, svc,   "orchestrates")
    Rel(svc, env,   "rollouts")
    Rel(svc, model, "forward / backward")
    Rel(env, graph, "queries current skills graph")
    Rel(graph, claw,"optional richer parse (ADR-001 shim)")
    Rel(graph, data,"reads/writes graph cache")
    Rel(data, repo, "AST walk (first call only)")
    Rel(svc, store, "checkpoints, ablation results, figures")
```

**Dependency rule.** Arrows point downward only — a UI/notebook never
imports an engine module; the SDK is the seam (CLAUDE.md §3). The
`pythonclaw_shim` package is the *only* place that ever names
`pythonclaw` so a 24h swap to a real implementation needs no consumer
edits (ADR-001).

**Action space.** Discrete, cardinality **4**:
`{ 0:MoveSymbol, 1:ExtractFunction, 2:InlineSymbol, 3:NoOp }`.
Variable-|V| graphs are padded to `max_nodes_v=512` with a boolean mask
(ADR-008) so the SB3 fixed-shape rollout buffer stays valid.

## §2 — Component responsibilities

| Layer | Modules (under `src/refactorrl/`) | Responsibility | Forbidden |
|---|---|---|---|
| **data** | `data/repo_snapshot.py`, `data/graph_cache.py`, `data/token_counter.py` | One-time AST walk of the target repo; pickle cache of the induced graph; tiktoken cl100k_base headline + chars/bytes appendix (ADR-003) | No torch, no RL logic |
| **graphify** | `graphify/adapter.py`, `graphify/featurizer.py`, `graphify/edge_builder.py`, `graphify/lazy_load_probe.py` | Local re-impl of the GRAPHIFY contract behind `GraphifyAdapter` (ADR-002); node features (LOC, fan-in/out, imports, role); edge types (call, import, inherit); `lazy_load_probe` over-approximates §"lazy-load-broken" via dependency-cycle detection (ADR-005) | No env stepping; no model code |
| **pythonclaw_shim** | `pythonclaw_shim/__init__.py`, `pythonclaw_shim/contract.py` | Thin import-shim that exposes the names a "real" `pythonclaw` would; raises `NotImplementedError` for unbuilt paths so the 24h swap window is a one-PR change (ADR-001) | Any business logic — must remain a façade |
| **env** | `env/skills_graph_env.py`, `env/reward.py`, `env/action_mask.py`, `env/padding.py` | Gymnasium-style `reset/step` over the skills graph; reward `r = α·ΔModularity + β·ΔCohesion − γ·Coupling + P_skills` (ADR-007); action-masking on inapplicable refactors; fixed-shape obs via padding to `max_nodes_v` (ADR-008) | No model training; no UI; no disk I/O on hot path |
| **model** | `model/graphsage_encoder.py`, `model/mlp_encoder.py`, `model/actor_critic_head.py` | `nn.Module` definitions only; GraphSAGE encoder is the headline (ADR-004), MLP-on-pooled-features is the ablation baseline; ActorCritic head emits `(logits[4], value)` | No env access; no training loop; no I/O |
| **services** | `services/ppo_service.py`, `services/gae_computer.py`, `services/rollout_collector.py`, `services/centrality_probe.py`, `services/convergence_monitor.py`, `services/ablation_runner.py`, `services/evaluator.py` | PPO clipped objective (ex04 §2.3, ε=0.2 FIXED); GAE(γ,λ) with λ=0.95 FIXED; centrality probe ≥5 seeds × ≥1 call/seed (ADR-006); dual convergence criterion (rolling-100 reward ±2% × 50 episodes AND entropy<0.5); full α/β/γ + P_skills ablation matrix | No direct user I/O |
| **sdk** | `sdk/sdk.py`, `sdk/types.py` | Single façade — orchestrates services; the **only** import surface for UIs and the notebook; explicit dataclasses cross the boundary, never raw dicts | No training math inline; no `print()` |
| **cli** | `cli/menu.py`, `cli/__main__.py` | Numeric terminal menu; argparse entry | Imports anything below SDK |
| **gui** | `gui/app.py` (Streamlit entry), `gui/pages/` (one file per surface), `gui/theme.py` (Bar-Ilan `#003D7A` / `#FFCD00`), `gui/state.py` (namespaced `gui.<page>.<key>`), `gui/components.py`, `gui/charts.py` (plotly + matplotlib factories, DPI=110), `gui/callbacks.py` (observer hooks for live PPO charts), `gui/data_helpers.py` (`@st.cache_data` SDK wrappers), `gui/explanations.py` (LaTeX blurbs for PPO clip + GAE) | **Direct imports of `env`, `model`, `services`** — all logic via SDK; no inline `plt.*` outside `charts.py`; no un-namespaced `st.session_state` keys |
| **utils** | `utils/seeding.py`, `utils/io.py` | Deterministic seeding (random, numpy, torch CPU+CUDA, `PYTHONHASHSEED`, cuDNN flags); atomic checkpoint writes | No business logic |
| **notebook** | `notebooks/analysis.ipynb` | Ablation matrix, PPO+GAE curves, centrality plots, Obsidian hero shots as a *consumer* of the SDK; LaTeX cells next to plots | Re-implementing engine logic |

Every `src/**/*.py` ≤ **150 code lines** (CLAUDE.md §1). DRY-shared
helpers live in `services/_shared.py` / `model/_blocks.py`.

## §3 — Data flow (training step, mermaid sequence)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant SDK as RefactorSDK
    participant GA as GraphifyAdapter
    participant ENV as SkillsGraphEnv
    participant POL as ActorCritic
    participant ROL as RolloutCollector
    participant GAE as GAEComputer
    participant PPO as PPOService
    participant MON as ConvergenceMonitor

    U->>SDK: train(seed, episodes)
    SDK->>GA: build_graph(repo_snapshot)
    GA-->>SDK: G (networkx DiGraph, |V|≤512)
    SDK->>ENV: reset(seed, G)
    ENV-->>SDK: s_0 (padded obs + mask)
    loop n_steps=128 transitions
      SDK->>POL: forward(s_t, mask)
      POL-->>SDK: (logits[4], V(s_t))
      SDK->>ENV: step(a_t)
      ENV-->>SDK: (s_{t+1}, r_t, done, info)
      Note over ENV: r_t = α·ΔMod + β·ΔCoh − γ·Coupling + P_skills
      SDK->>ROL: append(s_t, a_t, r_t, V(s_t), logπ(a_t), done)
    end
    SDK->>GAE: compute_advantages(rollout, γ=0.99, λ=0.95)
    GAE-->>SDK: Â_t, R̂_t (normalized)
    SDK->>PPO: update(rollout, Â, R̂, ε=0.2, epochs=4, batch=64)
    PPO-->>SDK: losses {policy_clip, value, entropy, kl}
    SDK->>MON: log(reward_t, entropy_t)
    MON-->>SDK: converged? (rolling-100 ±2% × 50 AND H<0.5)
```

Centrality is measured *out-of-band*: once per seed × ≥5 seeds (start +
end + final-eval = ~3 calls/seed) — never on the hot rollout path
(ADR-006). Variable-|V| is handled by padding inside `env/padding.py`
before the rollout buffer ever sees the obs, so SB3's fixed-shape
assumption holds (ADR-008).

## §4 — Hyperparameter tree

Every algorithm-relevant knob lives in `config/config.yaml`
(CLAUDE.md §4). Local UI styling (plotly DPI, button px) stays in its
rendering module. Tree shape:

```yaml
# config/config.yaml (authoritative — excerpt; see file for full)
version: "1.2.0"

ppo:
  clip_eps: 0.2          # FIXED by ex04 §2.3 — asserted in tests
  n_steps: 128
  batch_size: 64
  n_epochs: 4
  learning_rate: 3.0e-4
  vf_coef: 0.5
  ent_coef: 0.01
  max_grad_norm: 0.5

gae:
  lambda: 0.95           # FIXED by ex04 §2.3 — asserted in tests
  gamma: 0.99

reward:                  # MUST per locked decisions — full ablation
  alpha: 1.0             # ΔModularity weight
  beta: 1.0              # ΔCohesion weight
  gamma: 0.5             # Coupling penalty
  p_skills: -5.0         # lazy-load-break penalty (BIG NEGATIVE)

centrality:
  betweenness_calls_per_seed: 3   # start + end + final-eval

seeds: [42, 7, 123, 314, 271]     # ≥5 for statistical claims

training:
  max_episodes: 500
  convergence:
    rolling_window: 100
    reward_drift_pct: 2.0
    consecutive_episodes: 50
    entropy_threshold: 0.5

environment:
  max_nodes_v: 512        # padding cap for variable-|V| handling (ADR-008)

paths:
  vault_dir: "results/vault/"
  figures_dir: "results/figures/"
  ablation_dir: "results/ablation/"
```

**FIXED constants** (`ppo.clip_eps`, `gae.lambda`) are asserted in
`tests/test_ppo_constants.py` — tuning them fails CI. **Calibrated**
constants (reward weights, ent_coef, lr) sweep across the ablation
matrix (≥5 seeds/cell, mean ± std + 95% CI).

## §5 — PPO + GAE training loop sketch

Pseudocode below is the contract `services/ppo_service.py` +
`services/gae_computer.py` jointly satisfy. Loss math pins to ex04 §2.3
and is unit-tested in `tests/test_ppo_loss_formula.py`.

```python
# services/ppo_service.py (sketch — real file ≤150 LOC)
def train(env, policy, cfg, seeds):
    for seed in seeds:                                # ≥5 (ADR-006)
        set_global_seed(seed)
        for episode in range(cfg.training.max_episodes):
            rollout = collector.collect(env, policy, n_steps=cfg.ppo.n_steps)
            with torch.no_grad():
                last_v = policy.value(rollout.last_obs, rollout.last_mask)
            adv, ret = gae.compute(                   # GAE(γ=0.99, λ=0.95) FIXED
                rewards=rollout.rewards,
                values=rollout.values,
                dones=rollout.dones,
                last_value=last_v,
                gamma=cfg.gae.gamma,
                lam=cfg.gae.lambda,
            )
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # advantage norm
            for _ in range(cfg.ppo.n_epochs):               # 4 epochs FIXED
                for batch in minibatches(rollout, cfg.ppo.batch_size):
                    logp, entropy, value = policy.evaluate(batch.obs, batch.act, batch.mask)
                    ratio = torch.exp(logp - batch.old_logp)
                    surr1 = ratio * batch.adv
                    surr2 = torch.clamp(ratio, 1 - cfg.ppo.clip_eps,
                                               1 + cfg.ppo.clip_eps) * batch.adv
                    pol_loss = -torch.min(surr1, surr2).mean()        # ε=0.2 FIXED
                    val_loss = F.mse_loss(value, batch.ret)
                    ent_bonus = entropy.mean()
                    loss = (pol_loss
                            + cfg.ppo.vf_coef * val_loss
                            - cfg.ppo.ent_coef * ent_bonus)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(),
                                             cfg.ppo.max_grad_norm)
                    optimizer.step()
            if monitor.converged(rollout.episode_rewards, entropy_bonus=ent_bonus):
                break                                  # dual criterion (locked decision)
```

Key invariants tested in `tests/`:
- `test_gae_formula.py` — closed-form check on a 4-step toy trajectory.
- `test_ppo_clip_invariant.py` — when ratio is inside `[1−ε, 1+ε]`, gradient equals the unclipped surrogate gradient.
- `test_advantage_normalization.py` — post-norm advantage has mean ≈ 0, std ≈ 1.
- `test_convergence_dual_criterion.py` — neither criterion alone triggers a "converged" verdict.

## §6 — Parallel processing scope (single-process choice + rationale)

**Single-process, single-threaded by design**, same call as A3 because
the cost profile is identical in shape (CPU-bound forward/backward +
trivial I/O):

- Cost centres are **PPO forward/backward** (2–3-layer GraphSAGE or
  MLP on ≤512 nodes) and **GAE accumulation** (single backward scan
  over `n_steps=128`). Both CPU-bound; GIL does not bind.
- I/O is trivial: GRAPHIFY runs once per seed (cached pickle);
  centrality ≤3 times/seed (ADR-006). No worker pool warranted.
- Vector envs help only if `env.step` is the bottleneck; profile target
  is `<5 ms/step` at |V|=512 — forward/backward dominates by 10×.
- Reproducibility (per ADR-006) is materially harder under
  multiprocessing — per-worker seeding + cuDNN-deterministic flags
  become joint failure modes. We pay the single-process tax.

**State assumptions.** `SkillsGraphEnv` holds the current graph + edit
history (reset rebuilds from cached `GraphifyAdapter`); not
thread-safe — concurrent rollouts would need one env per worker.
`RolloutCollector` uses fixed-size tensors `[n_steps, ...]`; no replay
buffer. Checkpoints written atomically (`*.pt.tmp` → rename).

Upgrade path if env-step becomes the bottleneck:
`gymnasium.vector.AsyncVectorEnv` *behind the SDK boundary* — consumers
do not change. Recorded as ADR-010 re-open trigger.

## §7 — 4-phase roadmap with deliverables per phase

Sequenced phase-milestones; each phase is a single TDD commit
(RED → GREEN → REFACTOR) on branch `assignment-4`. Commit subject regex
`^(Phase \d+|Phase 0 bootstrap|chore: bootstrap)` enforced locally.

| # | Phase | Deliverables | Tests gating the phase |
|---|---|---|---|
| **0** | **Bootstrap** (already underway) | `pyproject.toml`, `config/config.yaml`, `pythonclaw_shim/` skeleton, ADR-001…009 stubs, ruff + uv green | `test_bootstrap_smoke.py` (import every package), `test_ppo_constants.py` (FIXED constants present) |
| **1** | **GRAPHIFY adapter + repo ingest** | `GraphifyAdapter.build_graph(repo_path) → nx.DiGraph`; `NodeFeaturizer` (LOC, fan-in/out, role); `EdgeBuilder` (call, import, inherit); `lazy_load_probe` cycle detector; tiktoken token-count headline; graph cache pickle | `test_graphify_adapter.py`, `test_node_features.py`, `test_edge_types.py`, `test_lazy_load_probe.py` (broken-cycle detected), `test_token_count_headline.py` (cl100k_base + chars/bytes appendix) |
| **2** | **Env + reward + masking + padding** | `SkillsGraphEnv.reset/step` (Gymnasium API); reward eq. `α·ΔMod + β·ΔCoh − γ·Coupling + P_skills`; `ActionMaskService`; padding to `max_nodes_v=512` | `test_env_step_shapes.py`, `test_reward_signs.py` (each term flips reward in the expected direction), `test_action_mask.py` (masked → prob 0 post-softmax), `test_padding_invariance.py` (policy output equals dense path for unmasked nodes) |
| **3** | **PPO + GAE service + convergence + centrality probe** | `GAEComputer.compute(γ,λ)`; `PPOService.train(seeds)`; `ConvergenceMonitor` (dual criterion); `CentralityProbe` (≥5 seeds, ≤3 calls/seed); `RolloutCollector` | `test_gae_formula.py`, `test_ppo_clip_invariant.py`, `test_advantage_normalization.py`, `test_convergence_dual_criterion.py`, `test_centrality_discipline.py` (call-count cap enforced) |
| **4** | **SDK facade + CLI + GUI + notebook + ablation matrix + screenshots + §2.4 essay** | `RefactorSDK.{build_graph, train, evaluate, ablate, recommend_refactor}`; `cli/menu.py`; `gui/` Streamlit dashboard; `notebooks/analysis.ipynb`; full α/β/γ + P_skills ablation matrix (≥5 seeds/cell); programmatic NetworkX/pyvis screenshots + Obsidian hero shots; `docs/essay_2_4.md` (2500–3000 words, 4 sections, 8–12 citations, 2 diagrams) | `test_sdk_smoke.py`, `test_cli_menu.py`, `test_gui_foundation.py`, `test_notebook_runs_nbclient.py`, `test_ablation_matrix_complete.py`, `test_screenshots_exist.py` |

**Definition of Done (per phase)** — PRD approved, TDD commit landed,
ruff zero, ≤150 LOC/file, coverage ≥85%, ADR updated if architecture
shifted, `docs/shared/PROMPTS.md` records the literal prompt used.

## §8 — ADR registry (10 ADRs)

ADRs live in `docs/adr/` and follow *Context → Decision → Consequences*.
Nine already drafted in bootstrap; ADR-010 closes out the parallel-
processing decision so §6 is auditable.

| ADR | Title | Why it matters |
|---|---|---|
| **ADR-001** | PythonClaw shim boundary (24h swap window) | Keeps the `pythonclaw` dependency name in exactly one package so a real implementation drop-in is a 1-PR change; consumers import the shim, never the upstream name. |
| **ADR-002** | GraphifyAdapter — local re-impl behind adapter | The external GRAPHIFY contract is unstable; we re-implement the minimal surface we need (`build_graph` + `NodeFeaturizer` + `EdgeBuilder`) behind a port, so a future port to the real tool is a single-file swap. |
| **ADR-003** | Tiktoken cl100k_base as headline cost metric (+ chars/bytes appendix) | Lecturer-grader-readable token count; cl100k_base is the de-facto LLM cost basis; chars/bytes kept as appendix so cl100k drift over time does not invalidate historical figures. |
| **ADR-004** | GraphSAGE vs MLP encoder | GraphSAGE is the headline (uses graph structure); MLP-on-pooled-features is the ablation baseline; lets us isolate the value of message passing. |
| **ADR-005** | Lazy-load-broken semantics via cycle detection | Honest over-approximation of the §"lazy-load-broken" signal; documented limitation: detects *necessary* but not *sufficient* break conditions. |
| **ADR-006** | Multi-seed eval discipline (≥5 seeds; centrality ≤3 calls/seed) | Forces statistical honesty (mean ± std + 95% CI) and bounds centrality cost (betweenness is O(VE)). |
| **ADR-007** | Reward upgrade MUST: `α·ΔMod + β·ΔCoh − γ·Coupling + P_skills` | Pins the reward shape to a defensible refactor-quality decomposition; full α/β/γ + P_skills ablation matrix proves each term matters. |
| **ADR-008** | SB3 variable-|V| buffer (padding + masking; 2h timebox spike) | SB3's rollout buffer requires fixed obs shape; we pad to `max_nodes_v=512` and carry a boolean mask; spike timebox is 2h before falling back to a hand-rolled buffer. |
| **ADR-009** | Screenshot pipeline (programmatic NetworkX/pyvis + Obsidian hero shots) | Deterministic, reproducible screenshots — no manual capture; each figure regenerable from `uv run scripts/render_figures.py`. |
| **ADR-010** | Single-process training; vector-env upgrade deferred (re-open if env-step >1ms) | Documents §6: reproducibility budget beats throughput at our scale; vector-env upgrade path through SDK boundary so consumers do not change. |

## §9 — OOP layer diagram

```mermaid
classDiagram
    direction LR
    class RefactorSDK {
      +build_graph(repo: Path) Graph
      +train(seed: int) TrainReport
      +evaluate(checkpoint: Path) EvalReport
      +ablate(grid: AblationGrid) AblationReport
      +recommend_refactor(state: GraphState) Action
    }
    class PPOService {
      -gae: GAEComputer
      -collector: RolloutCollector
      -monitor: ConvergenceMonitor
      +train(env, policy, cfg) TrainReport
    }
    class GAEComputer {
      +compute(rewards, values, dones, last_v, gamma, lam) tuple
    }
    class RolloutCollector {
      +collect(env, policy, n_steps) Rollout
    }
    class ConvergenceMonitor {
      +converged(rewards, entropy) bool
    }
    class CentralityProbe {
      +measure(graph, seed) CentralityReport
    }
    class AblationRunner {
      +run(grid, seeds) AblationReport
    }
    class SkillsGraphEnv {
      +reset(seed) Obs
      +step(action) tuple
      -graph: nx.DiGraph
      -mask: BoolTensor
    }
    class ActionMaskService {
      +mask(state) BoolTensor[4]
    }
    class GraphifyAdapter {
      +build_graph(repo) Graph
    }
    class NodeFeaturizer {
      +features(node) Tensor
    }
    class EdgeBuilder {
      +edges(ast) list
    }
    class LazyLoadProbe {
      +broken_cycles(graph) set
    }
    class ActorCriticHead {
      +forward(obs, mask) tuple
      +evaluate(obs, act, mask) tuple
      +value(obs, mask) Tensor
    }
    class GraphSAGEEncoder {
      +forward(graph) Tensor
    }
    class MLPEncoder {
      +forward(features) Tensor
    }
    class PythonClawShim {
      <<facade>>
      +parse(path) AST
    }

    RefactorSDK --> PPOService
    RefactorSDK --> AblationRunner
    RefactorSDK --> GraphifyAdapter
    RefactorSDK --> CentralityProbe
    PPOService --> GAEComputer
    PPOService --> RolloutCollector
    PPOService --> ConvergenceMonitor
    PPOService --> SkillsGraphEnv
    PPOService --> ActorCriticHead
    SkillsGraphEnv --> ActionMaskService
    SkillsGraphEnv --> GraphifyAdapter
    GraphifyAdapter --> NodeFeaturizer
    GraphifyAdapter --> EdgeBuilder
    GraphifyAdapter --> LazyLoadProbe
    GraphifyAdapter ..> PythonClawShim : optional (ADR-001)
    ActorCriticHead --> GraphSAGEEncoder
    ActorCriticHead --> MLPEncoder
    AblationRunner --> PPOService
```

**Inheritance discipline.** `GraphSAGEEncoder` and `MLPEncoder` both
implement a common `NodeEncoder` protocol (`forward(graph) → Tensor`)
so `ActorCriticHead` is encoder-agnostic — the ablation between them is
a single-line config swap, not a code edit. Shared loss/clip math
lives in `services/_shared.py` so `PPOService` and any future PPG/TRPO
sibling reuse it (DRY, CLAUDE.md §5). The SDK is the only public
import surface; every other module name is internal.
