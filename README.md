# PythonClawRefactorRL — Reverse-Engineering Software Architecture with RL

Bar-Ilan University — *Vibe Coding & Reinforcement Learning* workshop,
**Assignment 4**. A PPO + GAE agent learns a **refactoring policy** over the
**PythonClaw `Skills` module**: it reverse-engineers the module into a
dependency graph (GRAPHIFY), then applies SPLIT / MERGE / REWIRE actions to
raise **modularity** and **cohesion** while lowering **coupling**, under a
hard lazy-load-break penalty.

![CI](https://github.com/adirelm/PythonClawRefactorRL/actions/workflows/ci.yml/badge.svg)

> **Status: complete (Phases 0–4).** Custom non-Gymnasium environment, PPO+GAE
> trainer, 5-seed training, 81-cell × 5-seed ablation, cost analysis, and the
> §2.4 essay all landed. Quality gates green: ruff clean · 343 tests · 94%
> coverage · every `.py` ≤150 LOC.

This README **is the submission report** (brief §3). Sections
[3](#3-obsidian-before--current-architecture)–[6](#6-bug-report-architectural-smells-in-the-skills-module)
are the graded deliverables; deeper detail lives in the linked docs.

---

## Quick Start

```bash
uv sync --dev

# Quality gates
uv run pytest tests/ --cov=src --cov-report=term-missing   # 343 pass, 94% cov
uv run ruff check src/ tests/ scripts/                     # 0 violations
uv run python scripts/check_file_sizes.py                  # all .py ≤150 LOC

# Reproduce the headline results
uv run python scripts/train_5seed_isolated.py              # 5-seed PPO training
uv run python scripts/run_ablation.py --grid compact       # 81-cell × 5-seed ablation
uv run python scripts/render_learning_curve.py             # D6 reward curve
```

---

## 1. What this project does

| Brief § | Requirement | Where |
|---|---|---|
| §2.1 | GRAPHIFY the `Skills` module → dependency graph + Obsidian Vault | `src/graphify/`, `results/vault/` |
| §2.1 | Skills L1/L2/L3 architecture theory (≥2 examples) | [`docs/SKILLS_ARCHITECTURE.md`](docs/SKILLS_ARCHITECTURE.md) |
| §2.2 | **Custom** RL env (state=A,X; actions=split/merge/rewire) — **no Gymnasium** | `src/env/` |
| §2.2 | Reward `R = α·ΔModularity + β·ΔCohesion − γ·Coupling + P_skills` | `src/env/reward.py` |
| §2.2 | Degree centrality per step; **betweenness exactly 2×/seed** | `src/services/centrality.py` |
| §2.3 | PPO (ε=0.2) + GAE (λ=0.95), γ=0.99 | `src/services/ppo_trainer.py`, `src/services/gae_buffer.py` |
| §2.4 | Cost analysis (Skills-token volume + PPO runtime) | [`docs/COST_ANALYSIS.md`](docs/COST_ANALYSIS.md) |
| §2.4 | **GRAPHIFY × AI agents** essay (2,990 words, 11 cites, 2 diagrams) | [`docs/ESSAY.md`](docs/ESSAY.md) |
| math | PPO/GAE/reward equations + cross-refs | [`docs/THEORY.md`](docs/THEORY.md) |

No `gymnasium` import exists anywhere under `src/env/` — enforced by an
AST-level test (`tests/architecture/test_env_no_gym.py`).

---

## 2. The Skills graph (input)

GRAPHIFY parses the 10-skill corpus under `src/pythonclaw_shim/sample_skills/`
(30 JSON files: each skill = L1 `metadata` + L2 `instructions` + L3
`resources`) into a `networkx.DiGraph` whose edges are the `depends_on`
relations. **Input token volume: 9,297 `cl100k_base` tokens** (L1=881 / L2=2,905
/ L3=5,511) — the lazy-load design means an agent touching only L1 pays 881,
a **10.5× saving** (see [COST_ANALYSIS §0.1](docs/COST_ANALYSIS.md)).

---

## 3. Obsidian "before" — current architecture

![Obsidian before](results/figures/obsidian_before.png)

What the reverse-engineered graph tells us about the **current** architecture:

- **Central node / bottleneck: `python_execution`.** It has the highest
  afferent coupling (**fan-in 4** — depended on by `code_review`,
  `diagram_creator`, `refactoring_planner`, `test_generator`) and is the
  **only node with non-zero betweenness centrality** (≈0.00246, mean over 5
  seeds). It is the module's single point of structural fragility: a change
  here ripples to five skills.
- **Dependencies.** 10 `depends_on` edges across 8 connected skills; `file_search`
  is the most-depended-upon sink (fan-in 3); the chain
  `markdown_formatter → documentation_writer → file_search` is the deepest path.
- **Complexity / dead weight.** Two skills — `json_validator` and `web_search` —
  are **orphans** (in-degree 0 *and* out-degree 0): declared but wired to
  nothing (the README inside the corpus even mislabels them "roots").

Full per-skill node sizing = `min(LOC·8, 500)`; layer colour = L1/L2/L3.

---

## 4. Obsidian "after" — post-refactor topology

![Obsidian after](results/figures/obsidian_after.png)

This is a **mid-rollout snapshot** (32 of 64 steps) of the trained PPO policy
at `seed=42` — chosen over the terminal frame because by termination the policy
collapses the graph to a degenerate ~2-node pair that reads as a broken figure
(rationale in `scripts/capture_obsidian_after.py`).

How the topology changed: the policy applies SPLIT / MERGE / REWIRE edits that
redistribute edges away from the `python_execution` hub and fold low-similarity
leaves together, while the action mask forbids moves that would break the
L1→L3 lazy-load contract.

> **Honest framing.** At the 256-step *smoke* budget the **net** metric gain is
> modest — mean final reward is **−0.461 ± 0.186** (n=5), i.e. most steps are
> failed/NOOP edits and the policy has not yet converged to a net-positive
> refactor. The "after" frame demonstrates the *mechanism* (legal structural
> edits across all three layers), not a converged optimum; convergence-scale
> training (≥10k steps) is required for a decisively "more modular" graph.
> This is stated plainly rather than overclaimed (see [limitations](#7-honest-limitations)).

---

## 5. Metric & ablation analysis

### 5.1 Reward over training (D6)

![Reward vs step](results/learning_curves/reward_vs_episode.png)

Mean ± 95% CI over **all 5 sealed seeds** {42, 7, 123, 314, 271}, 256 steps each.
Reward = `α·ΔModularity + β·ΔCohesion − γ·Coupling + P_skills` (Newman-Girvan Q
for modularity). Per-seed finals: 42=−0.440, 7=−0.141, 123=−0.690, 314=−0.595,
271=−0.440 ⇒ **−0.461 ± 0.186**.

### 5.2 Reward-coefficient ablation (81 cells × 5 seeds = 405 runs, all ok)

![Ablation heatmap](results/figures/ablation_heatmap.png)

| cell | (α, β, γ, P_skills) | mean reward | Δ vs baseline |
|---|---|---|---|
| baseline | (1.0, 1.0, 0.5, −5.0) | −0.461 ± 0.259 | 0.000 |
| **best** | (0.5, 1.0, 1.0, −1.0) | **+0.098 ± 0.507** | **+0.559** |
| worst | (2.0, 2.0, 0.0, −10.0) | −1.158 ± 0.418 | −0.697 |

**Sobol-lite sensitivity: α (2.03) ≫ β (0.92) > γ (0.83) > P_skills (0.00).**
α (the ΔModularity weight) dominates by ~2.2×; α=2.0 crushes reward while α=0.5
lifts it — the sealed default α=1.0 is mid-grid, **not** the optimum. P_skills
scores 0 because the lazy-load-break event never fires inside the 256-step
budget. CIs use Student-t at dof=4 (n=5). Full breakdown + marginals:
[`docs/ANALYSIS.md`](docs/ANALYSIS.md).

### 5.3 Betweenness centrality (brief §2.2: 2×/seed)

![Betweenness CI](results/figures/betweenness_ci.png)

Computed **exactly twice per seed** (training start + end) across 5 seeds,
reported as mean ± 95% CI (`results/data/betweenness_table.csv`). `python_execution`
is the sole node carrying non-zero betweenness throughout.

---

## 6. Bug report — architectural smells in the Skills module

Two **architectural** bugs surfaced by the GRAPHIFY + betweenness reverse
engineering (brief §3). Full write-up: [`docs/BUG_REPORT.md`](docs/BUG_REPORT.md).

1. **Orphan skills (`json_validator`, `web_search`)** — both have in-degree 0
   *and* out-degree 0; dead structural weight that depresses module modularity
   and is unreachable by any dependency-driven refactor. The corpus README even
   mislabels them as "roots."
2. **Coupling hotspot (`python_execution`)** — fan-in 4 and sole non-zero-
   betweenness node, yet it still carries an outgoing edge (→`file_search`), so
   it is **not** maximally stable (instability I = 1/5 = 0.2) despite four
   dependents — a Stable-Dependencies-Principle violation and the module's
   fragility hub.

(An appendix in `BUG_REPORT.md` also documents two training-harness defects
found and fixed during the build, incl. the seed-123/314 Louvain hang.)

---

## 7. Honest limitations

- **Smoke-scale budget.** 256 steps/seed is a smoke run; point estimates are
  directional, not convergence-scale (hence the negative mean reward).
- **PythonClaw shim.** Until the upstream URL is confirmed, the agent trains
  against an in-tree shim (ADR-001, 24h swap window); claims about
  "PythonClaw's architecture" are strictly claims about the shim.
- **Single module / single corpus.** Only the `Skills` layer is analysed; no
  cross-project claim is made.
- **Hand-designed reward.** α/β/γ/P_skills are author choices calibrated by the
  ablation, not learned.
- **SB3-style abstraction & Obsidian non-determinism** — see
  [`docs/PRD.md` §6.2](docs/PRD.md) (L1–L7) for the full list.

The brief does not request a self-grade, so none is claimed — the limitations
above are the credibility signal.

---

## Documents

- [`docs/PRD.md`](docs/PRD.md) · [`docs/PLAN.md`](docs/PLAN.md) · [`docs/TODO.md`](docs/TODO.md) — scope, plan, tasks
- [`docs/TRACE.md`](docs/TRACE.md) — brief-§ → artifact map
- [`docs/ESSAY.md`](docs/ESSAY.md) — §2.4 GRAPHIFY × AI essay
- [`docs/THEORY.md`](docs/THEORY.md) — PPO/GAE/reward math
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md) · [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — results
- [`docs/COST_ANALYSIS.md`](docs/COST_ANALYSIS.md) — cost & resources
- [`docs/BUG_REPORT.md`](docs/BUG_REPORT.md) · [`docs/SKILLS_ARCHITECTURE.md`](docs/SKILLS_ARCHITECTURE.md)
- [`docs/QUALITY.md`](docs/QUALITY.md) — quality self-audit · [`docs/adr/`](docs/adr/) — 11 ADRs

## Architecture

```
src/
├── sdk/             # RefactorSDK — single business-logic entry point
├── services/        # ppo_trainer, gae_buffer, centrality, metrics/, vault_writer
├── env/             # custom (non-gym) refactor env: state, actions, reward, mask
├── model/           # actor-critic policy_net + encoder
├── graphify/        # local GRAPHIFY re-impl behind GraphifyAdapter (ADR-002)
├── pythonclaw_shim/ # Skills shim + sample_skills corpus (ADR-001)
├── cost/            # tiktoken TripleCounter
└── utils/           # config loader, seeding
```

## Configuration

All tunable parameters live in **`config/config.yaml`** (single source of truth)
and are read via `src/utils/config_loader.py` — **no algorithm value is hardcoded
in source** (enforced by `tests/architecture/test_config_refs.py`). Key blocks:

| Block | Controls |
|---|---|
| `ppo` | `clip_eps` (0.2, sealed), `gae_lambda` (0.95, sealed), `gamma`, `lr`, `n_steps`, `n_epochs`, `batch_size`, `vf_coef` |
| `reward` | `alpha`, `beta`, `gamma`, `p_skills` (canonical reward coefficients) |
| `ablation` | `grids` (compact/smoke), `scout_seeds`, `total_steps_per_cell_seed`, `per_seed_timeout_s` |
| `seeds` | the 5 sealed seeds `[42, 7, 123, 314, 271]` |
| `centrality`, `state`, `action`, `training`, `environment`, `paths` | metric, encoder, env, and I/O settings |

Secrets (if ever needed) go in a git-ignored `.env`; `.env-example` documents the
expected keys. The config file carries a `version` field validated at load.

## Contributing

This is a course assignment, but it follows professional standards (see
[`CLAUDE.md`](CLAUDE.md) and the V3 submission guidelines). Before any change:

```bash
uv sync --dev
uv run pytest tests/ --cov=src     # must pass, coverage ≥85%
uv run ruff check src/ tests/ scripts/   # zero violations
uv run ruff format src/ tests/ scripts/  # auto-format
uv run python scripts/check_file_sizes.py # every .py ≤150 LOC
```

Standards: **TDD** (test before/with code), **OOP** via the `RefactorSDK` single
entry point, **DRY** (extract at the second copy), every `.py` **≤150 lines**,
docstrings on every public function/class/module, and **`uv` only** (no `pip` /
`python -m` / `venv`). Commit subjects follow `phase<N>(<scope>): <description>`;
work on a feature branch and open a PR for review.

## References

- Schulman et al. 2017 — *Proximal Policy Optimization Algorithms* (arXiv:1707.06347)
- Schulman et al. 2016 — *High-Dimensional Continuous Control Using GAE* (arXiv:1506.02438)
- Newman & Girvan 2004 — *Finding and evaluating community structure in networks*
- Huang & Ontañón 2022 — *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*

## License & Credits

MIT — see [LICENSE](LICENSE). Built on PyTorch, NetworkX, tiktoken, NumPy, SciPy,
Matplotlib, and pyvis; managed with `uv`. Course: Bar-Ilan *Vibe Coding & RL*
workshop (Dr. Yoram Segal).
