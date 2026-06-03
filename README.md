# PythonClawRefactorRL

Bar-Ilan University — *Vibe Coding & Reinforcement Learning* workshop,
Assignment 4. Applies **PPO + GAE** to the **PythonClaw Skills graph** to
learn a refactoring policy that reduces module coupling and raises
modularity / cohesion under a hard lazy-load-break penalty.

> Status: **bootstrap (Phase 0)** — skeleton, config, and CI only. No
> trainer, no evaluation results yet. Subsequent phases land the PRD, ADRs,
> PLAN, TODO, TRACE, and the actual implementation.

## Quick Start

```bash
uv sync --dev
uv run pytest tests/ --cov=src --cov-report=term-missing
uv run ruff check src/ tests/ scripts/
uv run python scripts/check_file_sizes.py
```

A CLI entry point will land with the training loop:

```bash
# placeholder — wired in Phase 2+
uv run python -m src.cli --help
```

## Status

CI status badge slot (filled once the repo is pushed to GitHub):

```
![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)
```

## Brief Mapping

The ex04 brief sections map onto this repo as follows (full mapping lands in
`docs/TRACE.md` during Phase 2):

| ex04 § | Deliverable | Location |
|---|---|---|
| §2.1 Skills-only scope | Graph builder + scope-lock test | `src/graphify/`, `tests/architecture/` |
| §2.2 Custom training loop | PPO trainer (not gym.Env) | `src/services/ppo_trainer.py` |
| §2.3 PPO ε=0.2, GAE λ=0.95 | Fixed-constants assertion test | `tests/architecture/test_ppo_constants.py` |
| §2.3 Reward α/β/γ + P_skills | Reward modules + ablation matrix | `src/services/reward.py`, `results/ablation/` |
| §2.4 Theory essay | 2 500–3 000 words, 4 sections | `docs/THEORY.md` |
| §2.5 Statistical reporting | ≥ 5 seeds, mean ± std + 95 % CI | `results/figures/`, `results/ablation/` |

## Documents

- [`docs/PRD.md`](docs/PRD.md) — product / scope contract
- [`docs/PLAN.md`](docs/PLAN.md) — phased build plan
- [`docs/TODO.md`](docs/TODO.md) — live task list
- [`docs/TRACE.md`](docs/TRACE.md) — brief-§ → deliverable mapping
- [`docs/QUALITY.md`](docs/QUALITY.md) — quality gates + self-audit log
- [`docs/THEORY.md`](docs/THEORY.md) — §2.4 essay with citations
- [`docs/adr/`](docs/adr/) — architecture decision records
  (ADR-001 PythonClaw shim · ADR-002 GraphifyAdapter · ADR-003 SB3 timebox)

## Architecture (preview)

```
src/
├── sdk/              # RefactorSDK — single business-logic entry point
├── services/         # PPO trainer, reward, evaluation
├── env/              # Custom (non-gym) refactor environment
├── model/            # Actor-critic network
├── utils/            # Seeding, config loader, logging
├── gui/              # (Phase 9) optional dashboard
├── cli/              # Command-line interface
├── data/             # PythonClaw-derived Skills graph snapshots
├── graphify/         # Local re-impl behind GraphifyAdapter (ADR-002)
└── pythonclaw_shim/  # Shim behind ADR-001 (24h swap window)
```

## References

- Schulman et al. 2017 — *Proximal Policy Optimization Algorithms*
- Schulman et al. 2015 — *High-Dimensional Continuous Control Using GAE*
- Newman 2006 — *Modularity and community structure in networks*
- ex04 brief PDF (gitignored, local-only under `instructions/assignment-4/`)

## Honest Limitations

- The PythonClaw shim lives behind ADR-001 with a 24h swap window — once the
  upstream library lands, the shim must be replaced and the test diff
  reported in `docs/TRACE.md`.
- Results generalise to **one** codebase (the PythonClaw Skills layer);
  no cross-project claim is made.
- The lazy-load-break detector walks `sys.modules` and asserts a P95 token
  budget; it cannot catch every runtime regression, only the ones that
  surface in the import graph.

## License

MIT — see [LICENSE](LICENSE).
