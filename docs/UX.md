# UX — Usability Surface & Nielsen Heuristics (§10)

> **Scope: CLI + static artifacts. No interactive GUI was built** (and none is
> required by the brief). Nielsen's 10 heuristics are written for interactive
> UIs, so most are **N/A** here; the rows below map the ones that *do* apply to
> the two real surfaces, and mark the rest N/A with a reason. This is the honest
> §10 position for a non-GUI tool — not a placeholder for an unshipped dashboard.

## Surfaces that actually exist

1. **CLI** — `python -m src.cli <graph|cost|info>` (`src/cli/__main__.py`):
   prints the GRAPHIFY graph summary, the tiktoken cost table, and project
   metadata. `--help` documents every subcommand; it imports only the SDK /
   thin readers (no business logic in the CLI).
2. **Static analysis artifacts** — committed PNG/CSV outputs a reader inspects
   directly: `results/figures/obsidian_{before,after}.png`,
   `metric_improvement_curves*.png`, `ablation_heatmap.png`, `betweenness_ci.png`,
   and `results/learning_curves/reward_vs_episode.png`. These are the "screens"
   the brief §3 asks for, rendered deterministically by `scripts/`.

There is **no Streamlit/Flask GUI and no live Obsidian plugin** — `src/gui/`
is an empty package placeholder, and there is no `streamlit` dependency.

## Nielsen's 10 heuristics → this project

| # | Heuristic | Applies here? |
|---|---|---|
| 1 | Visibility of system status | ✅ CLI prints progress + result tables; training scripts log per-seed `final_reward`/`betweenness_calls`. |
| 2 | Match between system & real world | ✅ Domain vocabulary throughout (modularity, cohesion, coupling, SPLIT/MERGE/REWIRE); figures are titled in those terms. |
| 3 | User control & freedom | ◑ Partial — CLI is non-destructive and re-runnable; no undo needed (read-only analysis). |
| 4 | Consistency & standards | ✅ Uniform argparse subcommands; figures share one colour map + legend convention. |
| 5 | Error prevention | ✅ argparse validates flags; SDK/dataclasses raise `ValueError`/`TypeError` on bad input (`Trajectory.__post_init__`). |
| 6 | Recognition over recall | ✅ `--help` lists subcommands + defaults; no memorised invocations required. |
| 7 | Flexibility & efficiency | ◑ Partial — flags expose source/seed/steps overrides for power users; sensible defaults otherwise. |
| 8 | Aesthetic & minimalist design | ✅ Figures are labelled, captioned, legended, high-DPI; CLI output is terse tables. |
| 9 | Recognize / diagnose / recover from errors | ✅ Errors surface as typed exceptions with messages; CI gates catch regressions pre-merge. |
| 10 | Help & documentation | ✅ `--help` + this README-as-report + `docs/` cover every surface. |
| — | Heuristics assuming a rich interactive GUI (animated transitions, modal dialogs, drag-and-drop) | **N/A** — no such surface exists. |

## Accessibility

CLI output is plain text (screen-reader friendly); figures always pair colour
with explicit titles, axis labels, and legends, so no information is encoded by
colour alone. No keyboard-navigation concerns — there is no GUI.
