# ADR-006 — Multi-Seed Evaluation Discipline

- **Status:** Accepted
- **Date:** 2026-06-04
- **Supersedes:** none
- **Related:** ADR-003 (tiktoken cost metric), ADR-010 (dual-criterion convergence),
  OQ-4 (statistical-power open question), A3 retrospective,
  CLAUDE.md §CANONICAL "Betweenness call discipline" (twice per seed)

## Context

Assignment 3 (WorkoutRecommenderA2C) reported headline numbers averaged over
**three** seeds. The lecturer's feedback and our own A3 retrospective both
landed on the same conclusion: three seeds produce **directional** results
only — pairwise Welch t-tests almost never cleared p < 0.05, and the rolling
mean was visibly dominated by one outlier seed in two of the four ablation
cells. We do not want to repeat that in A4.

OQ-4 in `docs/open_questions.md` framed the same concern formally: with the
variance we observed across A3 ablation cells (σ ≈ 0.08 normalised return),
a power analysis at α = 0.05, 1-β = 0.8 needs **n ≥ 5** to detect the
effect sizes our ablation matrix is designed to surface (Cohen's d ≈ 1.3).
Three seeds gave us 1-β ≈ 0.45 — i.e. we were more likely to miss a real
effect than to find it.

## Decision

**Every headline number in A4 reports `mean ± std` and a 95% confidence
interval, computed over a fixed seed set of size ≥ 5.**

The canonical seed set is **five** seeds, frozen and cited verbatim:

```
SEEDS = (42, 7, 123, 314, 271)   # |SEEDS| = 5 — matches CANONICAL
```

These five seeds are frozen in `config/config.yaml#eval.seeds` (single
source of truth per CLAUDE.md §4 — there is no separate `config/eval.yaml`)
and are the **only** seeds permitted for any number that appears in a
chart, a table, the §2.4 essay, or the README results section.
Exploratory runs during development may use any seed; they just cannot
be cited.

Concretely, this means:

1. **Charts** (training curves, ablation bars, betweenness plots) draw the
   per-seed mean as a solid line and the 95% CI as a shaded band.
2. **Tables** report `μ ± σ` with the CI in a parenthetical column.
3. **Pairwise comparisons** between ablation cells use **Welch's t-test**
   (unequal variances) and report the p-value alongside the effect size
   (Cohen's d). Five seeds give us enough degrees of freedom that Welch
   is well-defined; three did not.
4. **Betweenness centrality** runs **exactly twice per seed** — once on
   the initial training graph and once on the final training graph
   (per CLAUDE.md §CANONICAL "Betweenness call discipline" and this
   ADR's cross-reference rule). The reported number aggregates across
   the five canonical seeds as `mean ± std + 95% CI` for **both**
   endpoints **and** their Δ — never a single-seed snapshot, and never
   more or fewer than 2 calls per seed. The call count itself is
   enforced by `tests/architecture/test_betweenness_call_count.py`
   asserting exactly 2 invocations per seed.

## Consequences

**Positive.** Headline claims in the §2.4 essay become defensible: when we
say "removing the skill prior degrades return by 18%", that claim is
backed by n = 5, a p-value, and an effect size. The A3 failure mode
("looks like a trend, p = 0.21") is structurally prevented.

**Negative — compute.** Five seeds × the full ablation matrix (2³ = 8
cells) × convergence budget (~200k steps) ≈ 8 GPU-hours on the lab
machine. We accept this; the §2.4 essay grade is downstream of how
trustworthy the numbers look.

**Negative — failure modes.** If even one seed in the canonical set
diverges or NaN-crashes, we **do not** silently drop it. The eval
script raises and the run is re-launched with a logged justification
in `results/eval/seed_failures.md`.

## Enforcement

`tests/architecture/test_seeds_discipline.py` enforces this ADR. It
walks every artefact under `results/` and asserts the following:

**Tabular artefacts** (`.csv`, `.json`, `.png.meta.json`):

- `len(set(rows["seed"])) >= 5` for any charted result.
- The seed set is a subset of `SEEDS` (no off-canon seeds in headline
  numbers).
- A `ci_low` and `ci_high` column exists for any row marked
  `is_headline = True`.

**Charted artefacts (concrete assertion pattern).** Any
`results/figures/*.png` referenced in `docs/EXPERIMENTS.md` or
`docs/ANALYSIS.md` MUST have an accompanying sidecar
`results/<chart>_seeded.json` (same stem as the PNG) whose contents
load into a NumPy array of shape `(seeds, episodes)`, where:

- `arr.ndim == 2`
- `arr.shape[0] >= 5`   (≥ 5 seeds — matches `|SEEDS|`)
- the seed labels listed in the sidecar header are a subset of `SEEDS`

The test discovers PNG references by parsing the two markdown files for
`results/figures/*.png` link targets, then for each target asserts the
sidecar exists at the matching `_seeded.json` path and that
`np.asarray(json.load(...)["data"]).shape[0] >= 5`. A PNG that has no
sidecar — or whose sidecar has < 5 seeds — fails the suite with a
diagnostic naming the offending chart. This closes the loophole where
a chart could be cited in the essay without a backing per-seed array.

**Betweenness call discipline (cross-ref).**
`tests/architecture/test_betweenness_call_count.py` asserts the
twice-per-seed rule from CLAUDE.md §CANONICAL — exactly 2 NetworkX
betweenness invocations per seed (initial + final graph). Both this
test and `test_seeds_discipline.py` MUST pass for the multi-seed
discipline to be considered enforced end-to-end.

The tests are part of the `pytest tests/architecture/` quality gate and
run in CI on every push.

## Alternatives considered

- **n = 3 with bootstrap CIs.** Cheaper, but the bootstrap CI on n = 3
  is a known statistical anti-pattern — the resamples are not
  independent enough to widen the interval honestly. Rejected.
- **n = 10.** More statistical power, but 2× the compute for marginal
  gains past n = 5 in the variance regime we measured. Rejected as
  over-spend against the §1.4 cost-budget envelope.
- **Variable n per cell.** Tempting (spend more seeds where variance is
  high), but breaks the "every headline is comparable" property and
  complicates the Welch test bookkeeping. Rejected.
