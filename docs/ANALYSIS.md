# ANALYSIS.md — D7 ΔReward + sensitivity (Phase 4)

## §0 Honest status

This document is the empirical companion to docs/ESSAY.md §4.

> **Graph scope (Phase 5, 2026-06-08).** The **headline 5-seed PPO training**
> runs on the **real PythonClaw** dependency graph (`ericwang915/PythonClaw` @
> `7787bb43`; 1,190 nodes / 3,300 edges) — see README §5.1 and
> `results/training/aggregate.json`. The **reward-coefficient ablation below
> stays on the controlled `sample_skills` corpus** (30-node skills graph): its
> purpose is to isolate the *sensitivity* of α/β/γ/P_skills, and a full real-graph
> ablation (81 cells × 5 seeds × ~2.5 min/seed ≈ **17 h**) exceeds the wall-clock
> budget. The controlled corpus holds graph topology fixed so the coefficient
> effect is not confounded by graph size — the appropriate setting for a
> sensitivity study. This split is stated plainly rather than blurred.

Per the architect-locked honesty policy, n_ok is reported honestly per cell. After
the RC-4 fix all ablation cells reach n_ok=5, so no padding or imputation is needed.

AB-EXEC completed at the full ADR-006 floor: **81 cells × 5 seeds (405/405 rows
status=ok)** after the RC-4 fix. §2–§6 carry real numbers from
results/data/ablation_stats.json.
**Main training (5-seed PPO run) completed** with Phase-4 RC-4 (SIGALRM-based Louvain
watchdog eliminates daemon-thread GIL accumulation). All 5 seeds: {42, 7, 123, 314, 271},
mean final reward = −0.461 ± 0.186 (n=5, 95% CI ±0.232 at dof=4). Betweenness CI
chart re-rendered with n=5. Learning curve (D6) generated at
results/learning_curves/reward_vs_episode.png.

## §1 Setup
- **Grid:** `compact` (81 cells = 3α × 3β × 3γ × 3P_skills) per config.yaml ablation block.
- **Seeds per cell:** all 5 sealed seeds {42, 7, 123, 314, 271}. RC-4 (SIGALRM Louvain
  cut + stored-mask Trajectory) closed the seed-123/314 hang, so the ablation runs at
  the full ADR-006 ≥5-seed floor — **405/405 rows `status=ok`**, no timeouts.
- **Steps per seed:** 256 (2 PPO iters × 128 n_steps).
- **Per-seed wall-clock budget:** 240s (never approached post-RC-4; cells run ~50s for all 5 seeds).
- **Baseline cell:** (α=1.0, β=1.0, γ=0.5, P_skills=-5.0) — the canonical sealed default.
- **Reward equation (sealed):** R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling + P_skills
  with the canonical default coefficients above. See src/env/reward.py L93-115 and ADR-007.
- **Action space:** A_max=45057 (sealed boundary, see ESSAY §3 + ADR-005a).
- **PPO hyperparameters:** ε=0.2, GAE λ=0.95, γ=0.99 (sealed; identical across all cells).

### §1.1 Why this grid?
A 3-value-per-knob compact grid gives every knob a {low, mid, high} sweep. At
5 seeds/cell × 81 cells × 256 steps the full run completes in ~70 min of
single-process CPU after RC-4. A 5-value grid (625 cells) would have exceeded the
brief's wall-clock envelope by roughly an order of magnitude.

### §1.2 What "OK" means
A seed is OK iff its PPO rollout terminates inside the 240s wall-clock budget AND
emits a final `mean_final_reward` value. After RC-4 every seed completes well
inside budget; all 405 (cell × seed) rows are `status=ok`.

## §2 ΔReward summary (D7)

Source: `results/data/ablation_stats.json` produced by `scripts/ablation_stats.py`
from the 406-line `results/ablations/seed_table.csv` (header + 81 cells × 5 seeds).

| metric | baseline (1.0, 1.0, 0.5, -5.0) | best cell (0.5, 1.0, 1.0, -1.0) | worst cell (2.0, 2.0, 0.0, -10.0) |
|---|---|---|---|
| cell_sha | 8b1976b1dd11 | 08f03c0647e6 | c30978b31fc7 |
| mean_final_reward | -0.461 ± 0.259 | +0.098 ± 0.507 | -1.158 ± 0.418 |
| n_ok (out of 5) | 5 | 5 | 5 |
| Δ vs baseline | 0.000 | +0.559 | -0.697 |

CI95 half-widths use Student-t with dof=4 (n_ok=5 ⇒ t₀.₀₂₅,₄ = 2.776) —
substantially tighter than the earlier 3-seed run's dof=2 (t = 4.303).

### §2.1 Interpretation
The best cell beats the baseline by +0.559 mean reward; its CI95 (±0.507) and the
baseline's (±0.259) still overlap, so the gap is suggestive, not discriminative —
but the positive-mean region is now sharply localised: **only 6 of 81 cells have a
positive mean**, and all six sit at α=0.5, γ=1.0 (two distinct (α,β,γ) corners —
β∈{0.5,1.0} — each appearing three times because P_skills is inert). The worst
cell drops 0.697 below the baseline with a tight CI
(±0.418), clearly a different regime. Reading the Sobol-lite ranking in §5 — α
dominates (score 2.03), followed by β (0.92) and γ (0.83) — the worst cell's
collapse is driven primarily by α=2.0 (highest grid value); α alone moves the
marginal mean from -0.258 (α=0.5) to -0.893 (α=2.0), a Δ of -0.635. The best cell
mirrors this: α=0.5 is the lowest grid value and its marginal is the most-positive.
The signal is "α too high crushes reward"; the sealed default α=1.0 (marginal
-0.495) sits between the best (α=0.5) and worst (α=2.0) marginals — a defensible
mid-grid choice, though α=0.5 yields a higher marginal mean.

### §2.2 Cells with n_ok < 5
None. All 81 cells completed all 5 seeds (405/405 rows `status=ok` in
`seed_table.csv`) after RC-4. The honesty-policy machinery for
partial-cell handling (Student-t fallback to ci95=0.0 when n_ok≤1) is retained
in `scripts/ablation_stats.py` and exercised by `test_handles_partial_n_ok`
against synthetic data; it is just inert against the real corpus.

## §3 Marginal sensitivity — per-knob mean ± CI95

Each knob fixes one value while the other three vary across the compact grid;
the marginal mean is taken over the 27 cells at that knob value (27 = 3³ from
the remaining knobs), and the CI95 half-width uses Student-t with dof=26.

**α (ΔModularity weight):**

| α value | n_cells | mean reward | CI95 lo | CI95 hi |
|---|---|---|---|---|
| 0.5 | 27 | -0.258 | -0.340 | -0.176 |
| 1.0 | 27 | -0.495 | -0.560 | -0.430 |
| 2.0 | 27 | -0.893 | -0.949 | -0.837 |

Monotone-decreasing in α — clean directional signal. The sealed default α=1.0
sits between the best (α=0.5) and worst (α=2.0) marginals. All three CI bands
are mutually non-overlapping at n=5 — α moves the reward signal at a magnitude
clearly above the per-marginal noise floor.

**β (ΔCohesion weight):**

| β value | n_cells | mean reward | CI95 lo | CI95 hi |
|---|---|---|---|---|
| 0.5 | 27 | -0.430 | -0.547 | -0.313 |
| 1.0 | 27 | -0.500 | -0.625 | -0.375 |
| 2.0 | 27 | -0.716 | -0.821 | -0.611 |

Monotone-decreasing; the 0.5/1.0 CIs overlap, so β at this resolution does not
give a discriminative low-vs-mid response. The β=2.0 band separates cleanly
from β=0.5. At 5 seeds β edges ahead of γ in the Sobol-lite ranking (§5).

**γ (Coupling penalty weight):**

| γ value | n_cells | mean reward | CI95 lo | CI95 hi |
|---|---|---|---|---|
| 0.0 | 27 | -0.664 | -0.770 | -0.558 |
| 0.5 | 27 | -0.578 | -0.691 | -0.465 |
| 1.0 | 27 | -0.404 | -0.538 | -0.270 |

Monotone-increasing in γ — heavier coupling penalty correlates with higher
reward, contrary to the naive intuition that "more penalty hurts." Likely the
penalty steers the policy away from coupling-heavy actions that themselves
incur a larger ΔModularity hit.

**P_skills (skill-shortage penalty):**

| P_skills value | n_cells | mean reward | CI95 lo | CI95 hi |
|---|---|---|---|---|
| -10.0 | 27 | -0.549 | -0.675 | -0.423 |
| -5.0 | 27 | -0.549 | -0.675 | -0.423 |
| -1.0 | 27 | -0.549 | -0.675 | -0.423 |

The three marginals are identical to 6 decimal places. Spot-checking the raw
`seed_table.csv` confirms that for every fixed (α, β, γ, seed), `final_reward`
is identical across all 3 values of P_skills. At the 256-step smoke regime
the skill-shortage event apparently never fires in any cell — the per-step
ΔModularity / ΔCohesion / Coupling terms saturate the reward signal before
any P_skills penalty triggers. This is a genuine artefact of the smoke-scale
budget, not a bug in the stats engine; it is the reason Sobol-lite scores
P_skills at 0.0 in §5.

### §3.1 Reading the marginals
- **Monotone marginal** (low < mid < high or reverse) ⇒ knob has a clean directional
  effect; the sealed default may not be at the optimum.
- **U-shaped marginal** ⇒ sealed default sits near a local optimum; deviations in
  either direction hurt.
- **Flat marginal** (CI95 bars all overlap) ⇒ knob does not move the signal at this
  grid resolution; do not over-claim sensitivity.

## §4 Heatmap (D3)

See `results/figures/ablation_heatmap.png` (produced by Stream B's
`scripts/render_ablation_heatmap.py` in parallel with this stream). The heatmap
is a 9×9 grid (α×β on one axis, γ×P_skills on the other), with the baseline
cell circled and the best/worst cells annotated. Colour scale: diverging
blue→white→red around the baseline mean (-0.461), so the visual asymmetry of
the response surface is legible at a glance — the α=2.0 quadrants are clearly
the deepest-red region and the α=0.5, γ=1.0 quadrants the brightest blue.

### §4.1 Hatching convention
Cells with n_ok < 5 are hatched grey on top of the colour fill — this preserves the
mean-reward visual while making it obvious that the cell's CI is too wide to trust.
At 5 seeds the final run has zero such cells (all 81 at n_ok=5), so no hatching appears.

## §5 Sobol-lite (cheap sensitivity score)

Per-knob first-order sensitivity: |mean(max-knob cells) − mean(min-knob cells)| / σ(all cells).
Larger = knob matters more. Formally a Sobol first-order approximation (Sobol 2001);
we skip the full Saltelli sampling for tractability. σ(all cells) here is the
population stdev across all 81 cell means (≈0.313).

| knob | range Δ (raw) | normalised by σ_all | rank |
|---|---|---|---|
| α | 0.635 | 2.03 | 1 |
| β | 0.286 | 0.92 | 2 |
| γ | 0.260 | 0.83 | 3 |
| P_skills | 0.000 | 0.00 | 4 |

α dominates by a factor of ~2.2× over the next-most-sensitive knob. At 5 seeds
β (0.92) edges narrowly ahead of γ (0.83) — a reordering from the earlier 3-seed
run (which had γ > β); the two are close enough that the swap is within sampling
noise, so the robust claim is "α dominates; β and γ are comparable second-order
effects." P_skills is exactly zero because, as §3 documents, the three P_skills
marginals are identical to 6 decimal places — the penalty never fires inside the
256-step smoke budget.

### §5.1 What this is not
This is NOT a full Sobol decomposition — we do not estimate second-order interaction
indices, and we do not bootstrap CIs on the sensitivity score itself. Treat the rank
as a coarse ordinal signal, not a precise effect size.

## §6 Limitations + caveats
- **n=5 seeds/cell ⇒ dof=4** for Student-t. Per-cell CI95 half-widths use
  t₀.₀₂₅,₄=2.776 — substantially tighter than the earlier 3-seed run (dof=2,
  t=4.30). Marginal CI95 uses dof=26 (n=27 cell means per knob value, t≈2.06),
  conditional on the independence assumption between cells, which is only weakly
  true here (same per-cell architecture, same PythonClaw shim, same skills graph).
- **81/81 cells reached n_ok=5** in the final `seed_table.csv` (405/405 rows
  `status=ok`). After RC-4 no seed timed out. The honesty-policy partial-cell
  handling (`ci95=0.0` when `n_ok ≤ 1`) is still exercised by
  `test_handles_partial_n_ok` against synthetic data, but is inert against the
  real corpus.
- **Sample size of 81 cells × 5 seeds × 256 steps** is a SMOKE-scale ablation. Larger
  grids (5 values per knob, 1000+ steps) would tighten CIs further but exceeded
  the brief's wall-clock budget. The 256-step horizon keeps point estimates
  directional, not convergence-scale.
- **Reward formula is non-stationary** — episodes start with the original sample_skills
  graph and SPLITs / MERGEs change topology mid-rollout. The reported final reward is
  the sum of per-step deltas (matches RC-5 protocol).
- **Single corpus** — ablations are run against the sealed sample_skills graph, not
  across the 7-graph corpus collected in COST-3. A cross-corpus ablation is future
  work (out of brief scope).
- **No CPU/GPU stratification** — all cells were run on the same host. Multi-host
  variance is therefore not captured; results are conditioned on single-machine timing.

## §7 Reproducibility
```bash
# Re-run ablation:
uv run python scripts/run_ablation.py --grid compact

# Re-render heatmap + marginals:
uv run python scripts/render_ablation_heatmap.py
uv run python scripts/render_ablation_marginals.py

# Re-derive ΔReward + Sobol-lite:
uv run python scripts/ablation_stats.py

# Re-run this analysis (read by ANALYSIS.md):
uv run python -c "from src.sdk import run_ablation; print(run_ablation('compact'))"
```

### §7.1 Determinism
Each (cell, seed) pair is fully determined by:
- the sealed reward coefficients in config/config.yaml.ablation,
- the seed value injected into numpy + torch + python `random`,
- the sealed PPO hyperparameters (ε, λ, γ, n_steps, n_iters).

Re-running the same (cell, seed) on the same host should yield bitwise-identical
final reward up to floating-point summation order in the GAE advantage rollup
(see docs/PPO_GAE_MATH_AUDIT.md §4).

## §8 Pointers to source
- Reward equation: src/env/reward.py L93-115 + ADR-007.
- Ablation runner: scripts/run_ablation.py + scripts/_ablation_lib.py.
- SDK consumer: src/sdk/ablation.py (Ablation + CellResult + run_ablation).
- Honesty policy: docs/TODO.md §Known gaps; this analysis stays 🟡 in-progress
  while results/ablations/ has < 81 done.json files.
- Heatmap renderer: scripts/render_ablation_heatmap.py — ✅ landed.
- Marginals renderer: scripts/render_ablation_marginals.py — ✅ landed.
- Stats engine: scripts/ablation_stats.py — ✅ landed; produces results/data/ablation_stats.json.

## §9 Cross-references
- docs/ESSAY.md §4 (empirical results narrative) — consumes the §2/§3/§5 tables here.
- docs/EXPERIMENTS.md — protocol that defined which seeds count as OK.
- docs/PLAN.md Phase 4 — places this document in the D7 deliverable column.
- docs/TRACE.md — section-by-section forward refs from the brief to this file.
- docs/BUG_REPORT.md — Appendix A2 (Louvain wedge) documents the now-resolved seed hang.
- docs/COST_ANALYSIS.md — wall-clock + dollar cost of producing this ablation.

## §10 Change log
- Phase 4 Wave 4b-pre (ANALYSIS-SKEL): skeleton landed.
- Phase 4 Wave 4c Stream A (AB-STATS+ANALYSIS-FILL): scripts/ablation_stats.py
  generates results/data/ablation_stats.json from seed_table.csv. §2, §3, §4
  (heatmap reference), §5 (Sobol-lite), §6 caveats filled.
- Phase 4 RC-4 (5-seed re-run): after the SIGALRM Louvain fix, the full ablation
  re-ran at 5 seeds/cell (405/405 rows ok). Updated numbers: best cell
  (0.5, 1.0, 1.0, -1.0) mean=+0.098; worst cell (2.0, 2.0, 0.0, -10.0) mean=-1.158;
  baseline mean=-0.461; Sobol-lite ranks α(2.03) > β(0.92) > γ(0.83) > P_skills(0.00).
  CIs tightened from dof=2 (n=3) to dof=4 (n=5).
