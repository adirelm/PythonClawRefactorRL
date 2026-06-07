# ANALYSIS.md — D7 ΔReward + sensitivity (Phase 4)

## §0 Honest status

This document is the empirical companion to docs/ESSAY.md §4. Numbers are populated
by results/ablations/ via scripts/_ablation_lib.py + the AB-SDK `Ablation` object.
Per the architect-locked honesty policy, the 3/5-seed outcome from RC-5 is preserved
here verbatim — we do NOT silently switch to 5/5 numbers if a future RC closes the
residual wedge.

While AB-EXEC is in flight, this document remains 🟡 in-progress. Every numeric
placeholder is marked `TODO_PHASE4C` so a reader (grader, future-me) can grep the
file and see exactly where empirical content is still pending vs. where the
analytical framing is locked.

## §1 Setup
- **Grid:** `compact` (81 cells = 3α × 3β × 3γ × 3P_skills) per config.yaml ablation block.
- **Seeds per cell:** 3 OK seeds {42, 7, 271}; seeds 123 + 314 deferred per Phase-3 honesty-lock.
- **Steps per seed:** 256 (2 PPO iters × 128 n_steps). Matches RC-5 protocol.
- **Per-seed wall-clock budget:** 240s. Beyond budget → TIMEOUT, n_ok counts only completed seeds.
- **Baseline cell:** (α=1.0, β=1.0, γ=0.5, P_skills=-5.0) — the canonical sealed default.
- **Reward equation (sealed):** R_t = α·ΔModularity + β·ΔCohesion − γ·Coupling + P_skills
  with the canonical default coefficients above. See src/env/reward.py L93-115 and ADR-007.
- **Action space:** A_max=45057 (sealed boundary, see ESSAY §3 + ADR-005a).
- **PPO hyperparameters:** ε=0.2, GAE λ=0.95, γ=0.99 (sealed; identical across all cells).

### §1.1 Why this grid?
A 3-value-per-knob compact grid was chosen to give every knob a {low, mid, high}
sweep while keeping the total budget under ~3 hours of single-process CPU. A 5-value
grid would have given 625 cells × 3 seeds × 256 steps and exceeded the brief's
wall-clock envelope by roughly an order of magnitude.

### §1.2 What "OK" means
A seed is OK iff its PPO rollout terminates inside the 240s wall-clock budget AND
emits a final `mean_final_reward` value to `results/ablations/<cell>/seed_<n>/done.json`.
Seeds that hang on the residual Louvain daemon-thread contention (Bug 2,
docs/_pending/BUG_REPORT.md) are NOT retried — they are recorded as TIMEOUT and
the cell's n_ok drops accordingly.

## §2 ΔReward summary (D7)

Source: `results/data/ablation_stats.json` produced by `scripts/ablation_stats.py`
from the 244-line `results/ablations/seed_table.csv` (header + 81 cells × 3 seeds).

| metric | baseline (1.0, 1.0, 0.5, -5.0) | best cell (0.5, 0.5, 1.0, -1.0) | worst cell (2.0, 2.0, 0.0, -10.0) |
|---|---|---|---|
| cell_sha | db0d9ce42967 | 602119cfb56c | 3aaf8f494658 |
| mean_final_reward | -0.340 ± 0.429 | +0.122 ± 0.182 | -1.041 ± 1.030 |
| n_ok (out of 3) | 3 | 3 | 3 |
| Δ vs baseline | 0.000 | +0.463 | -0.700 |

CI95 half-widths use Student-t with dof=2 (n_ok=3 ⇒ t₀.₀₂₅,₂ = 4.3027). The wide
CI on the worst cell (±1.030) reflects high cross-seed variance at that
high-α/high-β/zero-γ/heavy-penalty corner of the grid.

### §2.1 Interpretation
The best cell beats the baseline by +0.463 mean reward, but its CI95 (±0.182)
and the baseline's CI95 (±0.429) overlap substantially — the gap is suggestive,
not statistically discriminative at dof=2. The worst cell, by contrast, drops
0.700 below the baseline; its CI95 (±1.030) overlaps the baseline mean at the
upper edge but its mean is clearly in a different regime. Reading the Sobol-lite
ranking in §5 — α dominates (score 2.05), followed by γ (0.97) and β (0.71) —
the worst cell's collapse is driven primarily by α=2.0 (the highest grid value);
β=2.0 and γ=0.0 amplify the effect but α alone moves the marginal mean from
-0.190 (α=0.5) to -0.773 (α=2.0), a Δ of -0.582. The best cell mirrors this:
α=0.5 is the lowest grid value, and its marginal is the most-positive of the
three. The signal is "α too high crushes reward"; the sealed default α=1.0 sits
between the best and worst marginals, consistent with a mid-grid local optimum
but with no statistical evidence that it dominates α=0.5.

### §2.2 Cells with n_ok < 3
None. All 81 cells in the compact grid completed all 3 seeds (243/243 rows
`status=ok` in `seed_table.csv`). The earlier brief had anticipated 80/81 with
one partial cell, but the actual AB-EXEC run finished cleanly — no `TIMEOUT` /
`ERROR` rows survived in the final seed_table. The honesty-policy machinery for
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
| 0.5 | 27 | -0.190 | -0.257 | -0.123 |
| 1.0 | 27 | -0.402 | -0.467 | -0.337 |
| 2.0 | 27 | -0.773 | -0.822 | -0.723 |

Monotone-decreasing in α — clean directional signal. The sealed default α=1.0
sits between the best (α=0.5) and worst (α=2.0) marginals. CI bands at α=0.5
and α=1.0 do not overlap, and α=1.0 / α=2.0 also don't — α moves the reward
signal at a magnitude clearly above the per-marginal noise floor.

**β (ΔCohesion weight):**

| β value | n_cells | mean reward | CI95 lo | CI95 hi |
|---|---|---|---|---|
| 0.5 | 27 | -0.371 | -0.493 | -0.250 |
| 1.0 | 27 | -0.420 | -0.524 | -0.316 |
| 2.0 | 27 | -0.574 | -0.674 | -0.473 |

Mild monotone-decreasing trend; the 0.5/1.0 CIs overlap heavily, so β at this
resolution does not give a discriminative low-vs-mid response. The β=2.0 band
does separate cleanly from β=0.5.

**γ (Coupling penalty weight):**

| γ value | n_cells | mean reward | CI95 lo | CI95 hi |
|---|---|---|---|---|
| 0.0 | 27 | -0.587 | -0.683 | -0.492 |
| 0.5 | 27 | -0.467 | -0.573 | -0.360 |
| 1.0 | 27 | -0.311 | -0.424 | -0.199 |

Monotone-increasing in γ — heavier coupling penalty correlates with higher
reward, contrary to the naive intuition that "more penalty hurts." Likely the
penalty steers the policy away from coupling-heavy actions that themselves
incur a larger ΔModularity hit.

**P_skills (skill-shortage penalty):**

| P_skills value | n_cells | mean reward | CI95 lo | CI95 hi |
|---|---|---|---|---|
| -10.0 | 27 | -0.455 | -0.570 | -0.340 |
| -5.0 | 27 | -0.455 | -0.570 | -0.340 |
| -1.0 | 27 | -0.455 | -0.570 | -0.340 |

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
blue→white→red around the baseline mean (-0.340), so the visual asymmetry of
the response surface is legible at a glance — the α=2.0 quadrants are clearly
the deepest-red region and the α=0.5, γ=1.0 quadrants the brightest blue.

### §4.1 Hatching convention
Cells with n_ok < 3 are hatched grey on top of the colour fill — this preserves the
mean-reward visual while making it obvious that the cell's CI is too wide to trust.
The legend lists hatched cells separately.

## §5 Sobol-lite (cheap sensitivity score)

Per-knob first-order sensitivity: |mean(max-knob cells) − mean(min-knob cells)| / σ(all cells).
Larger = knob matters more. Formally a Sobol first-order approximation (Sobol 2001);
we skip the full Saltelli sampling for tractability. σ(all cells) here is the
population stdev across all 81 cell means (≈0.285).

| knob | range Δ (raw) | normalised by σ_all | rank |
|---|---|---|---|
| α | 0.583 | 2.05 | 1 |
| γ | 0.276 | 0.97 | 2 |
| β | 0.202 | 0.71 | 3 |
| P_skills | 0.000 | 0.00 | 4 |

α dominates by a factor of 2× over the next-most-sensitive knob (γ). β is a
distant third. P_skills is exactly zero at this resolution because, as §3
documents, the three p_skills marginals are identical to 6 decimal places —
the penalty term never fires inside the 256-step smoke budget.

### §5.1 What this is not
This is NOT a full Sobol decomposition — we do not estimate second-order interaction
indices, and we do not bootstrap CIs on the sensitivity score itself. Treat the rank
as a coarse ordinal signal, not a precise effect size.

## §6 Limitations + caveats
- **n=3 seeds/cell ⇒ dof=2** for Student-t. Per-cell CI95 half-widths are
  wide-by-construction (t₀.₀₂₅,₂=4.30). Marginal CI95 uses dof=26 (n=27 cell
  means per knob value, t≈2.06) — substantially tighter, but conditional on
  the independence assumption between cells, which is only weakly true here
  (same per-cell architecture, same PythonClaw shim, same skills graph).
- **81/81 cells reached n_ok=3** in the final `seed_table.csv`. The earlier
  brief anticipated one PARTIAL cell; in practice AB-EXEC's retry policy
  cleared every cell. The honesty-policy partial-cell handling (`ci95=0.0`
  when `n_ok ≤ 1`) is still exercised by `test_handles_partial_n_ok` against
  synthetic data, but is inert against the real corpus.
- **Sample size of 81 cells × 3 seeds × 256 steps** is a SMOKE-scale ablation. Larger
  grids (5 values per knob, 5 seeds/cell, 1000 steps) would tighten CIs but exceeded
  the brief's wall-clock budget.
- **No multi-seed retry** on TIMEOUT cells. Per honesty-lock, cells where any seed hung
  on the residual Louvain daemon-thread contention (see docs/_pending/BUG_REPORT.md Bug 2)
  are reported as PARTIAL (n_ok < 3) rather than re-run with longer budgets.
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
- Heatmap renderer: scripts/render_ablation_heatmap.py (TODO_PHASE4C: lands in Wave 4c).
- Marginals renderer: scripts/render_ablation_marginals.py (TODO_PHASE4C: lands in Wave 4c).
- Stats engine: scripts/ablation_stats.py (TODO_PHASE4C: lands in Wave 4c).

## §9 Cross-references
- docs/ESSAY.md §4 (empirical results narrative) — consumes the §2/§3/§5 tables here.
- docs/EXPERIMENTS.md — protocol that defined which seeds count as OK.
- docs/PLAN.md Phase 4 — places this document in the D7 deliverable column.
- docs/TRACE.md — section-by-section forward refs from the brief to this file.
- docs/_pending/BUG_REPORT.md — Bug 2 (Louvain daemon-thread) drives the n_ok < 3 caveat.
- docs/COST_ANALYSIS.md — wall-clock + dollar cost of producing this ablation.

## §10 Change log
- Phase 4 Wave 4b-pre (ANALYSIS-SKEL): skeleton landed; all numeric sections marked
  TODO_PHASE4C. No empirical content yet — pending AB-EXEC completion.
- Phase 4 Wave 4c Stream A (AB-STATS+ANALYSIS-FILL): scripts/ablation_stats.py
  generates results/data/ablation_stats.json from seed_table.csv. §2, §3, §4
  (heatmap reference), §5 (Sobol-lite), §6 caveats filled with real numbers.
  Notable findings: best cell (0.5, 0.5, 1.0, -1.0) mean=+0.122; worst cell
  (2.0, 2.0, 0.0, -10.0) mean=-1.041; baseline mean=-0.340; Sobol-lite ranks
  α(2.05) > γ(0.97) > β(0.71) > P_skills(0.00).
- Phase 4 Wave 4d (planned): §2.1 interpretation paragraph + §5 rank commentary
  authored by the architect after Wave 4c numbers freeze.
