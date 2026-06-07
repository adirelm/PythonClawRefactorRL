#!/usr/bin/env -S uv run python
"""Compute ablation statistics from results/ablations/seed_table.csv (Wave 4c D7).

Reads a CSV with schema
``cell_sha,alpha,beta,gamma,p_skills,seed,final_reward,status,elapsed_s``
and writes a JSON summary with per-cell aggregates, per-knob marginals, and
Sobol-lite first-order sensitivity scores. No pandas — stdlib ``csv`` only,
to stay inside the 150-LOC file-size cap (CLAUDE.md §1).

CLI: ``python scripts/ablation_stats.py [--input PATH] [--output PATH]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from scipy import stats as sp_stats  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "ablations" / "seed_table.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "data" / "ablation_stats.json"
BASELINE = (1.0, 1.0, 0.5, -5.0)  # canonical sealed default per CLAUDE.md
KNOBS = ("alpha", "beta", "gamma", "p_skills")
MIN_FULL_N_OK = 3  # seeds-per-cell threshold for "full" coverage


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _cell_key(r: dict[str, str]) -> tuple[str, float, float, float, float]:
    return (r["cell_sha"], float(r["alpha"]), float(r["beta"]), float(r["gamma"]), float(r["p_skills"]))


def _ci95(vals: list[float]) -> float:
    """Student-t 95% half-width; returns 0.0 when n<=1 (no CI definable).

    Uses dof = n-1 via scipy.stats.t.ppf, so per-cell aggregates (dof=2) and
    27-cell marginal aggregates (dof=26) both get the correct critical value.
    """
    n = len(vals)
    if n <= 1:
        return 0.0
    t = float(sp_stats.t.ppf(0.975, n - 1))
    return t * statistics.stdev(vals) / math.sqrt(n)


def _per_cell_stats(rows: list[dict[str, str]]) -> dict[tuple, dict]:
    """Group rows by (sha, knobs) and aggregate ok-status rewards."""
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            by_cell[_cell_key(r)].append(float(r["final_reward"]))
    out: dict[tuple, dict] = {}
    for key, vals in by_cell.items():
        out[key] = {
            "sha": key[0],
            "alpha": key[1],
            "beta": key[2],
            "gamma": key[3],
            "p_skills": key[4],
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "ci95": _ci95(vals),
            "n_ok": len(vals),
        }
    return out


def _marginal(cells: Iterable[dict], knob: str) -> list[dict]:
    """Per-knob-value mean ± CI95 across all cells (others varying)."""
    by_val: dict[float, list[float]] = defaultdict(list)
    for c in cells:
        by_val[c[knob]].append(c["mean"])
    result = []
    for val in sorted(by_val):
        vals = by_val[val]
        result.append(
            {
                "value": val,
                "n_cells": len(vals),
                "mean": statistics.fmean(vals),
                "ci95": _ci95(vals),
            }
        )
    return result


def _sobol_lite(cells: list[dict], knob: str) -> float:
    """First-order range-delta / sigma_all. Larger = knob moves the signal more."""
    vals_by_knob: dict[float, list[float]] = defaultdict(list)
    for c in cells:
        vals_by_knob[c[knob]].append(c["mean"])
    knob_vals = sorted(vals_by_knob)
    lo_mean = statistics.fmean(vals_by_knob[knob_vals[0]])
    hi_mean = statistics.fmean(vals_by_knob[knob_vals[-1]])
    all_means = [c["mean"] for c in cells]
    sigma_all = statistics.pstdev(all_means)
    return abs(hi_mean - lo_mean) / sigma_all if sigma_all > 0 else 0.0


def compute_stats(input_path: Path) -> dict:
    rows = _load_rows(input_path)
    per_cell = _per_cell_stats(rows)
    cells = list(per_cell.values())
    full = [c for c in cells if c["n_ok"] >= MIN_FULL_N_OK]
    partial = [c for c in cells if c["n_ok"] < MIN_FULL_N_OK]
    full_sorted = sorted(full, key=lambda c: c["mean"])
    best, worst = full_sorted[-1], full_sorted[0]
    baseline = next(c for c in cells if (c["alpha"], c["beta"], c["gamma"], c["p_skills"]) == BASELINE)
    marginals = {k: _marginal(cells, k) for k in KNOBS}
    sobol = {k: _sobol_lite(cells, k) for k in KNOBS}
    return {
        "num_cells": len(cells),
        "num_cells_full_n_ok": len(full),
        "baseline": baseline,
        "best_cell": best,
        "worst_cell": worst,
        "partial_cells": partial,
        "marginals": marginals,
        "sobol_lite": sobol,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args(argv)
    stats = compute_stats(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"wrote {args.output} ({stats['num_cells']} cells, {stats['num_cells_full_n_ok']} full)")
    print(f"best:  {stats['best_cell']['sha']} mean={stats['best_cell']['mean']:.4f}")
    print(f"worst: {stats['worst_cell']['sha']} mean={stats['worst_cell']['mean']:.4f}")
    print(f"baseline mean={stats['baseline']['mean']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
