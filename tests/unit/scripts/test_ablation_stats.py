"""Tests for ``scripts/ablation_stats.py`` (Phase-4 Wave-4c AB-STATS).

Locks the per-cell aggregation, Student-t CI95 dof = n_ok-1 contract, the
3-value marginal grouping, Sobol-lite non-negativity, baseline-cell presence,
and the partial-cell (n_ok<3) handling.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ablation_stats.py"
_SPEC = importlib.util.spec_from_file_location("ablation_stats", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["ablation_stats"] = mod
_SPEC.loader.exec_module(mod)


_FIELDS = ("cell_sha", "alpha", "beta", "gamma", "p_skills", "seed", "final_reward", "status", "elapsed_s")


def _row(sha, a, b, g, p, seed, r, status="ok", elapsed=1.0):  # noqa: PLR0913
    return dict(zip(_FIELDS, (sha, a, b, g, p, seed, r, status, elapsed), strict=True))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _full_grid(tmp_path: Path) -> Path:
    """Synthesise a compact 3^4 = 81-cell grid, 3 seeds each, all ok."""
    csv_path = tmp_path / "seed_table.csv"
    rows = []
    coords = [
        (a, b, g, p)
        for a in (0.5, 1.0, 2.0)
        for b in (0.5, 1.0, 2.0)
        for g in (0.0, 0.5, 1.0)
        for p in (-10.0, -5.0, -1.0)
    ]
    for n, (a, b, g, p) in enumerate(coords):
        sha = f"cell{n:04d}"
        for seed in (42, 7, 271):
            r = -0.1 * a + 0.05 * b - 0.2 * g + 0.01 * p + (seed % 3) * 0.001
            rows.append(_row(sha, a, b, g, p, seed, r))
    _write_csv(csv_path, rows)
    return csv_path


def test_loads_seed_table_csv(tmp_path: Path) -> None:
    """81 distinct (sha) cells produce 81 entries in the stats output."""
    stats = mod.compute_stats(_full_grid(tmp_path))
    assert stats["num_cells"] == 81
    assert stats["num_cells_full_n_ok"] == 81


def test_per_cell_stats_t_ci95_dof(tmp_path: Path) -> None:
    """For n_ok=3, ci95 = t(dof=2) * stdev / sqrt(n); t(2)=4.302652729911275."""
    csv_path = tmp_path / "tiny.csv"
    _write_csv(
        csv_path,
        [_row("x", 1.0, 1.0, 0.5, -5.0, s, v) for s, v in zip((42, 7, 271), (0.1, 0.2, 0.3), strict=True)],
    )
    per = mod._per_cell_stats(mod._load_rows(csv_path))
    cell = next(iter(per.values()))
    assert cell["n_ok"] == 3
    expected = 4.302652729911275 * (((0.1 - 0.2) ** 2 + 0 + (0.3 - 0.2) ** 2) / 2) ** 0.5 / math.sqrt(3)
    assert cell["ci95"] == pytest.approx(expected, rel=1e-9)


def test_marginal_grouping(tmp_path: Path) -> None:
    """Each of the 4 knobs surfaces exactly 3 distinct values (compact grid)."""
    stats = mod.compute_stats(_full_grid(tmp_path))
    for knob in ("alpha", "beta", "gamma", "p_skills"):
        assert len(stats["marginals"][knob]) == 3
        assert all(entry["n_cells"] == 27 for entry in stats["marginals"][knob])


def test_sobol_lite_nonnegative(tmp_path: Path) -> None:
    """All 4 first-order Sobol-lite scores must be >= 0 (absolute Δ)."""
    stats = mod.compute_stats(_full_grid(tmp_path))
    for knob in ("alpha", "beta", "gamma", "p_skills"):
        assert stats["sobol_lite"][knob] >= 0.0


def test_baseline_cell_found(tmp_path: Path) -> None:
    """Canonical (α=1.0, β=1.0, γ=0.5, P_skills=-5.0) must surface in stats."""
    stats = mod.compute_stats(_full_grid(tmp_path))
    b = stats["baseline"]
    assert (b["alpha"], b["beta"], b["gamma"], b["p_skills"]) == (1.0, 1.0, 0.5, -5.0)
    assert b["n_ok"] == 3


def test_handles_partial_n_ok(tmp_path: Path) -> None:
    """n_ok=1 cell ⇒ ci95=0.0 (no t-stat definable), not NaN; appears in partial_cells."""
    csv_path = tmp_path / "partial.csv"
    rows = [
        *(_row("base", 1.0, 1.0, 0.5, -5.0, s, 0.1) for s in (42, 7, 271)),
        *(_row("extra", 0.5, 0.5, 0.0, -10.0, s, -0.2) for s in (42, 7, 271)),
        _row("partial1", 2.0, 2.0, 1.0, -1.0, 42, 0.5),
        _row("partial1", 2.0, 2.0, 1.0, -1.0, 7, 0.0, status="TIMEOUT", elapsed=240.0),
    ]
    _write_csv(csv_path, rows)
    stats = mod.compute_stats(csv_path)
    partial = [c for c in stats["partial_cells"] if c["sha"] == "partial1"]
    assert len(partial) == 1
    assert partial[0]["n_ok"] == 1
    assert partial[0]["ci95"] == 0.0
    assert not math.isnan(partial[0]["ci95"])
