"""Pure-function tests for scripts/_ablation_lib.py (Wave-3 Stream A)."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._ablation_lib import (  # noqa: E402
    Cell,
    cell_done,
    cell_sha,
    make_grid,
    mark_cell_done,
    t_ci95,
)


def _cell(**overrides) -> Cell:
    defaults = {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.5,
        "p_skills": -5.0,
        "total_steps": 256,
        "seed_list": (42, 7, 271),
    }
    defaults.update(overrides)
    return Cell(**defaults)


def test_cell_sha_deterministic() -> None:
    """Equal Cells → equal SHAs; run_id label is excluded from the digest."""
    a = _cell()
    b = _cell()
    assert cell_sha(a) == cell_sha(b)
    # Different reward coeffs → different SHA.
    c = _cell(alpha=2.0)
    assert cell_sha(a) != cell_sha(c)
    # run_id is a human label — should NOT affect the SHA.
    d = replace(a, run_id="ignored-label")
    assert cell_sha(a) == cell_sha(d)
    # SHA is 12 hex chars.
    assert len(cell_sha(a)) == 12
    assert all(ch in "0123456789abcdef" for ch in cell_sha(a))


def test_make_grid_size_compact() -> None:
    """3×3×3×3 reward grid × 3 seeds → 81 cells; each cell carries all 3 seeds."""
    grid = {
        "alpha": [0.5, 1.0, 2.0],
        "beta": [0.5, 1.0, 2.0],
        "gamma": [0.0, 0.5, 1.0],
        "p_skills": [-10.0, -5.0, -1.0],
    }
    seeds = [42, 7, 271]
    cells = make_grid(grid, seeds, total_steps=256)
    assert len(cells) == 81
    for cell in cells:
        assert cell.seed_list == tuple(seeds)
        assert cell.total_steps == 256
    # Sanity: all 81 SHAs are unique → no accidental collisions in the product.
    assert len({cell_sha(c) for c in cells}) == 81


def test_make_grid_size_smoke() -> None:
    """1×1×1×1 smoke grid × 3 seeds → 1 cell."""
    grid = {"alpha": [1.0], "beta": [1.0], "gamma": [0.5], "p_skills": [-5.0]}
    cells = make_grid(grid, [42, 7, 271], total_steps=256)
    assert len(cells) == 1
    only = cells[0]
    assert (only.alpha, only.beta, only.gamma, only.p_skills) == (1.0, 1.0, 0.5, -5.0)


def test_make_grid_missing_knob_raises() -> None:
    """Missing reward knob in grid_dict → KeyError (fail loud, not silent shrink)."""
    incomplete = {"alpha": [1.0], "beta": [1.0], "gamma": [0.5]}  # p_skills missing
    with pytest.raises(KeyError, match="p_skills"):
        make_grid(incomplete, [42], total_steps=256)


def test_t_ci95_known_value() -> None:
    """mean=10, std=2, n=5 → matches Student-t critical * std/√n at dof=4."""
    # values chosen so mean=10, sample std=2 exactly (uses Bessel-corrected stdev).
    # Easy construction: [8, 9, 10, 11, 12] → mean=10, stdev=√(10/4)=√2.5≈1.5811.
    # We hand-pick the values to recover std=2.0:
    # variance(2) for n=5 with mean 10: choose ±2, ±2, 0 → values 12,12,8,8,10
    values = [12.0, 12.0, 8.0, 8.0, 10.0]
    m, half = t_ci95(values)
    assert m == pytest.approx(10.0, abs=1e-9)
    # Sample stdev (Bessel) of those values = sqrt(sum((x-10)^2) / 4) = sqrt(16/4) = 2.0
    # t_crit at dof=4 (97.5%) = 2.7764; half = 2.7764 * 2 / sqrt(5)
    expected_half = 2.7764 * 2.0 / math.sqrt(5)
    assert half == pytest.approx(expected_half, rel=1e-4)


def test_t_ci95_small_n_returns_zero_halfwidth() -> None:
    """n<2 → (mean, 0.0); n=0 → (NaN, 0.0)."""
    m, half = t_ci95([7.5])
    assert m == 7.5
    assert half == 0.0
    m2, half2 = t_ci95([])
    assert math.isnan(m2)
    assert half2 == 0.0


def test_t_ci95_filters_non_finite() -> None:
    """NaN/inf values are excluded before computing the CI."""
    m, half = t_ci95([10.0, float("nan"), float("inf"), 12.0, 8.0])
    # Only [10, 12, 8] survive → mean=10, std=2, dof=2 → t=4.3027
    assert m == pytest.approx(10.0)
    expected_half = 4.3027 * 2.0 / math.sqrt(3)
    assert half == pytest.approx(expected_half, rel=1e-3)


def test_cell_done_false_when_missing(tmp_path: Path) -> None:
    """No marker → cell_done returns False."""
    assert cell_done(tmp_path) is False


def test_cell_done_false_on_corrupt_marker(tmp_path: Path) -> None:
    """Garbage / partial done.json → cell_done returns False (not raises)."""
    (tmp_path / "done.json").write_text("{not json", encoding="utf-8")
    assert cell_done(tmp_path) is False
    (tmp_path / "done.json").write_text("[]", encoding="utf-8")  # list, not dict
    assert cell_done(tmp_path) is False
    (tmp_path / "done.json").write_text("{}", encoding="utf-8")  # empty dict
    assert cell_done(tmp_path) is False


def test_cell_done_atomic_write(tmp_path: Path) -> None:
    """mark_cell_done writes via .tmp + rename — no partial done.json ever observed."""
    payload = {"alpha": 1.0, "n_ok": 3, "mean": -0.34, "ci95": 0.14}
    mark_cell_done(tmp_path, payload)
    marker = tmp_path / "done.json"
    assert marker.exists()
    # No leftover .tmp file after a successful write.
    assert not (tmp_path / "done.json.tmp").exists()
    reloaded = json.loads(marker.read_text(encoding="utf-8"))
    assert reloaded == payload
    assert cell_done(tmp_path) is True
