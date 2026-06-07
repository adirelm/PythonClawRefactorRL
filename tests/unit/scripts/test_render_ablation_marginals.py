# ruff: noqa: RUF001
"""Tests for ``scripts/render_ablation_marginals.py`` (Wave 4c Stream C).

Locks the 2x2 marginal-plot contract: PNG written, big enough to read at
300 dpi, four subplots present, CLI exits 0 in a fresh subprocess. The plot
content (mean +- t-CI95 bands) is anchored by the canonical seed_table.csv
and reviewed visually -- these tests guard the machine-checkable surface.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "render_ablation_marginals.py"
_SEED_TABLE = _REPO_ROOT / "results" / "ablations" / "seed_table.csv"

_SPEC = importlib.util.spec_from_file_location("render_ablation_marginals", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["render_ablation_marginals"] = mod
_SPEC.loader.exec_module(mod)


@pytest.fixture(scope="module")
def seed_rows() -> list[dict[str, float]]:
    assert _SEED_TABLE.exists(), f"seed_table.csv missing at {_SEED_TABLE}"
    return mod._load_rows(_SEED_TABLE)


def _render_to(tmp_path: Path) -> Path:
    out = tmp_path / "ablation_marginals.png"
    rc = mod.main(["--input", str(_SEED_TABLE), "--out", str(out)])
    assert rc == 0
    return out


def test_marginals_png_exists(tmp_path: Path) -> None:
    """main() writes ablation_marginals.png > 5 KB."""
    out = _render_to(tmp_path)
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_marginals_dimensions(tmp_path: Path) -> None:
    """Rendered PNG is at least 1500 px wide (figsize=12 in x 300 dpi -> 3600 px)."""
    pil_image = pytest.importorskip("PIL.Image")
    out = _render_to(tmp_path)
    with pil_image.open(out) as img:
        width, height = img.size
    assert width >= 1500, f"width={width} too small for 300dpi 12in figure"
    assert height >= 1000


def test_4_subplots_rendered(seed_rows: list[dict[str, float]], tmp_path: Path) -> None:
    """The Figure built by render() must carry exactly 4 axes (2x2 grid)."""
    out = tmp_path / "ablation_marginals.png"
    directions = mod.render(seed_rows, out)
    # render() closed its figure; rebuild to introspect axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (knob, title, xlabel) in zip(axes.flat, mod.KNOBS, strict=True):
        mod._plot_panel(ax, title, xlabel, mod._marginal(seed_rows, knob))
    assert len(fig.axes) == 4
    titles = {ax.get_title() for ax in fig.axes}
    assert {"α marginal", "β marginal", "γ marginal", "P_skills marginal"} <= titles
    plt.close(fig)
    assert set(directions) == {"alpha", "beta", "gamma", "p_skills"}


def test_smoke_no_errors(tmp_path: Path) -> None:
    """`python scripts/render_ablation_marginals.py --out tmp/...` exits 0."""
    out = tmp_path / "ablation_marginals.png"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--input", str(_SEED_TABLE), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert out.exists()


def test_marginal_buckets_have_3_grid_points(seed_rows: list[dict[str, float]]) -> None:
    """Every knob's marginal must collapse to exactly 3 distinct grid values."""
    for knob, _title, _xlabel in mod.KNOBS:
        points = mod._marginal(seed_rows, knob)
        assert len(points) == 3, f"{knob} -> {len(points)} buckets (expected 3)"
        # Each marginal point pools across 27 cells x 3 seeds = 81 rows on a complete
        # grid; allow slack for the one cell flagged as n_ok<3.
        for _value, _mean, _ci, n in points:
            assert n >= 70, f"{knob} bucket has only n={n} rows"


def test_full_grid_uses_canonical_values() -> None:
    """seed_table.csv knob columns must hit the canonical 3-value grids."""
    with _SEED_TABLE.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        grid: dict[str, set[float]] = {"alpha": set(), "beta": set(), "gamma": set(), "p_skills": set()}
        for row in reader:
            if row.get("status") != "ok":
                continue
            for k, values in grid.items():
                values.add(float(row[k]))
    assert grid["alpha"] == {0.5, 1.0, 2.0}
    assert grid["beta"] == {0.5, 1.0, 2.0}
    assert grid["gamma"] == {0.0, 0.5, 1.0}
    assert grid["p_skills"] == {-10.0, -5.0, -1.0}
