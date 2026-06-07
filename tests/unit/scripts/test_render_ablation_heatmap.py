"""Tests for ``scripts/render_ablation_heatmap.py`` (Phase-4 AB-HEATMAP + D2).

Locks the ablation-heatmap deliverable contract: PNG outputs exist, the
headline heatmap is rendered at a print-readable width (>= 1500 px at
300 dpi), the script's CLI returns 0 from a clean subprocess, and the
hatch code path fires when a cell has ``n_ok < 3``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "render_ablation_heatmap.py"
_SPEC = importlib.util.spec_from_file_location("render_ablation_heatmap", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["render_ablation_heatmap"] = mod
_SPEC.loader.exec_module(mod)


def _run_main(tmp_path: Path) -> tuple[Path, Path, int]:
    out_heat = tmp_path / "ablation_heatmap.png"
    out_d2 = tmp_path / "essay_d2_ablation_summary.png"
    rc = mod.main(
        [
            "--input",
            str(_REPO / "results" / "ablations" / "seed_table.csv"),
            "--out-heatmap",
            str(out_heat),
            "--out-d2",
            str(out_d2),
        ]
    )
    return out_heat, out_d2, rc


def test_heatmap_png_exists_and_nonempty(tmp_path: Path) -> None:
    out_heat, _, rc = _run_main(tmp_path)
    assert rc == 0
    assert out_heat.exists()
    assert out_heat.stat().st_size > 5_000


def test_heatmap_dimensions(tmp_path: Path) -> None:
    """Heatmap PNG must be >= 1500 px wide for print readability."""
    pil_image = pytest.importorskip("PIL.Image")
    out_heat, _, rc = _run_main(tmp_path)
    assert rc == 0
    with pil_image.open(out_heat) as img:
        width, _ = img.size
    assert width >= 1500


def test_d2_png_exists(tmp_path: Path) -> None:
    _, out_d2, rc = _run_main(tmp_path)
    assert rc == 0
    assert out_d2.exists()
    assert out_d2.stat().st_size > 5_000


def test_invocation_no_errors(tmp_path: Path) -> None:
    """Fresh-subprocess CLI invocation returns 0 and emits both files."""
    out_heat = tmp_path / "heat.png"
    out_d2 = tmp_path / "d2.png"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--input",
            str(_REPO / "results" / "ablations" / "seed_table.csv"),
            "--out-heatmap",
            str(out_heat),
            "--out-d2",
            str(out_d2),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert out_heat.exists() and out_d2.exists()


def test_hatch_for_partial_cells(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hatch path fires for any cell with n_ok < 3.

    Monkeypatch ``_slice_for_heatmap`` to force a partial cell into the
    n_ok matrix; assert ``_draw_heatmap`` adds at least one hatched
    Rectangle patch to the axes (the script's hatching code path).
    """
    del monkeypatch  # signature kept for fixture parity; not needed here
    del tmp_path
    xs = [0.5, 1.0, 2.0]
    ys = [0.0, 0.5, 1.0]
    mean = np.full((3, 3), -0.5)
    ci = np.full((3, 3), 0.1)
    nok = np.full((3, 3), 3, dtype=int)
    nok[1, 1] = 2  # one partial cell at the center
    grid = {"xs": xs, "ys": ys, "mean": mean, "ci": ci, "nok": nok}
    fig, ax = plt.subplots()
    mod._draw_heatmap(ax, grid, ("alpha", "gamma"))
    hatched = [
        p
        for p in ax.patches
        if hasattr(p, "get_hatch") and p.get_hatch() is not None and "/" in (p.get_hatch() or "")
    ]
    plt.close(fig)
    assert hatched, "expected at least one hatched Rectangle for n_ok<3 cell"
