"""Tests for ``scripts/render_metric_curves.py`` (brief §3 D9b).

Locks the machine-checkable surface: per-metric aggregation (mean ± t-CI95
across seeds, aligned by step), a 3-panel PNG, and a tidy CSV. The heavy
checkpoint-replay path (``_collect``) is exercised when the artifact is
regenerated, not in unit tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "render_metric_curves.py"

_SPEC = importlib.util.spec_from_file_location("render_metric_curves", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["render_metric_curves"] = mod
_SPEC.loader.exec_module(mod)


def _synthetic() -> dict[str, list[dict[str, float]]]:
    """Two seeds, three steps, monotone improvement (mod↑ coh↑ coup↓)."""
    rows_a = [
        {"step": 0.0, "modularity": 0.10, "cohesion": 0.20, "coupling": 0.50},
        {"step": 1.0, "modularity": 0.15, "cohesion": 0.25, "coupling": 0.45},
        {"step": 2.0, "modularity": 0.20, "cohesion": 0.30, "coupling": 0.40},
    ]
    rows_b = [
        {"step": 0.0, "modularity": 0.12, "cohesion": 0.18, "coupling": 0.52},
        {"step": 1.0, "modularity": 0.17, "cohesion": 0.23, "coupling": 0.47},
        {"step": 2.0, "modularity": 0.22, "cohesion": 0.28, "coupling": 0.42},
    ]
    return {"seed_42": rows_a, "seed_7": rows_b}


def test_aggregate_per_metric_steps() -> None:
    agg = mod._aggregate(_synthetic())
    assert set(agg) == {"modularity", "cohesion", "coupling"}
    for metric in ("modularity", "cohesion", "coupling"):
        series = agg[metric]
        assert len(series["step"]) == 3
        assert len(series["mean"]) == 3
        assert len(series["ci"]) == 3
    # modularity mean at step 0 = (0.10 + 0.12) / 2 = 0.11
    assert abs(agg["modularity"]["mean"][0] - 0.11) < 1e-9
    assert agg["modularity"]["n"] == 2


def test_render_writes_3panel_png(tmp_path: Path) -> None:
    agg = mod._aggregate(_synthetic())
    out = tmp_path / "metric_improvement_curves.png"
    mod.render(agg, out)
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_write_csv_is_tidy(tmp_path: Path) -> None:
    agg = mod._aggregate(_synthetic())
    out = tmp_path / "metric_curves.csv"
    mod._write_csv(agg, out)
    text = out.read_text(encoding="utf-8")
    assert "metric,step,mean,ci95,n" in text
    assert "modularity" in text and "coupling" in text
