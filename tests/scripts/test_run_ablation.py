"""Runner-level tests for scripts/run_ablation.py (Wave-3 Stream A).

The runner is subprocess-heavy by design — these tests monkey-patch the
single-seed worker (``_run_seed``) to a synchronous in-process stub so we
can exercise resume / SHA / seed_table contracts without spending wall-clock
on actual PPO. The smoke-grid test stays cheap (1 cell × 3 seeds, mocked).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _ablation_lib, run_ablation  # noqa: E402


def _fast_run_seed(cell, seed: int, timeout_s: int) -> dict:
    """Stub: pretend the seed converged in 0.01s with a deterministic reward."""
    # timeout_s is unused — kept for signature compatibility with the real worker.
    del timeout_s
    return {
        "seed": int(seed),
        "final_reward": float(seed) * 0.001 - cell.alpha,
        "status": "ok",
        "elapsed_s": 0.01,
    }


def test_smoke_grid_runs_one_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--grid smoke --max-cells 1` produces 1 cell dir with a valid done.json."""
    monkeypatch.setattr(run_ablation, "_run_seed", _fast_run_seed)
    rc = run_ablation.main(
        [
            "--grid",
            "smoke",
            "--max-cells",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    cell_dirs = sorted(tmp_path.glob("cell_*"))
    assert len(cell_dirs) == 1, f"expected 1 cell dir, got: {cell_dirs}"
    payload = json.loads((cell_dirs[0] / "done.json").read_text(encoding="utf-8"))
    # Smoke grid pins (1.0, 1.0, 0.5, -5.0).
    assert payload["alpha"] == 1.0
    assert payload["p_skills"] == -5.0
    assert payload["n_ok"] == len(payload["seed_outcomes"]) == 3
    # SHA in payload matches the directory name.
    assert cell_dirs[0].name == f"cell_{payload['cell_sha']}"
    # seed_table.csv written with one row per (cell, seed).
    seed_table = tmp_path / "seed_table.csv"
    assert seed_table.exists()
    lines = seed_table.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + 3  # header + 3 seeds


def test_resume_skips_completed_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second invocation with --resume skips the cell whose done.json already exists."""
    monkeypatch.setattr(run_ablation, "_run_seed", _fast_run_seed)

    # First pass — write the marker.
    rc1 = run_ablation.main(
        [
            "--grid",
            "smoke",
            "--max-cells",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc1 == 0
    cell_dirs = sorted(tmp_path.glob("cell_*"))
    assert len(cell_dirs) == 1
    marker = cell_dirs[0] / "done.json"
    first_payload = json.loads(marker.read_text(encoding="utf-8"))
    first_mtime = marker.stat().st_mtime_ns

    # Replace the stub with one that EXPLODES on call — the resume path
    # must skip it without spawning the worker.
    def _no_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("worker called during --resume; cell_done() not respected")

    monkeypatch.setattr(run_ablation, "_run_seed", _no_call)

    rc2 = run_ablation.main(
        [
            "--grid",
            "smoke",
            "--max-cells",
            "1",
            "--output-dir",
            str(tmp_path),
            "--resume",
        ]
    )
    assert rc2 == 0
    # Payload byte-identical → the worker was never invoked.
    second_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert second_payload == first_payload
    assert marker.stat().st_mtime_ns == first_mtime


def test_force_reruns_completed_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--force overrides the resume short-circuit even when done.json exists."""
    monkeypatch.setattr(run_ablation, "_run_seed", _fast_run_seed)

    # Pre-seed a fake done.json so we can detect overwrite by checking payload contents.
    cells = run_ablation._build_cells(
        run_ablation._parse_args(["--grid", "smoke", "--output-dir", str(tmp_path)]),
        cfg={
            "ablation": {
                "grids": {"smoke": {"alpha": [1.0], "beta": [1.0], "gamma": [0.5], "p_skills": [-5.0]}},
                "scout_seeds": [42, 7, 271],
                "total_steps_per_cell_seed": 256,
            }
        },
    )
    assert len(cells) == 1
    sha = _ablation_lib.cell_sha(cells[0])
    cell_dir = tmp_path / f"cell_{sha}"
    _ablation_lib.mark_cell_done(cell_dir, {"alpha": -999.0, "sentinel": True})

    rc = run_ablation.main(
        [
            "--grid",
            "smoke",
            "--max-cells",
            "1",
            "--output-dir",
            str(tmp_path),
            "--force",
        ]
    )
    assert rc == 0
    refreshed = json.loads((cell_dir / "done.json").read_text(encoding="utf-8"))
    assert "sentinel" not in refreshed
    assert refreshed["alpha"] == 1.0  # smoke-grid value, not the sentinel
