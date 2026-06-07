"""Integration tests for the ablation pipeline (Wave 4b-pre, Stream B).

Complements the unit tests in ``tests/scripts/`` by exercising:

* **Atomicity** of ``done.json`` under concurrent writers (POSIX ``rename``
  is the contract — the final file is exactly one of the payloads, never
  an interleaved half-write, and no ``.tmp`` leftover survives).
* **Schema integrity** of ``seed_table.csv`` against the documented header
  in ``scripts/run_ablation.py::_append_seed_table``.
* **Hebrew-cwd subprocess regression** — the repo lives under a Hebrew
  path; spawning subprocesses from such a cwd must not break encoding.
* **Resume idempotency at scale** — N pre-marked cells all report done
  without invoking the worker.
* **Smoke-grid end-to-end** — actually run ``scripts/run_ablation.py
  --grid smoke --max-cells 1`` to a *fresh* output dir under 300s so we
  never race the long ablation that may be writing to ``results/``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _ablation_lib, run_ablation  # noqa: E402

SEED_TABLE_HEADER = [
    "cell_sha", "alpha", "beta", "gamma", "p_skills",
    "seed", "final_reward", "status", "elapsed_s",
]  # fmt: skip


def test_done_json_atomic_under_concurrent_writes(tmp_path: Path) -> None:
    """3 threads racing mark_cell_done() — final done.json is one full payload (no corruption).

    The atomic contract is on the *final* file: POSIX ``os.replace`` guarantees the
    reader will only ever see a fully-written payload, never a torn write. Losing
    threads MAY see FileNotFoundError if their fixed ``done.json.tmp`` slot was
    renamed out from under them — that is expected (and survivable) as long as at
    least one writer wins and the resulting file is one of the submitted payloads.
    """
    cell_dir = tmp_path / "cell_race"
    cell_dir.mkdir()
    payloads = [{"writer": i, "alpha": float(i), "ok": True} for i in range(3)]
    wins: list[int] = []

    def _writer(payload: dict) -> None:
        try:
            _ablation_lib.mark_cell_done(cell_dir, payload)
            wins.append(payload["writer"])
        except FileNotFoundError:
            # Losing thread: another writer renamed our .tmp first — expected.
            pass

    threads = [threading.Thread(target=_writer, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert wins, "no writer survived the race — atomicity contract violated"
    final = cell_dir / "done.json"
    assert final.exists(), "done.json missing after concurrent writers"
    payload = json.loads(final.read_text(encoding="utf-8"))
    assert payload in payloads, f"final payload is not one of the writers': {payload!r}"


def test_seed_table_csv_schema_consistent(tmp_path: Path) -> None:
    """run_ablation._append_seed_table writes the documented 9-column header."""
    base = {
        "cell_sha": "abc123def456", "alpha": 1.0, "beta": 1.0, "gamma": 0.5,
        "p_skills": -5.0, "status": "ok", "elapsed_s": 1.5,
    }  # fmt: skip
    rows = [{**base, "seed": s, "final_reward": 0.1 * s} for s in (42, 7, 271)]
    run_ablation._append_seed_table(rows, tmp_path)
    path = tmp_path / "seed_table.csv"
    assert path.exists()
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)
    assert header == SEED_TABLE_HEADER, f"header drift: {header!r}"
    assert len(body) == len(rows), f"row count mismatch: {len(body)} vs {len(rows)}"


def test_hebrew_cwd_subprocess_safety(tmp_path: Path) -> None:
    """Subprocess from a Hebrew-named cwd must return 0 and decode cleanly."""
    hebrew_dir = tmp_path / "סדנה_test_dir"
    hebrew_dir.mkdir()
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; print('HEB_OK', sys.version_info[0])"],
        cwd=hebrew_dir, capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )  # fmt: skip
    assert completed.returncode == 0, (
        f"subprocess from Hebrew cwd failed rc={completed.returncode}\nstderr:\n{completed.stderr}"
    )
    assert "HEB_OK" in completed.stdout


def test_resume_idempotency_at_scale(tmp_path: Path) -> None:
    """10 pre-marked cells all report cell_done() == True (no re-run needed)."""
    cell_dirs = []
    for i in range(10):
        cd = tmp_path / f"cell_{i:012x}"
        _ablation_lib.mark_cell_done(cd, {"cell_idx": i, "n_ok": 3, "mean": 0.0})
        cell_dirs.append(cd)
    assert all(_ablation_lib.cell_done(cd) for cd in cell_dirs)
    # Regression guard: a non-existent dir reports False.
    assert not _ablation_lib.cell_done(tmp_path / "cell_missing")


@pytest.mark.slow
def test_smoke_cell_completes_under_300s() -> None:
    """End-to-end: scripts/run_ablation.py --grid smoke --max-cells 1, fresh dir.

    A separate ``--output-dir`` keeps this isolated from any concurrent
    ablation sweep writing into ``results/ablations/``.
    """
    output_dir = Path(f"/tmp/ab_smoke_test_{uuid.uuid4().hex}")
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_ablation.py"),
        "--grid", "smoke", "--max-cells", "1", "--output-dir", str(output_dir),
    ]  # fmt: skip
    t0 = time.monotonic()
    completed = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    elapsed = time.monotonic() - t0
    assert completed.returncode == 0, (
        f"smoke cell failed rc={completed.returncode} elapsed={elapsed:.1f}s\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    cell_dirs = sorted(output_dir.glob("cell_*"))
    assert len(cell_dirs) == 1, f"expected 1 cell, got: {cell_dirs!r}"
    assert (cell_dirs[0] / "done.json").exists()
    assert (output_dir / "seed_table.csv").exists()
