"""Unit tests for ``src/sdk/ablation.py`` (Wave 4a §AB-SDK).

Each test synthesises its own tmp dir so the suite never races the
live AB-EXEC background sweep under ``results/ablations/``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.sdk import Ablation, run_ablation
from tests.unit.sdk._fixtures import DONE_OK, seed_row, write_cell, write_seed_table


def test_cellresult_from_done_json_maps_every_field(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_cell(out, DONE_OK)
    abl = run_ablation(grid_name="compact", output_dir=out)
    assert len(abl.cells) == 1
    cell = abl.cells[0]
    assert cell.cell_sha == "abc123def456"
    assert cell.alpha == pytest.approx(1.0)
    assert cell.beta == pytest.approx(1.0)
    assert cell.gamma == pytest.approx(0.5)
    assert cell.p_skills == pytest.approx(-5.0)
    assert cell.n_ok == 2
    assert cell.mean_final_reward == pytest.approx(0.42)
    assert cell.ci95_halfwidth == pytest.approx(0.05)
    assert len(cell.seed_outcomes) == 2
    assert cell.seed_outcomes[0]["seed"] == 42


def test_ablation_from_seed_table_csv_when_no_done_markers(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_seed_table(
        out,
        [
            seed_row("deadbeef0001", 42, 0.1, "ok"),
            seed_row("deadbeef0001", 7, 0.3, "ok", elapsed=11.0),
        ],
    )
    abl = run_ablation(grid_name="compact", output_dir=out)
    assert isinstance(abl, Ablation)
    assert abl.grid_name == "compact"
    assert len(abl.cells) == 1
    cell = abl.cells[0]
    assert cell.cell_sha == "deadbeef0001"
    assert cell.n_ok == 2
    assert cell.mean_final_reward == pytest.approx(0.2)
    assert abl.total_runs == 2
    assert abl.n_ok_total == 2


def test_run_ablation_idempotent_when_resume(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_cell(out, DONE_OK)
    a1 = run_ablation(grid_name="compact", output_dir=out)
    a2 = run_ablation(grid_name="compact", output_dir=out)
    assert a1.cells == a2.cells
    assert a1.total_runs == a2.total_runs
    assert a1.n_ok_total == a2.n_ok_total


def test_run_ablation_smoke_grid_returns_single_cell(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    payload = dict(DONE_OK)
    payload["cell_sha"] = "smoke0000aaaa"
    write_cell(out, payload)
    abl = run_ablation(grid_name="smoke", output_dir=out)
    assert abl.grid_name == "smoke"
    assert len(abl.cells) == 1


def test_sdk_no_subprocess_leak_when_empty_dir(tmp_path: Path) -> None:
    out = tmp_path / "empty"
    out.mkdir()
    abl = run_ablation(grid_name="compact", output_dir=out)
    assert abl.cells == ()
    assert abl.total_runs == 0
    assert abl.n_ok_total == 0
    assert abl.total_wall_clock_s >= 0.0


def test_run_ablation_handles_missing_dir(tmp_path: Path) -> None:
    abl = run_ablation(grid_name="compact", output_dir=tmp_path / "does-not-exist")
    assert abl.cells == ()


def test_seed_table_skips_failed_seeds_in_n_ok(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_seed_table(
        out,
        [
            seed_row("mixed0000aaa", 42, 0.5, "ok"),
            seed_row("mixed0000aaa", 7, "nan", "fail", elapsed=0.0),
        ],
    )
    abl = run_ablation(output_dir=out)
    cell = abl.cells[0]
    assert cell.n_ok == 1
    assert cell.mean_final_reward == pytest.approx(0.5)
    assert cell.ci95_halfwidth == 0.0


def test_done_json_takes_precedence_over_seed_table(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_cell(out, DONE_OK)
    write_seed_table(out, [seed_row("deadbeef0002", 42, 0.0, "ok", elapsed=1.0)])
    abl = run_ablation(output_dir=out)
    assert len(abl.cells) == 1
    assert abl.cells[0].cell_sha == "abc123def456"


def test_ablation_and_cellresult_are_frozen(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_cell(out, DONE_OK)
    abl = run_ablation(output_dir=out)
    with pytest.raises(FrozenInstanceError):
        abl.grid_name = "tampered"  # type: ignore[misc]
    cell = abl.cells[0]
    with pytest.raises(FrozenInstanceError):
        cell.alpha = 9.0  # type: ignore[misc]


def test_default_output_dir_is_repo_results_ablations() -> None:
    abl = run_ablation(grid_name="compact")
    assert isinstance(abl, Ablation)
    assert abl.grid_name == "compact"


def test_run_ablation_with_string_path(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_cell(out, DONE_OK)
    abl = run_ablation(output_dir=str(out))
    assert len(abl.cells) == 1


def test_run_ablation_skips_corrupt_done_json(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    write_cell(out, DONE_OK)
    bad = out / "cell_corruptcell11"
    bad.mkdir()
    (bad / "done.json").write_text("{not-json", encoding="utf-8")
    abl = run_ablation(output_dir=out)
    assert len(abl.cells) == 1
    assert abl.cells[0].cell_sha == "abc123def456"
