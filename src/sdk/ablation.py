"""SDK ablation surface — structured ``Ablation`` object for notebook consumers.

CLAUDE.md §3 ("SDK is the single entry point"): notebooks consume the sweep via
``run_ablation`` (resume-aware: parses ``results/ablations/`` artefacts written
by ``scripts/run_ablation.py`` — never shells out). Dataclasses are frozen so a
notebook mutation fails loudly instead of silently desynchronising from disk.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.sdk._ab_stats import t_ci95

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "ablations"


@dataclass(frozen=True)
class CellResult:
    """One ablation grid cell — coefficients + per-seed outcomes + mean/CI."""

    cell_sha: str
    alpha: float
    beta: float
    gamma: float
    p_skills: float
    n_ok: int
    seed_outcomes: tuple[dict, ...]
    mean_final_reward: float
    ci95_halfwidth: float


@dataclass(frozen=True)
class Ablation:
    """Aggregate of one grid run — sealed view over results/ablations/."""

    grid_name: str
    cells: tuple[CellResult, ...]
    total_runs: int
    n_ok_total: int
    total_wall_clock_s: float


def _cellresult_from_done(payload: dict) -> CellResult:
    """Map a ``done.json`` payload onto :class:`CellResult` (sealed schema)."""
    outcomes = tuple(dict(row) for row in payload.get("seed_outcomes", ()))
    mean_val = payload.get("mean")
    return CellResult(
        cell_sha=str(payload["cell_sha"]),
        alpha=float(payload["alpha"]),
        beta=float(payload["beta"]),
        gamma=float(payload["gamma"]),
        p_skills=float(payload["p_skills"]),
        n_ok=int(payload.get("n_ok", 0)),
        seed_outcomes=outcomes,
        mean_final_reward=(float("nan") if mean_val is None else float(mean_val)),
        ci95_halfwidth=float(payload.get("ci95", 0.0)),
    )


def _cellresult_from_seed_rows(cell_sha: str, rows: list[dict]) -> CellResult:
    """Reconstruct a :class:`CellResult` from per-(cell,seed) CSV rows."""
    first = rows[0]
    outcomes: list[dict] = []
    ok_rewards: list[float] = []
    for r in rows:
        try:
            reward = float(r["final_reward"])
        except (TypeError, ValueError):
            reward = float("nan")
        outcomes.append(
            {
                "seed": int(r["seed"]),
                "final_reward": reward,
                "status": str(r["status"]),
                "elapsed_s": float(r.get("elapsed_s", 0.0) or 0.0),
            }
        )
        if r["status"] == "ok" and math.isfinite(reward):
            ok_rewards.append(reward)
    m, ci = t_ci95(ok_rewards)
    return CellResult(
        cell_sha=cell_sha,
        alpha=float(first["alpha"]),
        beta=float(first["beta"]),
        gamma=float(first["gamma"]),
        p_skills=float(first["p_skills"]),
        n_ok=len(ok_rewards),
        seed_outcomes=tuple(outcomes),
        mean_final_reward=m,
        ci95_halfwidth=ci,
    )


def _parse_seed_table(csv_path: Path) -> list[CellResult]:
    """Bucket rows by ``cell_sha`` and synthesise one :class:`CellResult` each."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            buckets[row["cell_sha"]].append(row)
    return [_cellresult_from_seed_rows(sha, rs) for sha, rs in buckets.items()]


def _load_done_cells(output_dir: Path) -> list[CellResult]:
    """Prefer per-cell ``done.json`` (richer schema) over the flat seed_table."""
    cells: list[CellResult] = []
    for cell_dir in sorted(output_dir.glob("cell_*")):
        marker = cell_dir / "done.json"
        if not marker.exists():
            continue
        try:
            cells.append(_cellresult_from_done(json.loads(marker.read_text(encoding="utf-8"))))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return cells


def run_ablation(grid_name: str = "compact", output_dir: Path | str | None = None) -> Ablation:
    """SDK entry point — return a structured :class:`Ablation` for ``grid_name``.

    Resume-aware: if ``output_dir`` already contains either ``cell_*/done.json``
    markers or a flat ``seed_table.csv`` from a prior
    ``scripts/run_ablation.py`` invocation, this *parses* those artefacts and
    never spawns a subprocess (CLAUDE.md "no subprocess leak from the SDK").
    """
    t0 = time.monotonic()
    out_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    cells = _load_done_cells(out_dir) if out_dir.exists() else []
    if not cells:
        seed_table = out_dir / "seed_table.csv"
        if seed_table.exists():
            cells = _parse_seed_table(seed_table)
    cells_tuple = tuple(cells)
    total_runs = sum(len(c.seed_outcomes) for c in cells_tuple)
    n_ok_total = sum(c.n_ok for c in cells_tuple)
    return Ablation(
        grid_name=str(grid_name),
        cells=cells_tuple,
        total_runs=total_runs,
        n_ok_total=n_ok_total,
        total_wall_clock_s=time.monotonic() - t0,
    )
