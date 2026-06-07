"""Wave-3 Stream A helpers — grid expansion, cell hashing, t-CI, atomic done.json.

The ablation runner (``scripts/run_ablation.py``) sweeps the reward-coefficient
grid declared in ``config.yaml`` under ``ablation:``. Every (alpha, beta,
gamma, p_skills) point becomes one *cell*; each cell runs the scout seed list
through ``train_ppo.py`` and reports mean ± Student-t 95% CI of the per-seed
final reward. This module hosts the pure-function helpers so the runner stays
orchestration-only and the schema is independently testable.

Sealed schema (write-once per cell):
* ``done.json`` payload — cell coefficients, per-seed outcome rows, ``n_ok``,
  ``mean``, ``ci95``; serialised atomically via tmp-file + rename so a SIGINT
  mid-write can never leave a half-written marker on disk.
* ``cell_sha`` — stable 12-char sha256 of the deterministic-fields tuple, used
  as the directory name so re-running with ``--resume`` can skip completed
  cells without re-parsing every payload.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean as _mean
from statistics import stdev as _stdev

# Student-t two-sided 97.5% critical values, dof = 1..30 (then 1.96 for dof≥30).
# Hard-coded as a one-liner to keep _ablation_lib.py scipy-free at import time.
# fmt: off
_T_CRIT_975: dict[int, float] = dict(enumerate([
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
    2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
], start=1))
# fmt: on
_T_INF = 1.9600


@dataclass(frozen=True)
class Cell:
    """One ablation grid cell — sealed reward coefficients + scout seeds."""

    alpha: float
    beta: float
    gamma: float
    p_skills: float
    total_steps: int
    seed_list: tuple[int, ...] = field(default_factory=tuple)
    run_id: str = ""

    def deterministic_tuple(self) -> tuple:
        """Fields that participate in the SHA — excludes the human run_id label."""
        return (
            float(self.alpha),
            float(self.beta),
            float(self.gamma),
            float(self.p_skills),
            int(self.total_steps),
            tuple(int(s) for s in self.seed_list),
        )


def cell_sha(cell: Cell) -> str:
    """Stable 12-char sha256 of the cell's deterministic fields (round to 6dp)."""
    rounded = (
        round(cell.alpha, 6),
        round(cell.beta, 6),
        round(cell.gamma, 6),
        round(cell.p_skills, 6),
        int(cell.total_steps),
        tuple(int(s) for s in cell.seed_list),
    )
    blob = json.dumps(rounded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def make_grid(
    grid_dict: dict[str, Sequence[float]],
    seeds: Sequence[int],
    total_steps: int,
) -> list[Cell]:
    """Cartesian product over alpha x beta x gamma x p_skills (seeds attached per-cell).

    Raises KeyError if any of the four reward knobs is missing from ``grid_dict``
    so a typo in ``config.yaml`` fails loudly instead of silently shrinking the
    sweep.
    """
    for key in ("alpha", "beta", "gamma", "p_skills"):
        if key not in grid_dict:
            raise KeyError(f"grid_dict missing required knob: {key!r}")
    seed_tuple = tuple(int(s) for s in seeds)
    cells: list[Cell] = []
    for a, b, g, p in itertools.product(
        grid_dict["alpha"], grid_dict["beta"], grid_dict["gamma"], grid_dict["p_skills"]
    ):
        cells.append(
            Cell(
                alpha=float(a),
                beta=float(b),
                gamma=float(g),
                p_skills=float(p),
                total_steps=int(total_steps),
                seed_list=seed_tuple,
            )
        )
    return cells


_MIN_N_FOR_CI = 2  # need at least 2 observations to estimate sample stdev


def t_ci95(values: Iterable[float]) -> tuple[float, float]:
    """Return (mean, Student-t 95% half-width) with dof = n-1; (mean, 0.0) if n<2."""
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(finite)
    if n == 0:
        return (float("nan"), 0.0)
    m = _mean(finite)
    if n < _MIN_N_FOR_CI:
        return (m, 0.0)
    s = _stdev(finite)
    dof = n - 1
    t_crit = _T_CRIT_975.get(dof, _T_INF)
    half = t_crit * s / math.sqrt(n)
    return (m, half)


def cell_done(cell_dir: Path) -> bool:
    """True iff ``cell_dir/done.json`` exists AND is a non-empty parseable JSON object."""
    marker = Path(cell_dir) / "done.json"
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and len(payload) > 0


def mark_cell_done(cell_dir: Path, payload: dict) -> None:
    """Atomic done.json write: serialize to .tmp, fsync, rename to final path."""
    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    final = cell_dir / "done.json"
    tmp = cell_dir / "done.json.tmp"
    blob = json.dumps(payload, indent=2, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
