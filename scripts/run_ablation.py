#!/usr/bin/env -S uv run python
"""Wave-3 Stream A: ablation runner — sweep reward-coefficient grid via train_ppo.

For each cell in the configured grid (``compact`` by default, ``smoke`` for a
single 1x1x1x1 sanity point), spawn ``scripts/train_ppo.py`` per scout seed
under a per-seed wall-clock timeout, collect per-seed final reward from
``aggregate.json``, and write an atomic ``done.json`` marker plus one row per
(cell, seed) into ``results/ablations/seed_table.csv``. ``--resume`` (default
on) skips any cell whose ``done.json`` is already present.

Pure orchestration — all schema / stats / hashing logic lives in
:mod:`scripts._ablation_lib` so this file stays subprocess-glue only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._ablation_lib import Cell, cell_done, cell_sha, make_grid, mark_cell_done, t_ci95  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "ablations"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_ppo.py"
logger = logging.getLogger("run_ablation")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation runner — sweep alpha/beta/gamma/P_skills.")
    p.add_argument("--grid", choices=["compact", "smoke"], default="compact")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Override scout_seeds")
    p.add_argument("--max-cells", type=int, default=None, help="Debug stop after N cells")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--force", action="store_true", help="Force re-run even if done.json exists")
    return p.parse_args(argv)


def _build_cells(args: argparse.Namespace, cfg: dict) -> list[Cell]:
    ab = cfg["ablation"]
    grid_dict = ab["grids"][args.grid]
    seeds = args.seeds if args.seeds is not None else ab["scout_seeds"]
    total_steps = int(ab["total_steps_per_cell_seed"])
    cells = make_grid(grid_dict, seeds, total_steps)
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    return cells


def _nan_row(seed: int, status: str, elapsed: float) -> dict:
    """Shorthand for a non-OK seed outcome row (NaN reward + status reason)."""
    return {"seed": seed, "final_reward": float("nan"), "status": status, "elapsed_s": elapsed}


def _build_cmd(cell: Cell, seed: int, out_dir: Path) -> list[str]:
    """Argv for one train_ppo.py invocation pinned to this (cell, seed)."""
    # fmt: off
    return [
        sys.executable, str(TRAIN_SCRIPT),
        "--seeds", str(seed), "--total-steps", str(int(cell.total_steps)),
        "--output-dir", str(out_dir),
        "--alpha", str(cell.alpha), "--beta", str(cell.beta),
        "--gamma", str(cell.gamma), "--p-skills", str(cell.p_skills),
    ]
    # fmt: on


def _run_seed(cell: Cell, seed: int, timeout_s: int) -> dict:
    """Spawn train_ppo.py for one (cell, seed); return outcome row."""
    with tempfile.TemporaryDirectory(prefix="abl_") as tmp:
        out_dir = Path(tmp) / "training"
        cmd = _build_cmd(cell, seed, out_dir)
        t0 = time.monotonic()
        try:
            done = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            return _nan_row(seed, "timeout", float(timeout_s))
        elapsed = time.monotonic() - t0
        if done.returncode != 0:
            return _nan_row(seed, "fail", elapsed)
        agg_path = out_dir / "aggregate.json"
        if not agg_path.exists():
            return _nan_row(seed, "no_aggregate", elapsed)
        payload = json.loads(agg_path.read_text(encoding="utf-8"))
        final = float(payload["per_seed_final_reward"][str(seed)])
        return {"seed": seed, "final_reward": final, "status": "ok", "elapsed_s": elapsed}


def _run_cell(cell: Cell, output_dir: Path, per_seed_timeout_s: int) -> dict:
    """Run all seeds for one cell; serialize done.json with stats."""
    sha = cell_sha(cell)
    cell_dir = output_dir / f"cell_{sha}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict] = []
    for seed in cell.seed_list:
        outcomes.append(_run_seed(cell, int(seed), per_seed_timeout_s))
    ok_rewards = [row["final_reward"] for row in outcomes if row["status"] == "ok"]
    m, ci = t_ci95(ok_rewards)
    # fmt: off
    payload = {
        "alpha": cell.alpha, "beta": cell.beta, "gamma": cell.gamma, "p_skills": cell.p_skills,
        "total_steps": cell.total_steps, "cell_sha": sha,
        "seed_outcomes": outcomes, "n_ok": len(ok_rewards),
        "mean": (None if math.isnan(m) else m), "ci95": ci,  # JSON has no NaN; null on empty
    }
    # fmt: on
    mark_cell_done(cell_dir, payload)
    return payload


def _append_seed_table(rows: list[dict], output_dir: Path) -> None:
    """One row per (cell, seed) — appended atomically (rewrite each invocation)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "seed_table.csv"
    cols = ["cell_sha", "alpha", "beta", "gamma", "p_skills", "seed", "final_reward", "status", "elapsed_s"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    cfg = load_config()
    per_seed_timeout_s = int(cfg["ablation"]["per_seed_timeout_s"])
    cells = _build_cells(args, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict] = []
    total = len(cells)
    for idx, cell in enumerate(cells, start=1):
        sha = cell_sha(cell)
        cell_dir = args.output_dir / f"cell_{sha}"
        if args.resume and not args.force and cell_done(cell_dir):
            payload = json.loads((cell_dir / "done.json").read_text(encoding="utf-8"))
            verb = "SKIP (resume)"
        else:
            payload = _run_cell(cell, args.output_dir, per_seed_timeout_s)
            verb = "DONE"
        logger.info(
            "cell %d/%d sha=%s %s n_ok=%s mean=%s",
            idx,
            total,
            sha,
            verb,
            payload.get("n_ok"),
            payload.get("mean"),
        )
        # fmt: off
        for row in payload.get("seed_outcomes", []):
            flat_rows.append({
                "cell_sha": sha, "alpha": cell.alpha, "beta": cell.beta,
                "gamma": cell.gamma, "p_skills": cell.p_skills,
                "seed": row["seed"], "final_reward": row["final_reward"],
                "status": row["status"], "elapsed_s": row["elapsed_s"],
            })
        # fmt: on
    _append_seed_table(flat_rows, args.output_dir)
    print(f"cells_total={total} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
