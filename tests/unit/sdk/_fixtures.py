"""Tmp-dir fixtures for SDK ablation tests (split to keep test files ≤150 LOC)."""

from __future__ import annotations

import json
from pathlib import Path

DONE_OK = {
    "alpha": 1.0,
    "beta": 1.0,
    "cell_sha": "abc123def456",
    "ci95": 0.05,
    "gamma": 0.5,
    "mean": 0.42,
    "n_ok": 2,
    "p_skills": -5.0,
    "seed_outcomes": [
        {"elapsed_s": 12.0, "final_reward": 0.4, "seed": 42, "status": "ok"},
        {"elapsed_s": 13.0, "final_reward": 0.44, "seed": 7, "status": "ok"},
    ],
    "total_steps": 256,
}

_CSV_HEADER = "cell_sha,alpha,beta,gamma,p_skills,seed,final_reward,status,elapsed_s\n"


def write_cell(out_dir: Path, payload: dict) -> Path:
    """Write one ``cell_<sha>/done.json`` under ``out_dir`` and return the dir."""
    cell_dir = out_dir / f"cell_{payload['cell_sha']}"
    cell_dir.mkdir(parents=True)
    (cell_dir / "done.json").write_text(json.dumps(payload), encoding="utf-8")
    return cell_dir


def seed_row(cell_sha: str, seed: int, reward, status: str, elapsed: float = 10.0) -> dict:
    """One canonical-coeff seed row for ``write_seed_table`` ingestion."""
    return {
        "cell_sha": cell_sha,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 0.5,
        "p_skills": -5.0,
        "seed": seed,
        "final_reward": reward,
        "status": status,
        "elapsed_s": elapsed,
    }


def write_seed_table(out_dir: Path, rows: list[dict]) -> Path:
    """Write a flat ``seed_table.csv`` from a list of seed rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "seed_table.csv"
    lines = [_CSV_HEADER]
    for r in rows:
        lines.append(
            f"{r['cell_sha']},{r['alpha']},{r['beta']},{r['gamma']},"
            f"{r['p_skills']},{r['seed']},{r['final_reward']},{r['status']},{r['elapsed_s']}\n"
        )
    csv_path.write_text("".join(lines), encoding="utf-8")
    return csv_path
