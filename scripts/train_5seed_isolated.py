#!/usr/bin/env -S uv run python
"""Phase-3 5-seed PPO driver — one fresh subprocess per seed, hard wall-clock
budget enforced via :class:`subprocess.run(timeout=...)`. Survives per-seed
hangs (some seeds enter a still-undiagnosed slow path in PPO iter 2+) by
recording PARTIAL and moving on; rebuilds ``aggregate.json`` from whichever
``seed_*/metrics.json`` made it to disk.

Used after R1's NOOP-pin fix landed: standalone seed runs are fast (~12-20 s
at total-steps 256) but some seeds still wedge on a long path in iter 2+,
so each seed is process-isolated AND time-boxed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [42, 7, 123, 314, 271]
DEFAULT_TIMEOUT_S = 120
DEFAULT_TOTAL_STEPS = 256
OUTPUT_DIR = REPO_ROOT / "results" / "training"
_MIN_REWARD_FIELDS = 2  # rewards.csv schema: step,reward


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolated 5-seed PPO driver with per-seed timeout.")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--per-seed-timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    return p.parse_args()


def _run_one_seed(seed: int, total_steps: int, timeout_s: int) -> dict:
    """Spawn fresh `train_ppo.py --seeds N`, time-boxed; report status row."""
    log_path = Path("/tmp") / f"a4_5seed_iso_seed_{seed}.log"
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "train_ppo.py"),
        "--seeds", str(seed),
        "--total-steps", str(total_steps),
    ]
    t0 = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s, cwd=REPO_ROOT, check=False)
        elapsed = time.perf_counter() - t0
        return {"seed": seed, "status": "OK" if proc.returncode == 0 else f"FAIL_rc={proc.returncode}", "elapsed_s": round(elapsed, 2), "log": str(log_path)}
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return {"seed": seed, "status": "TIMEOUT", "elapsed_s": round(elapsed, 2), "log": str(log_path)}


def _read_final_reward(seed_dir: Path) -> float:
    """Sum rewards.csv reward column (matches train_ppo._final_reward semantics)."""
    rewards_csv = seed_dir / "rewards.csv"
    if not rewards_csv.exists():
        return float("nan")
    total = 0.0
    n = 0
    with rewards_csv.open(encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) >= _MIN_REWARD_FIELDS:
                try:
                    total += float(parts[1])
                    n += 1
                except ValueError:
                    continue
    return total if n > 0 else float("nan")


def _rebuild_aggregate(seeds: list[int], total_steps: int) -> dict:
    """Walk every seed_*/ on disk; write aggregate.json with whatever survived."""
    rows: list[dict] = []
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed_{seed}"
        if not (seed_dir / "metrics.json").exists():
            continue
        rows.append({"seed": seed, "final_reward": _read_final_reward(seed_dir)})
    finals = [r["final_reward"] for r in rows if not math.isnan(r["final_reward"])]
    aggregate = {
        "seeds": [r["seed"] for r in rows],
        "per_seed_final_reward": {str(r["seed"]): r["final_reward"] for r in rows},
        "mean_final_reward": float(mean(finals)) if finals else float("nan"),
        "std_final_reward": float(pstdev(finals)) if len(finals) > 1 else 0.0,
        "total_steps_per_seed": int(total_steps),
        "num_seeds": len(rows),
        "attempted_seeds": list(seeds),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def main() -> int:
    args = _parse_args()
    print(f"=== Phase-3 5-seed isolated PPO ({len(args.seeds)} seeds x {args.total_steps} steps, {args.per_seed_timeout_s}s/seed budget) ===")
    statuses: list[dict] = []
    for seed in args.seeds:
        print(f"[{time.strftime('%H:%M:%S')}] seed={seed} starting (timeout={args.per_seed_timeout_s}s)...", flush=True)
        row = _run_one_seed(seed, args.total_steps, args.per_seed_timeout_s)
        statuses.append(row)
        print(f"[{time.strftime('%H:%M:%S')}] seed={seed} {row['status']} elapsed={row['elapsed_s']}s log={row['log']}", flush=True)
    aggregate = _rebuild_aggregate(args.seeds, args.total_steps)
    print("\n=== Per-seed status ===")
    for r in statuses:
        print(f"  seed={r['seed']:4d} {r['status']:12s} elapsed={r['elapsed_s']}s")
    print("\n=== aggregate.json ===")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
