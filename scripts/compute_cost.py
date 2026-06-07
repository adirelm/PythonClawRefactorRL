"""COST-4: TripleCount per-phase JSONL -> 15-col cost_table.csv (ADR-003a).

Role split: prompt/prompts_md -> input, commit_subject/commit_body -> output.
Pricing snapshot https://www.anthropic.com/pricing 2026-06-07: Opus 4.x
$15/$75 per M input/output; Sonnet 4.x $3/$15.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.cost.meter import Counts, TripleCounter

PRICE_TIMESTAMP_ISO = "2026-06-07T00:00:00Z"
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
}
INPUT_ROLES = frozenset({"prompt", "prompts_md"})
OUTPUT_ROLES = frozenset({"commit_subject", "commit_body"})
TRAIN_PHASE = 3  # only PPO-training phase carries an episode budget
PPO_EPISODES = 6  # 3 OK seeds x 2 iters/seed (RC-5 outcome)

# Sealed 15-column ADR-003a schema (column order is part of the contract).
SCHEMA: tuple[str, ...] = (
    "phase", "model", "input_tokens", "output_tokens", "chars_in",
    "chars_out", "bytes_in", "bytes_out", "wall_clock_sec", "episodes",
    "price_in_per_M", "price_out_per_M", "price_timestamp_iso",
    "subtotal_usd", "run_id",
)  # fmt: skip

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "cost"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "cost_table.csv"


@dataclass
class PhaseAgg:
    phase: int
    in_: Counts = field(default_factory=lambda: Counts(0, 0, 0))
    out: Counts = field(default_factory=lambda: Counts(0, 0, 0))
    first_ts: str = ""
    last_ts: str = ""
    first_sha: str = ""

    def add(self, c: Counts, side: str) -> None:
        cur = self.in_ if side == "in" else self.out
        new = Counts(cur.tokens + c.tokens, cur.chars + c.chars, cur.bytes + c.bytes)
        if side == "in":
            self.in_ = new
        else:
            self.out = new


def _wall_clock(first_ts: str, last_ts: str) -> float:
    if not first_ts or not last_ts:
        return 0.0
    return (datetime.fromisoformat(last_ts) - datetime.fromisoformat(first_ts)).total_seconds()


def _episodes_for(phase: int) -> int:
    """Only phase 3 (PPO) carries the episode budget; everything else is 0."""
    return PPO_EPISODES if phase == TRAIN_PHASE else 0


def aggregate_phase(jsonl_path: Path, counter: TripleCounter) -> PhaseAgg:
    """Stream a phase JSONL and triple-count input/output sides."""
    agg = PhaseAgg(phase=int(jsonl_path.stem.split("_")[-1]))
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        role, text = rec.get("role", ""), rec.get("text", "")
        if not text:
            continue
        c = counter.count(text)
        if role in INPUT_ROLES:
            agg.add(c, "in")
        elif role in OUTPUT_ROLES:
            agg.add(c, "out")
        ts = rec.get("timestamp", "")
        if ts and (not agg.first_ts or ts < agg.first_ts):
            agg.first_ts = ts
            agg.first_sha = rec.get("sha", "") or agg.first_sha
        if ts and ts > agg.last_ts:
            agg.last_ts = ts
    return agg


def build_row(agg: PhaseAgg, model: str) -> dict[str, object]:
    """Materialise one CSV row for ``agg`` per ADR-003a column order."""
    price_in, price_out = PRICING.get(model, (15.0, 75.0))
    subtotal = (agg.in_.tokens / 1e6) * price_in + (agg.out.tokens / 1e6) * price_out
    sha7 = (agg.first_sha or "0000000")[:7]
    return dict(zip(SCHEMA, (
        agg.phase, model, agg.in_.tokens, agg.out.tokens,
        agg.in_.chars, agg.out.chars, agg.in_.bytes, agg.out.bytes,
        round(_wall_clock(agg.first_ts, agg.last_ts), 3),
        _episodes_for(agg.phase), price_in, price_out, PRICE_TIMESTAMP_ISO,
        round(subtotal, 6), f"phase{agg.phase}-{sha7}",
    ), strict=True))  # fmt: skip


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCHEMA))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default="claude-opus-4-7")
    args = parser.parse_args(argv)

    counter = TripleCounter()
    rows: list[dict[str, object]] = []
    total_usd, per_phase = 0.0, {}
    for jsonl_path in sorted(Path(args.input_dir).glob("phase_*.jsonl")):
        agg = aggregate_phase(jsonl_path, counter)
        row = build_row(agg, args.model)
        rows.append(row)
        total_usd += float(row["subtotal_usd"])
        per_phase[agg.phase] = float(row["subtotal_usd"])

    write_csv(rows, args.output)
    summary = {
        "rows": len(rows),
        "total_usd": round(total_usd, 4),
        "per_phase_usd": per_phase,
        "output": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
