"""Unit tests for ``scripts/compute_cost.py`` (Phase-4 COST-4).

Locks the ADR-003a 15-column schema, per-phase emission, and the Hebrew
non-ASCII regime that ADR-003 cites as the whole reason for triple-counting.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("tiktoken")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_cost.py"
_SPEC = importlib.util.spec_from_file_location("compute_cost", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
compute_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["compute_cost"] = compute_mod
_SPEC.loader.exec_module(compute_mod)

EXPECTED_SCHEMA = (
    "phase", "model", "input_tokens", "output_tokens", "chars_in",
    "chars_out", "bytes_in", "bytes_out", "wall_clock_sec", "episodes",
    "price_in_per_M", "price_out_per_M", "price_timestamp_iso",
    "subtotal_usd", "run_id",
)  # fmt: skip


def _write(in_dir: Path, phase: int, records: list[dict]) -> Path:
    in_dir.mkdir(parents=True, exist_ok=True)
    path = in_dir / f"phase_{phase}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", "utf-8")
    return path


def _rec(role: str, text: str, day: int, sha: str = "a" * 40) -> dict:
    return {"role": role, "text": text, "timestamp": f"2026-06-0{day}T00:00:00+00:00", "sha": sha}


def _seed_full(in_dir: Path) -> None:
    for p in range(5):
        _write(
            in_dir,
            p,
            [
                _rec("commit_subject", f"phase {p} subject", p + 1, f"{p}" * 40),
                _rec("commit_body", "body lines\nmore text", p + 1, f"{p}" * 40),
            ],
        )


def _run(in_dir: Path, out: Path, model: str = "claude-opus-4-7") -> int:
    return compute_mod.main(["--input-dir", str(in_dir), "--output", str(out), "--model", model])


def _rows(out: Path) -> list[dict]:
    with out.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_15_column_schema_locked(tmp_path: Path) -> None:
    assert EXPECTED_SCHEMA == compute_mod.SCHEMA
    assert len(compute_mod.SCHEMA) == 15
    _seed_full(tmp_path / "in")
    out = tmp_path / "cost.csv"
    assert _run(tmp_path / "in", out) == 0
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == EXPECTED_SCHEMA


def test_per_phase_row_emitted(tmp_path: Path) -> None:
    """5 phases -> 5 data rows; phase 3 episodes=6 (PPO budget), else 0."""
    _seed_full(tmp_path / "in")
    out = tmp_path / "cost.csv"
    assert _run(tmp_path / "in", out) == 0
    rows = _rows(out)
    assert len(rows) == 5
    assert sorted(int(r["phase"]) for r in rows) == [0, 1, 2, 3, 4]
    assert rows[3]["episodes"] == "6"
    assert all(r["episodes"] == "0" for r in (rows[0], rows[1], rows[2], rows[4]))


def test_hebrew_unicode_handled(tmp_path: Path) -> None:
    """Hebrew text -> bytes_out > chars_out (UTF-8 multi-byte), no crash."""
    _write(tmp_path / "in", 0, [_rec("commit_subject", "שלום עולם — בדיקת יוניקוד עברית", 1)])
    for p in (1, 2, 3, 4):
        _write(tmp_path / "in", p, [])
    out = tmp_path / "cost.csv"
    assert _run(tmp_path / "in", out) == 0
    row0 = _rows(out)[0]
    assert int(row0["bytes_out"]) > int(row0["chars_out"]) > 0
    assert int(row0["output_tokens"]) > 0


def test_pricing_timestamp_iso_format() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", compute_mod.PRICE_TIMESTAMP_ISO)


def test_subtotal_math(tmp_path: Path) -> None:
    """subtotal_usd == input/1e6 * price_in + output/1e6 * price_out."""
    _seed_full(tmp_path / "in")
    out = tmp_path / "cost.csv"
    assert _run(tmp_path / "in", out) == 0
    for row in _rows(out):
        inp, outp = int(row["input_tokens"]), int(row["output_tokens"])
        p_in, p_out = float(row["price_in_per_M"]), float(row["price_out_per_M"])
        expected = round(inp / 1e6 * p_in + outp / 1e6 * p_out, 6)
        assert abs(float(row["subtotal_usd"]) - expected) < 1e-6


def test_resume_when_input_changes(tmp_path: Path) -> None:
    """Modified JSONL regenerates rows (no stale cache)."""
    _seed_full(tmp_path / "in")
    out = tmp_path / "cost.csv"
    assert _run(tmp_path / "in", out) == 0
    first_tokens = int(_rows(out)[0]["output_tokens"])
    _write(tmp_path / "in", 0, [_rec("commit_subject", "vastly longer subject " * 50, 1, "z" * 40)])
    assert _run(tmp_path / "in", out) == 0
    assert int(_rows(out)[0]["output_tokens"]) != first_tokens


def test_input_role_routes_to_input_side(tmp_path: Path) -> None:
    """role=prompt populates input_* cols; commit_* still populates output."""
    _write(
        tmp_path / "in",
        0,
        [
            _rec("prompt", "user input prompt body", 1, ""),
            _rec("commit_subject", "assistant output", 1, "f" * 40),
        ],
    )
    for p in (1, 2, 3, 4):
        _write(tmp_path / "in", p, [])
    out = tmp_path / "cost.csv"
    assert _run(tmp_path / "in", out) == 0
    row0 = _rows(out)[0]
    assert int(row0["input_tokens"]) > 0
    assert int(row0["output_tokens"]) > 0


def test_sonnet_model_pricing(tmp_path: Path) -> None:
    """Sonnet model picks up cheaper $3/$15 rates per ADR-003a snapshot."""
    _seed_full(tmp_path / "in")
    out = tmp_path / "cost.csv"
    assert _run(tmp_path / "in", out, model="claude-sonnet-4-5") == 0
    row0 = _rows(out)[0]
    assert float(row0["price_in_per_M"]) == 3.0
    assert float(row0["price_out_per_M"]) == 15.0
