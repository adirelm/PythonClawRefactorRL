"""Phase-4 COST-3: emit per-phase JSONL corpora for the TripleCounter (COST-4).

For each of the five project phases (0 = bootstrap through 4 = current),
walk ``git log`` for the phase's commit range plus the optional verbatim
prompt log at ``docs/PROMPTS.md``, and write a JSONL file at
``results/cost/phase_<n>.jsonl`` with one record per source unit.

Record schema (sealed for COST-4 downstream consumption):

* ``phase`` -- int, 0..4
* ``source`` -- "git" | "prompts_md"
* ``role`` -- "commit_subject" | "commit_body" | "prompts_md" | "prompt"
* ``text`` -- the verbatim text being counted (UTF-8 safe)
* ``timestamp`` -- ISO-8601 author date for git rows; file mtime for prompts
* ``sha`` -- 40-char commit hash for git rows; empty for prompt rows

The output is intentionally append-clean: re-running overwrites each phase
file in place so reruns stay deterministic. COST-4 will stream these files
through ``TripleCounter`` and aggregate into ``cost_table.csv`` per
ADR-003a's 15-column schema.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "cost"
PROMPTS_MD = REPO_ROOT / "docs" / "PROMPTS.md"

# Sealed phase boundaries (end_sha is the last commit of the phase, inclusive).
# Phase 0 starts at the root commit (no parent), so we special-case it below.
PHASE_RANGES: list[tuple[int, str, str]] = [
    (0, "ROOT", "3b0ed5b2e96392b2d56fea3621c6ac64b88f9deb"),  # bootstrap..gate-fix
    (1, "0165fa2925228558fa23e658c72d7da5d1d19dea", "9660830f97d64a8a5903ab063805b81359f711e8"),
    (2, "ec1288a05e059c6602a3b2fa22834dee8bee9a23", "8264a84c3b6a191eecfec1c0cb804bfc202e0e7f"),
    (3, "71f0213653fdc5ea56b6eb88062dcac9478851e7", "1bb2d8f2b8b23ec84ca59048d9abde418f54a7a9"),
    (4, "dbdd1a5021f2928a03751b8fcb3db62c0623bd26", "HEAD"),
]


@dataclass(frozen=True)
class Record:
    phase: int
    source: str
    role: str
    text: str
    timestamp: str
    sha: str

    def as_jsonl(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def _git_rev_list(start_sha: str, end_sha: str) -> list[str]:
    """Return commit SHAs in phase range, oldest-first.

    Degrades gracefully to ``[]`` if a boundary SHA is unreachable (e.g. a
    shallow CI checkout without ``fetch-depth: 0``, or a rewritten history) so
    corpus collection never hard-crashes — CI fetches full history, but this
    keeps the script robust everywhere.
    """
    if start_sha == "ROOT":
        spec = [end_sha]  # all commits reachable from end_sha, including the root
    else:
        spec = [f"{start_sha}^..{end_sha}"]
    try:
        out = subprocess.check_output(["git", "rev-list", "--reverse", *spec], cwd=REPO_ROOT, text=True)
    except subprocess.CalledProcessError:
        return []
    return [s for s in out.splitlines() if s]


def _commit_detail(sha: str) -> tuple[str, str, str]:
    """Return (iso_timestamp, subject, body) for ``sha``."""
    out = subprocess.check_output(
        ["git", "show", "--no-patch", "--format=%aI%n%s%n%b", sha],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    )
    lines = out.splitlines()
    ts = lines[0] if lines else ""
    subject = lines[1] if len(lines) > 1 else ""
    body = "\n".join(lines[2:]).rstrip("\n")
    return ts, subject, body


def collect_phase(phase: int, start_sha: str, end_sha: str) -> list[Record]:
    """Build the git-side records for one phase."""
    records: list[Record] = []
    for sha in _git_rev_list(start_sha, end_sha):
        ts, subject, body = _commit_detail(sha)
        if subject:
            records.append(Record(phase, "git", "commit_subject", subject, ts, sha))
        if body:
            records.append(Record(phase, "git", "commit_body", body, ts, sha))
    return records


def collect_prompts_md(phase: int, prompts_path: Path) -> list[Record]:
    """Emit a single ``prompts_md`` record if the file exists; else empty."""
    if not prompts_path.exists():
        return []
    text = prompts_path.read_text(encoding="utf-8")
    if not text:
        return []
    mtime = datetime.fromtimestamp(prompts_path.stat().st_mtime, tz=UTC).isoformat()
    return [Record(phase, "prompts_md", "prompts_md", text, mtime, "")]


def write_jsonl(records: list[Record], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.as_jsonl())
            fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for phase_<n>.jsonl files (default: results/cost/)",
    )
    parser.add_argument(
        "--prompts-md",
        type=Path,
        default=PROMPTS_MD,
        help="Path to docs/PROMPTS.md (skipped silently if absent)",
    )
    parser.add_argument(
        "--git-range",
        type=str,
        default=None,
        help="Override: single 'START..END' range applied to phase 4 only.",
    )
    args = parser.parse_args(argv)

    phase_ranges = list(PHASE_RANGES)
    if args.git_range:
        start, _, end = args.git_range.partition("..")
        phase_ranges[-1] = (4, start or "ROOT", end or "HEAD")

    sizes: dict[int, int] = {}
    for phase, start_sha, end_sha in phase_ranges:
        records = collect_phase(phase, start_sha, end_sha)
        records += collect_prompts_md(phase, args.prompts_md)
        out_path = Path(args.output_dir) / f"phase_{phase}.jsonl"
        write_jsonl(records, out_path)
        sizes[phase] = len(records)
        print(f"phase {phase}: {len(records)} records -> {out_path}")

    print(json.dumps({"sizes": sizes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
