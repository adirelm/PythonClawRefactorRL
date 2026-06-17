"""Phase-4 COST-3: emit per-phase JSONL corpora for the TripleCounter (COST-4).

For each of the five project phases (0 = bootstrap through 4 = current),
walk ``git log`` and the optional verbatim prompt log at ``docs/PROMPTS.md``,
and write a JSONL file at ``results/cost/phase_<n>.jsonl`` with one record per
source unit.

Phase assignment is derived from the **commit-subject "Phase N" marker** (a
running counter carried forward across commits that don't restate the phase),
NOT from pinned commit SHAs. SHAs are mutable — a history rewrite (e.g. a PII
scrub) orphans any hardcoded hash and silently empties the corpora — whereas
the human-authored "Phase N —" subject convention survives rebases and
filter-repo runs. This keeps the script correct after history surgery.

Record schema (sealed for COST-4 downstream consumption):

* ``phase`` -- int, 0..4
* ``source`` -- "git" | "prompts_md"
* ``role`` -- "commit_subject" | "commit_body" | "prompts_md" | "prompt"
* ``text`` -- the verbatim text being counted (UTF-8 safe)
* ``timestamp`` -- ISO-8601 author date for git rows; file mtime for prompts
* ``sha`` -- 40-char commit hash for git rows; empty for prompt rows

The output is intentionally append-clean: re-running overwrites each phase
file in place so reruns stay deterministic.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "cost"
PROMPTS_MD = REPO_ROOT / "docs" / "PROMPTS.md"

NUM_PHASES = 5  # phases 0..4
_PHASE_RE = re.compile(r"phase[ \-]?([0-4])", re.IGNORECASE)


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


def _commits_oldest_first(git_range: str | None) -> list[str]:
    """Return commit SHAs (oldest-first) for ``git_range`` or all of HEAD.

    Degrades gracefully to ``[]`` if the range/repo is unreachable (shallow CI
    checkout, rewritten boundary, or a vanished ``.git``) so collection never
    hard-crashes — CI fetches full history; this keeps the script robust.
    """
    spec = [git_range] if git_range else ["HEAD"]
    try:
        out = subprocess.check_output(["git", "rev-list", "--reverse", *spec], cwd=REPO_ROOT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
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


def _phase_of(subject: str, running: int) -> int:
    """Phase from a ``Phase N`` subject marker; else carry ``running`` forward."""
    match = _PHASE_RE.search(subject)
    return min(NUM_PHASES - 1, int(match.group(1))) if match else running


def _git_records(git_range: str | None) -> dict[int, list[Record]]:
    """Bucket every commit into its phase. An explicit ``git_range`` is treated
    as a phase-4 override (matches the legacy ``--git-range`` contract)."""
    buckets: dict[int, list[Record]] = {p: [] for p in range(NUM_PHASES)}
    running = 0
    for sha in _commits_oldest_first(git_range):
        ts, subject, body = _commit_detail(sha)
        phase = 4 if git_range else _phase_of(subject, running)
        running = phase
        if subject:
            buckets[phase].append(Record(phase, "git", "commit_subject", subject, ts, sha))
        if body:
            buckets[phase].append(Record(phase, "git", "commit_body", body, ts, sha))
    return buckets


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompts-md", type=Path, default=PROMPTS_MD)
    parser.add_argument(
        "--git-range",
        type=str,
        default=None,
        help="Override: a 'START..END' range collected wholesale into phase 4.",
    )
    args = parser.parse_args(argv)

    buckets = _git_records(args.git_range)
    sizes: dict[int, int] = {}
    for phase in range(NUM_PHASES):
        records = buckets[phase] + collect_prompts_md(phase, args.prompts_md)
        out_path = Path(args.output_dir) / f"phase_{phase}.jsonl"
        write_jsonl(records, out_path)
        sizes[phase] = len(records)
        print(f"phase {phase}: {len(records)} records -> {out_path}")

    print(json.dumps({"sizes": sizes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
