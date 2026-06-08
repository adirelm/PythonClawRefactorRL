#!/usr/bin/env -S uv run python
"""Fetch the real PythonClaw source under analysis, pinned to a fixed commit.

The assignment analyses the **official PythonClaw** platform
(https://github.com/ericwang915/PythonClaw) — the Python port of OpenClaw,
published on PyPI as ``pythonclaw``. We clone it at a pinned SHA into the
gitignored ``vendor/`` directory so GRAPHIFY, the bug report, and the RL
training all run against the real upstream Skills subsystem reproducibly,
while keeping the third-party source out of our own lint/size gates.

Usage::

    uv run python scripts/fetch_pythonclaw.py        # clone/checkout pinned SHA
    uv run python scripts/fetch_pythonclaw.py --sha <other>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_URL = "https://github.com/ericwang915/PythonClaw.git"
PINNED_SHA = "7787bb433590ca2d1ee27b99819d68ad8fc3efd2"  # v0.6.6 (2026-03-08)
DEST = REPO_ROOT / "vendor" / "pythonclaw"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch pinned PythonClaw source into vendor/.")
    p.add_argument("--sha", default=PINNED_SHA, help="commit SHA to check out")
    p.add_argument("--url", default=UPSTREAM_URL)
    args = p.parse_args(argv)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not (DEST / ".git").exists():
        print(f"Cloning {args.url} → {DEST} …")
        _run(["git", "clone", args.url, str(DEST)])
    print(f"Checking out pinned SHA {args.sha} …")
    _run(["git", "fetch", "origin", args.sha], cwd=DEST)
    _run(["git", "checkout", "-q", args.sha], cwd=DEST)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=DEST, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head == args.sha, f"checkout mismatch: {head} != {args.sha}"
    skills = DEST / "pythonclaw"
    n_py = len(list(skills.rglob("*.py")))
    print(f"✅ PythonClaw at {head[:12]} — {n_py} .py files under {skills.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
