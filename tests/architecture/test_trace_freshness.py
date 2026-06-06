"""TRACE.md / TODO.md freshness gate (Phase 0 ruff-fix follow-up).

The validation report flagged this CI gate as missing. The gate runs
``scripts/stamp_trace.py --check`` as a subprocess and asserts a clean
exit. A non-zero exit indicates ``<phaseN-commit>`` placeholders are
still present somewhere in TRACE.md or TODO.md — meaning a phase landed
without its short SHA being stamped in.

Path misses (cited files that do not exist on disk) are reported by
stamp_trace but are *not* part of the exit-code contract: planned ⬜
rows in TRACE legitimately cite future artefacts (see the "Known gaps"
section). Drift on commit SHAs is the only hard failure here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_SCRIPT = REPO_ROOT / "scripts" / "stamp_trace.py"


def test_stamp_trace_check_exits_clean() -> None:
    """``stamp_trace.py --check`` must exit 0 — no SHA drift in docs."""
    assert STAMP_SCRIPT.exists(), f"missing enforcer at {STAMP_SCRIPT}"
    result = subprocess.run(
        [sys.executable, str(STAMP_SCRIPT), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "stamp_trace --check reported drift; stamp <phaseN-commit> "
        f"placeholders with the real SHA.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
