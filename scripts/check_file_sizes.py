"""Fail if any committed .py file exceeds 150 lines of code (CLAUDE.md §1).

Counts lines, excluding blank lines and pure-comment lines, so docstrings still count
but `# just a divider` separators don't. Returns exit 1 if any file is over.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT = 150
EXCLUDED_DIRS = {".venv", ".git", "build", "dist", "__pycache__", ".ruff_cache", ".pytest_cache"}


def count_loc(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    loc = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        loc += 1
    return loc


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    over: list[tuple[Path, int]] = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        loc = count_loc(path)
        if loc > LIMIT:
            over.append((path.relative_to(root), loc))
    if over:
        print(f"❌ {len(over)} file(s) exceed {LIMIT} LOC:")
        for path, loc in over:
            print(f"  {path}: {loc} LOC")
        return 1
    print(f"✅ all .py files ≤ {LIMIT} LOC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
