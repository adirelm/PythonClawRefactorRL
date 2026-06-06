"""TRACE/TODO freshness enforcer.

``--check`` (CI gate): scans TRACE.md + TODO.md for ``<phaseN-commit>``
placeholders; exits 1 if any remain. Path misses are reported but do
not flip the exit code (planned ⬜ rows legitimately cite future paths).

``--fix``: rewrites every ``<phaseN-commit>`` with HEAD short SHA.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGETS = (_REPO_ROOT / "docs" / "TRACE.md", _REPO_ROOT / "docs" / "TODO.md")
_PLACEHOLDER_RE = re.compile(r"<phase\d+-commit>")
# Repo-relative paths inside backticks. Anchored to known top-level dirs
# so we skip §-ids, regex tokens, and bare identifiers.
_PATH_PREFIXES = ("src/", "tests/", "scripts/", "docs/", "config/", "results/", "notebooks/", ".github/")
_PATH_RE = re.compile(r"`([^`\s]+)`")


def _head_short_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return out.stdout.strip()


def _find_placeholders(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        for match in _PLACEHOLDER_RE.finditer(line):
            hits.append((i, match.group(0)))
    return hits


def _looks_like_path(token: str) -> bool:
    if not token.startswith(_PATH_PREFIXES):
        return False
    # Skip glob/brace expansions and pure markdown fragments.
    return "{" not in token and "*" not in token and "#" not in token


def _find_path_misses(path: Path) -> list[tuple[int, str]]:
    misses: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        for match in _PATH_RE.finditer(line):
            token = match.group(1).rstrip(".,;:)")
            if not _looks_like_path(token):
                continue
            if not (_REPO_ROOT / token).exists():
                misses.append((i, token))
    return misses


def _report(label: str, file_path: Path, hits: list[tuple[int, str]]) -> None:
    if not hits:
        return
    rel = file_path.relative_to(_REPO_ROOT)
    for line_no, token in hits:
        print(f"{label} {rel}:{line_no}: {token}")


def _check() -> int:
    placeholder_total = 0
    for target in _TARGETS:
        hits = _find_placeholders(target)
        _report("DRIFT", target, hits)
        placeholder_total += len(hits)
        misses = _find_path_misses(target)
        _report("MISS ", target, misses)
    if placeholder_total:
        print(f"\nstamp_trace: {placeholder_total} placeholder(s) remain — exit 1")
        return 1
    print("stamp_trace: clean (no <phaseN-commit> placeholders)")
    return 0


def _fix() -> int:
    sha = _head_short_sha()
    replaced = 0
    for target in _TARGETS:
        text = target.read_text()
        new_text, count = _PLACEHOLDER_RE.subn(sha, text)
        if count:
            target.write_text(new_text)
            replaced += count
            print(f"stamped {count} placeholder(s) in {target.relative_to(_REPO_ROOT)} → {sha}")
        misses = _find_path_misses(target)
        _report("MISS ", target, misses)
    if replaced == 0:
        print("stamp_trace: nothing to stamp (no placeholders found)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRACE/TODO commit-SHA enforcer.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Report drift; exit 1 if any.")
    mode.add_argument("--fix", action="store_true", help="Stamp <phaseN-commit> with HEAD short SHA.")
    args = parser.parse_args(argv)
    return _check() if args.check else _fix()


if __name__ == "__main__":
    sys.exit(main())
