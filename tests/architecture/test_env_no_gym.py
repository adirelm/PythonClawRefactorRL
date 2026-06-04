"""Architectural contract: NO gymnasium imports under src/env/ or src/services/.

Brief §2.2 verbatim: "ללא סביבת Gymnasium" — the project MUST NOT depend on
gymnasium.Env, gymnasium.vector.AsyncVectorEnv, or any gymnasium.* symbol
inside the environment / services layers. Custom multiprocess wrapper or
single-process per ADR-007.

This test is AST-level (not grep) so that string-occurrences in comments or
docstrings do not produce false positives.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FORBIDDEN_ROOT = "gymnasium"
SCAN_DIRS = ("src/env", "src/services")


def _iter_py_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        d = repo_root / rel
        if not d.exists():
            continue
        files.extend(p for p in d.rglob("*.py") if p.is_file())
    return files


def _gym_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == FORBIDDEN_ROOT or mod.startswith(FORBIDDEN_ROOT + "."):
                offenders.append(f"{path}: from {mod} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FORBIDDEN_ROOT or alias.name.startswith(FORBIDDEN_ROOT + "."):
                    offenders.append(f"{path}: import {alias.name}")
    return offenders


@pytest.mark.xfail(
    reason="Phase 2 will populate src/env/ and src/services/; this contract holds from then on",
    strict=False,
)
def test_no_gymnasium_imports_under_env_or_services(repo_root: Path) -> None:
    """No gymnasium.* may be imported under src/env/ or src/services/."""
    py_files = _iter_py_files(repo_root)
    # While src/env and src/services are empty stubs we still want the test
    # to xfail rather than vacuously pass, so the contract is visible.
    if not py_files:
        pytest.xfail("src/env/ and src/services/ are still empty stubs (Phase 2 pending)")
    offenders: list[str] = []
    for f in py_files:
        offenders.extend(_gym_imports(f))
    assert not offenders, (
        "gymnasium import found in env/services layer — brief §2.2 forbids it:\n" + "\n".join(offenders)
    )
