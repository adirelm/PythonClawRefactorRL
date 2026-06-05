"""Architectural contract: NO gymnasium imports under src/env/ or src/services/.

Brief §2.2 verbatim: "ללא סביבת Gymnasium" — the project MUST NOT depend on
gymnasium.Env, gymnasium.vector.AsyncVectorEnv, or any gymnasium.* symbol
inside the environment / services layers. Custom multiprocess wrapper or
single-process per ADR-007.

This test is AST-level (not grep) so that string-occurrences in comments or
docstrings do not produce false positives. It is parametrised over every
``.py`` file under ``src/env/`` and ``src/services/`` so a regression in any
single file fails its own test case.

Two assertions per file:
    1. No ``ImportFrom`` whose module is ``gymnasium`` or starts with
       ``gymnasium.`` and no plain ``import gymnasium[.x]``.
    2. No ``ClassDef`` under ``src/env/`` whose bases reference
       ``gym.Env`` / ``gymnasium.Env`` (legacy ``gym`` also blocked
       defensively, since ADR-007 forbids both flavours).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FORBIDDEN_IMPORT_ROOTS = ("gymnasium",)
FORBIDDEN_BASE_ROOTS = ("gym", "gymnasium")
SCAN_DIRS = ("src/env", "src/services")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _iter_py_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_DIRS:
        d = _REPO_ROOT / rel
        if not d.exists():
            continue
        files.extend(p for p in d.rglob("*.py") if p.is_file())
    return sorted(files)


def _is_forbidden_import_name(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in FORBIDDEN_IMPORT_ROOTS)


def _gym_imports(tree: ast.AST, path: Path) -> list[str]:
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_forbidden_import_name(mod):
                offenders.append(f"{path}:{node.lineno}: from {mod} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_import_name(alias.name):
                    offenders.append(f"{path}:{node.lineno}: import {alias.name}")
    return offenders


def _base_dotted_name(base: ast.expr) -> str:
    """Return the dotted source name of a class base (best-effort)."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        parent = _base_dotted_name(base.value)
        return f"{parent}.{base.attr}" if parent else base.attr
    return ""


def _is_forbidden_base(dotted: str) -> bool:
    if not dotted:
        return False
    head = dotted.split(".", 1)[0]
    return head in FORBIDDEN_BASE_ROOTS


def _gym_env_bases(tree: ast.AST, path: Path) -> list[str]:
    offenders: list[str] = []
    if "src/env" not in str(path).replace("\\", "/"):
        return offenders
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            dotted = _base_dotted_name(base)
            if _is_forbidden_base(dotted):
                offenders.append(
                    f"{path}:{node.lineno}: class {node.name}({dotted}) "
                    "— brief §2.2 / ADR-007 forbid gym(nasium).Env inheritance"
                )
    return offenders


_PY_FILES = _iter_py_files()
_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _PY_FILES]


@pytest.mark.skipif(not _PY_FILES, reason="No .py files under src/env/ or src/services/ yet")
@pytest.mark.parametrize("py_file", _PY_FILES, ids=_IDS)
def test_no_gymnasium_in_env_or_services(py_file: Path) -> None:
    """No gymnasium.* may be imported and no gym(nasium).Env may be inherited."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    import_offenders = _gym_imports(tree, py_file)
    base_offenders = _gym_env_bases(tree, py_file)
    offenders = import_offenders + base_offenders
    assert not offenders, (
        "gymnasium contract violation under src/env/ or src/services/ "
        "(brief §2.2 + ADR-007):\n" + "\n".join(offenders)
    )
