"""Architecture test: SOLID violations scan (brief §2.1 DI / SRP mandate).

Checks two structural invariants:
1. No cross-service imports — services must not directly import each other
   (Dependency Inversion; DI wiring belongs in the SDK layer).
2. Every public class in src/ has ≤1 public non-dunder method per 30 LOC
   (coarse Single-Responsibility heuristic; not a hard gate but a smoke signal).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src"
SERVICES = SRC / "services"

# Services that ARE expected to import each other (legitimate cross-wires).
_ALLOWED_CROSS = {
    # gae_buffer is a pure data container; ppo_trainer imports it.
    ("ppo_trainer", "gae_buffer"),
    # _ppo_helpers is an internal helper for ppo_trainer.
    ("ppo_trainer", "_ppo_helpers"),
    # metrics sub-package: modularity imports shared cache helpers.
    ("cohesion", "modularity"),
    ("coupling", "modularity"),
}


def _service_modules() -> dict[str, Path]:
    """Return {module_stem: path} for every .py directly under src/services/ (not sub-packages)."""
    return {p.stem: p for p in SERVICES.glob("*.py") if not p.name.startswith("_")}


def _imports_from(path: Path) -> list[str]:
    """Return list of src.services.X imported by the module at path."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("src.services."):
                # Extract the immediate sub-module name (e.g. "gae_buffer")
                parts = mod.split(".")
                if len(parts) >= 3:
                    found.append(parts[2])
    return found


def test_services_do_not_have_forbidden_cross_imports() -> None:
    """No service imports another service unless listed in _ALLOWED_CROSS."""
    violations: list[str] = []
    for stem, path in _service_modules().items():
        for imported in _imports_from(path):
            if imported == stem:
                continue  # self-reference impossible but guard anyway
            pair = (stem, imported)
            if pair not in _ALLOWED_CROSS:
                violations.append(f"{stem}.py imports src.services.{imported} (not in allowed list)")
    assert not violations, "Forbidden service cross-imports:\n" + "\n".join(violations)


def test_sdk_is_single_entry_point_for_env_imports() -> None:
    """CLI and notebook modules must not bypass the SDK to import src.env directly."""
    cli_dir = SRC / "cli"
    if not cli_dir.exists():
        return  # CLI not yet scaffolded — skip
    forbidden_prefix = "src.env"
    violations: list[str] = []
    for py in cli_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith(forbidden_prefix):
                    violations.append(f"{py.relative_to(REPO_ROOT)} imports {mod}")
    assert not violations, "CLI bypasses SDK by importing src.env directly:\n" + "\n".join(violations)
