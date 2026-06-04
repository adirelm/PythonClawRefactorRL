"""CLI: graphify → vault pipeline (ADR-009 dual-track, programmatic leg).

Runs ``LocalGraphify().build(src, seed=42)`` against the PythonClaw shim's
sample-skills tree, persists the resulting ``nx.DiGraph`` as a pickle for
downstream reuse, and renders an Obsidian-compatible markdown vault via
``src.services.vault_writer.write_vault``. Defaults wire the canonical
``src/pythonclaw_shim/sample_skills`` source and ``results/`` outputs so
``uv run python scripts/build_vault.py`` is a no-arg one-shot pipeline.

Note: ``LocalGraphify.build`` is a Phase-0 stub that raises
``NotImplementedError`` until Phase 1+ lands the AST walker; this script
defines the orchestration so the call-site is wired once the impl arrives.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Make ``src`` importable when invoked as ``python scripts/build_vault.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graphify.local_impl import LocalGraphify  # noqa: E402
from src.services.vault_writer import write_vault  # noqa: E402

_DEFAULT_SRC = _REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"
_DEFAULT_VAULT = _REPO_ROOT / "results" / "vault"
_DEFAULT_PICKLE = _REPO_ROOT / "results" / "graphify_output.gpickle"
_SEED = 42


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dependency graph + Obsidian vault.")
    parser.add_argument("--src", type=Path, default=_DEFAULT_SRC, help="Skills source tree root.")
    parser.add_argument("--vault", type=Path, default=_DEFAULT_VAULT, help="Vault output directory.")
    parser.add_argument("--pickle", type=Path, default=_DEFAULT_PICKLE, help="Graph pickle output path.")
    return parser.parse_args(argv)


def _save_pickle(graph: object, pickle_path: Path) -> None:
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)


def run(src: Path, vault: Path, pickle_path: Path) -> int:
    """Execute graphify → vault pipeline and return a process exit code."""
    graph = LocalGraphify().build(src, seed=_SEED)
    _save_pickle(graph, pickle_path)
    write_vault(graph, vault)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"Vault written to {vault}")
    print(f"  nodes: {n_nodes}")
    print(f"  edges: {n_edges}")
    print(f"  pickle: {pickle_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args.src, args.vault, args.pickle)


if __name__ == "__main__":
    sys.exit(main())
