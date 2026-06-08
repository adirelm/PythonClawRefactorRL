#!/usr/bin/env -S uv run python
"""Reverse-engineer the real PythonClaw source and surface architectural bugs.

Runs GRAPHIFY (``LocalGraphify``, AST-based) over the pinned upstream source in
``vendor/pythonclaw/pythonclaw`` (fetch via ``scripts/fetch_pythonclaw.py``),
then computes the architectural smells the brief §3 bug report needs — at both
the fine (function/method) and module granularity — and writes a committed
JSON artefact so the findings are reproducible without vendoring the source.

Usage::  uv run python scripts/analyze_real_pythonclaw.py
"""

from __future__ import annotations

import ast
import json
import pickle
import sys
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graphify.local_impl import LocalGraphify  # noqa: E402

SRC = REPO_ROOT / "vendor" / "pythonclaw" / "pythonclaw"
PINNED_SHA = "7787bb433590ca2d1ee27b99819d68ad8fc3efd2"
OUT_JSON = REPO_ROOT / "results" / "data" / "real_pythonclaw_analysis.json"
OUT_GPICKLE = REPO_ROOT / "results" / "graphify_output.gpickle"
TOP = 8
LOC_LIMIT = 150  # professional file-size limit (CLAUDE.md §1)


def _resolve(importer: str, level: int, module: str | None, mods: set[str]) -> str | None:
    """Resolve a (possibly relative) ImportFrom target to an exact known module.

    Precise — NOT a fuzzy tail match — so stdlib imports (``base64``) and
    coincidental name collisions are never counted. Returns the matched module
    dotted-path, or None if the target is external/unresolved.
    """
    mod = (module or "").replace("pythonclaw.", "")
    if level:  # relative import: walk up from the importer's package
        base = importer.split(".")[:-level] if level <= len(importer.split(".")) else []
        cand = ".".join([*base, mod]) if mod else ".".join(base)
    else:
        cand = mod
    if cand in mods:
        return cand
    # `from pkg import name` where name is a submodule file → pkg.name
    return cand if cand in mods else None


def _module_graph(src: Path) -> nx.DiGraph:
    """Module-level import graph (the architectural view), exact import resolution."""
    mods = {
        ".".join(p.relative_to(src).with_suffix("").parts): p
        for p in src.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    g = nx.DiGraph()
    g.add_nodes_from(mods)
    for mod, py in mods.items():
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.level or (node.module and "pythonclaw" in node.module)
            ):
                tgt = _resolve(mod, node.level, node.module, set(mods))
                if tgt and tgt != mod:
                    g.add_edge(mod, tgt)
    return g


def _loc(p: Path) -> int:
    return sum(
        1
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )


def main() -> int:
    if not SRC.exists():
        print(f"❌ {SRC} missing — run: uv run python scripts/fetch_pythonclaw.py")
        return 1
    fine = LocalGraphify().build(SRC, seed=42)
    OUT_GPICKLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_GPICKLE.open("wb") as fh:
        pickle.dump(fine, fh)

    mg = _module_graph(SRC)
    mods = {
        ".".join(p.relative_to(SRC).with_suffix("").parts): p
        for p in SRC.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    loc = {m: _loc(p) for m, p in mods.items()}
    internal = [n for n in fine.nodes if fine.nodes[n].get("kind") != "external"]

    analysis = {
        "source": "github.com/ericwang915/PythonClaw",
        "pinned_sha": PINNED_SHA,
        "fine_graph": {"nodes": fine.number_of_nodes(), "edges": fine.number_of_edges()},
        "module_graph": {"nodes": mg.number_of_nodes(), "edges": mg.number_of_edges()},
        "module_import_cycles": [list(c) for c in nx.simple_cycles(mg)],
        "god_modules_by_loc": sorted(((v, k) for k, v in loc.items()), reverse=True)[:TOP],
        "top_module_fan_in": sorted(mg.in_degree(), key=lambda x: x[1], reverse=True)[:TOP],
        "top_module_fan_out": sorted(mg.out_degree(), key=lambda x: x[1], reverse=True)[:TOP],
        "top_method_fan_out": sorted(((fine.out_degree(n), n) for n in internal), reverse=True)[:TOP],
        "modules_over_150_loc": sorted(((v, k) for k, v in loc.items() if v > LOC_LIMIT), reverse=True),
        "total_loc": sum(loc.values()),
    }
    OUT_JSON.write_text(json.dumps(analysis, indent=2))
    print(f"✅ wrote {OUT_JSON.relative_to(REPO_ROOT)} and {OUT_GPICKLE.relative_to(REPO_ROOT)}")
    print(f"   fine graph: {analysis['fine_graph']}  module graph: {analysis['module_graph']}")
    print(
        f"   god module: {analysis['god_modules_by_loc'][0]}  | >150 LOC modules: {len(analysis['modules_over_150_loc'])}"
    )
    print(
        f"   top fan-in: {analysis['top_module_fan_in'][0]}  module cycles: {len(analysis['module_import_cycles'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
