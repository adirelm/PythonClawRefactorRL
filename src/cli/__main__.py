"""Minimal CLI for PythonClawRefactorRL — ``uv run python -m src.cli``.

Read-only inspection front-end over the public ``RefactorSDK`` (CLAUDE.md §3:
UIs depend only on the SDK). Heavy training / ablation are deliberately *not*
re-implemented here — they live in ``scripts/`` and this menu points at them —
so the CLI stays a thin, torch-free consumer.

Subcommands:
  graph  — build the Skills dependency graph and print a structural summary
  cost   — token volume of the Skills corpus (tiktoken cl100k_base, by layer)
  info   — (default) status + where artifacts live + how to run training
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.sdk.sdk import RefactorSDK

_SKILLS = Path("src/pythonclaw_shim/sample_skills")
_TOP_N = 5


def _cmd_graph() -> int:
    """Build the Skills graph and print |V|, |E|, top fan-in nodes, orphans."""
    graph = RefactorSDK().build_skills_graph()
    print(f"Skills dependency graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    fan_in = sorted(graph.in_degree(), key=lambda kv: kv[1], reverse=True)
    print("Top fan-in (coupling hotspots):")
    for node, deg in fan_in[:_TOP_N]:
        if deg:
            print(f"  {node}: fan-in {deg}")
    # depends_on edges connect L1 (skill-level) nodes; report orphan *skills*
    # (L1 nodes with no edges) — matches docs/BUG_REPORT.md Bug 1.
    orphans = [
        n
        for n in graph.nodes
        if str(n).endswith(".L1") and graph.in_degree(n) == 0 and graph.out_degree(n) == 0
    ]
    print(f"Orphan skills (disconnected L1 nodes): {orphans or 'none'}")
    return 0


def _cmd_cost() -> int:
    """Print the Skills corpus token volume (cl100k_base) split by L1/L2/L3."""
    from src.cost.meter import TripleCounter  # noqa: PLC0415 — keep tiktoken import lazy

    counter = TripleCounter()
    by_layer = {"metadata (L1)": 0, "instructions (L2)": 0, "resources (L3)": 0}
    total = 0
    for path in sorted(_SKILLS.glob("*.json")):
        tokens = counter.count(path.read_text(encoding="utf-8")).tokens
        total += tokens
        for key, hint in (
            ("metadata (L1)", "metadata"),
            ("instructions (L2)", "instructions"),
            ("resources (L3)", "resources"),
        ):
            if hint in path.name:
                by_layer[key] += tokens
    print(f"Skills corpus token volume (cl100k_base): {total} tokens")
    for layer, count in by_layer.items():
        print(f"  {layer}: {count}")
    print(f"Lazy-load saving (L1-only vs all): {total / max(by_layer['metadata (L1)'], 1):.1f}x")
    return 0


def _cmd_info() -> int:
    """Print project status and where the deliverables live."""
    print("PythonClawRefactorRL — PPO+GAE refactoring agent (Bar-Ilan A4).")
    print("Status: complete (Phases 0-4). See README.md for the full report.")
    print("\nInspect:")
    print("  uv run python -m src.cli graph   # Skills graph structure")
    print("  uv run python -m src.cli cost    # token volume by layer")
    print("\nReproduce results (heavier, in scripts/):")
    print("  uv run python scripts/train_5seed_isolated.py   # 5-seed PPO training")
    print("  uv run python scripts/run_ablation.py --grid compact")
    print("  uv run python scripts/render_learning_curve.py")
    print("\nArtifacts: results/figures/, results/learning_curves/, results/data/")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch. Defaults to ``info`` when no subcommand is given."""
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="PythonClawRefactorRL CLI.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("graph", help="print Skills dependency-graph summary")
    sub.add_parser("cost", help="print Skills corpus token volume by layer")
    sub.add_parser("info", help="status + how to reproduce results")
    args = parser.parse_args(argv)
    dispatch = {"graph": _cmd_graph, "cost": _cmd_cost, "info": _cmd_info}
    return dispatch.get(args.command, _cmd_info)()


if __name__ == "__main__":
    raise SystemExit(main())
