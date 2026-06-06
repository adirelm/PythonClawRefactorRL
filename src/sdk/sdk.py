"""RefactorSDK — public API surface for CLI / GUI / notebooks.

Phase-1 wiring complete for ``build_skills_graph`` (delegates to
``LocalGraphify`` per ADR-002). ``train`` / ``evaluate`` / ``run_ablation``
remain Phase-3/4 stubs and still raise ``NotImplementedError``.
UIs must depend only on this class — never on src.services / src.env / src.model
directly (CLAUDE.md §3).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.graphify.local_impl import LocalGraphify

_DEFAULT_SOURCE = Path("src/pythonclaw_shim/sample_skills")


class RefactorSDK:
    """Single business-logic entry point. Skills-graph wired in Phase 1."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path

    def build_skills_graph(self, source: Path | str | None = None) -> nx.DiGraph:
        """Build the Skills-layer dependency graph via ``LocalGraphify`` (ADR-002).

        Defaults to ``src/pythonclaw_shim/sample_skills`` when ``source`` is None.
        Uses ``seed=42`` for deterministic traversal ordering.
        """
        src_root = Path(source) if source is not None else _DEFAULT_SOURCE
        return LocalGraphify().build(src_root=src_root, seed=42)

    def train(self, seed: int) -> object:
        """Run a single PPO+GAE training session at the given seed."""
        raise NotImplementedError("RefactorSDK.train — Phase 4")

    def evaluate(self, checkpoint_path: str) -> object:
        """Evaluate a trained policy and return metrics."""
        raise NotImplementedError("RefactorSDK.evaluate — Phase 5")

    def run_ablation(self) -> object:
        """Sweep alpha/beta/gamma/P_skills over >= 5 seeds per cell."""
        raise NotImplementedError("RefactorSDK.run_ablation — Phase 6")
