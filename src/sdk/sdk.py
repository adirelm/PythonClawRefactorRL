"""RefactorSDK — public API surface for CLI / GUI / notebooks.

This is a Phase-0 stub. Every method raises NotImplementedError; subsequent
phases land the real implementations against the PRD-approved signatures.
UIs must depend only on this class — never on src.services / src.env / src.model
directly (CLAUDE.md §3).
"""

from __future__ import annotations


class RefactorSDK:
    """Single business-logic entry point. Stubbed until Phase 2."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path

    def build_skills_graph(self) -> object:
        """Build the Skills-layer dependency graph. Implemented in Phase 1."""
        raise NotImplementedError("RefactorSDK.build_skills_graph — Phase 1")

    def train(self, seed: int) -> object:
        """Run a single PPO+GAE training session at the given seed."""
        raise NotImplementedError("RefactorSDK.train — Phase 4")

    def evaluate(self, checkpoint_path: str) -> object:
        """Evaluate a trained policy and return metrics."""
        raise NotImplementedError("RefactorSDK.evaluate — Phase 5")

    def run_ablation(self) -> object:
        """Sweep alpha/beta/gamma/P_skills over >= 5 seeds per cell."""
        raise NotImplementedError("RefactorSDK.run_ablation — Phase 6")
