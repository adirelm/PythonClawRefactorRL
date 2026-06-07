"""SDK layer — single business-logic entry point for all UIs (CLAUDE.md §3).

Re-exports the ablation surface so notebooks and analysis code can do
``from src.sdk import Ablation, CellResult, run_ablation`` rather than
reaching into the submodule.
"""

from src.sdk.ablation import Ablation, CellResult, run_ablation

__all__ = ["Ablation", "CellResult", "run_ablation"]
