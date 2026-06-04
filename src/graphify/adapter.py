"""GraphifyAdapter Protocol (ADR-002).

Stub for Phase 1+ implementation. Defines the contract every concrete
adapter (local NetworkX impl, future binary swap) MUST honor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

try:
    import networkx as nx

    _DiGraph = nx.DiGraph
except ImportError:  # pragma: no cover  - nx is in pyproject deps; defensive fallback only
    _DiGraph = object  # Phase 0 stub; nx not yet wired


@runtime_checkable
class GraphifyAdapter(Protocol):
    """Static-analysis-to-graph adapter contract (brief §1.2).

    Concrete implementations:
    - src/graphify/local_impl.py — NetworkX-based local re-impl
    - (future) src/graphify/binary_impl.py — if GRAPHIFY binary becomes available

    See docs/adr/ADR-002-graphify-adapter.md (Contract Authority).
    """

    def build(self, src_root: Path, *, seed: int = 0) -> _DiGraph:
        """Extract dependency graph from source tree.

        Args:
            src_root: Path to the Skills module root (brief §1.1 scope).
            seed: Random seed for deterministic resolution (default 0).

        Returns:
            nx.DiGraph with node attrs {kind, LOC, cyclomatic, layer, lazy_load_flag}
            and edge attrs {rel_type, weight}.
        """
        ...  # pragma: no cover  - Protocol body, never executed

    def load(self, pickle_path: Path) -> _DiGraph:
        """Load a previously-built graph from pickle for fast iteration."""
        ...  # pragma: no cover  - Protocol body, never executed
