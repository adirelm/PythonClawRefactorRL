"""Local NetworkX-based GraphifyAdapter implementation (Phase 1+ stub).

This module will house the AST-walk + import-resolution logic that builds
a `nx.DiGraph` from a Python source tree, honoring the GraphifyAdapter
Protocol defined in `src/graphify/adapter.py` (ADR-002).

Phase 0: NotImplementedError-only stub so the Protocol is satisfied
structurally (method names + signatures) while the real implementation
is deferred to Phase 1.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import networkx as nx

    _DiGraph = nx.DiGraph
except ImportError:
    _DiGraph = object  # Phase 0 stub; nx not yet wired

logger = logging.getLogger(__name__)


class LocalGraphify:
    """NetworkX-backed concrete GraphifyAdapter — Phase 1+ pending.

    Honors the GraphifyAdapter Protocol structurally (duck-typed via
    `runtime_checkable`). All methods currently raise NotImplementedError
    and log the deferral so accidental Phase-0 invocations are loud.
    """

    def build(self, src_root: Path, *, seed: int = 0) -> _DiGraph:
        """Build dependency graph from source tree (Phase 1+ pending)."""
        logger.info(
            "LocalGraphify.build: Phase 1+ pending (src_root=%s, seed=%d)",
            src_root,
            seed,
        )
        raise NotImplementedError(
            "LocalGraphify.build is a Phase 0 stub; implementation lands in Phase 1+."
        )

    def load(self, pickle_path: Path) -> _DiGraph:
        """Load previously-built graph from pickle (Phase 1+ pending)."""
        logger.info("LocalGraphify.load: Phase 1+ pending (pickle=%s)", pickle_path)
        raise NotImplementedError(
            "LocalGraphify.load is a Phase 0 stub; implementation lands in Phase 1+."
        )
