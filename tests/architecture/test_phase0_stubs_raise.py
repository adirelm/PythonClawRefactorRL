"""Phase stubs MUST raise NotImplementedError until the owning phase lands.

This test exercises every still-stubbed phase entrypoint so that
(a) coverage reports honest %, and (b) when Phase N+ lands a real
implementation, the test fails — forcing us to update the contract.

Phase 1 has landed: ``LocalGraphify.build / .load`` are real impls now, so
their assertions check real behavior (empty-tree → empty graph; missing
pickle → ``FileNotFoundError`` with informative message) rather than
``NotImplementedError``. Phases 4–6 stubs (train / evaluate / run_ablation)
remain.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from src.graphify.local_impl import LocalGraphify
from src.sdk.sdk import RefactorSDK


def test_refactorsdk_build_skills_graph_raises() -> None:
    sdk = RefactorSDK()
    with pytest.raises(NotImplementedError, match="Phase 1"):
        sdk.build_skills_graph()


def test_refactorsdk_train_raises() -> None:
    sdk = RefactorSDK()
    with pytest.raises(NotImplementedError, match="Phase 4"):
        sdk.train(seed=42)


def test_refactorsdk_evaluate_raises() -> None:
    sdk = RefactorSDK()
    with pytest.raises(NotImplementedError, match="Phase 5"):
        sdk.evaluate(checkpoint_path="/tmp/none.pt")


def test_refactorsdk_run_ablation_raises() -> None:
    sdk = RefactorSDK()
    with pytest.raises(NotImplementedError, match="Phase 6"):
        sdk.run_ablation()


def test_local_graphify_build_returns_empty_for_missing_root(tmp_path: Path) -> None:
    """Phase 1 real impl: empty / missing tree → empty ``nx.DiGraph`` (no raise)."""
    impl = LocalGraphify()
    graph = impl.build(src_root=tmp_path)
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 0


def test_local_graphify_load_raises_filenotfound() -> None:
    """Phase 1 real impl: missing pickle → ``FileNotFoundError`` w/ adapter prefix."""
    impl = LocalGraphify()
    with pytest.raises(FileNotFoundError, match=r"LocalGraphify\.load"):
        impl.load(pickle_path=Path("/tmp/none-phase1.pkl"))
