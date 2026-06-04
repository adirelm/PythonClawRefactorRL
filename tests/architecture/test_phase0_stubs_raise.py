"""Phase-0 stubs MUST raise NotImplementedError.

This test exercises every Phase-0 NotImplementedError stub so that
(a) coverage reports honest %, and (b) when Phase 1+ lands a real
implementation, the test fails — forcing us to update the contract.
"""

from __future__ import annotations

from pathlib import Path

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


def test_local_graphify_build_raises() -> None:
    impl = LocalGraphify()
    with pytest.raises(NotImplementedError, match="Phase 0 stub"):
        impl.build(src_root=Path("/tmp/none"))


def test_local_graphify_load_raises() -> None:
    impl = LocalGraphify()
    with pytest.raises(NotImplementedError, match="Phase 0 stub"):
        impl.load(pickle_path=Path("/tmp/none.pkl"))
