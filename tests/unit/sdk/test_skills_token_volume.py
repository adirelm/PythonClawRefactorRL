"""Unit test for ``RefactorSDK.skills_token_volume`` — the cost surface the CLI
``cost`` subcommand calls, so the CLI imports only the SDK (CLAUDE.md §3:
UIs depend on the SDK, never on ``src.cost`` directly).
"""

from __future__ import annotations

from src.sdk.sdk import RefactorSDK


def test_skills_token_volume_shape_and_totals() -> None:
    vol = RefactorSDK().skills_token_volume()
    assert set(vol) == {"total", "by_layer", "lazy_load_saving"}
    assert isinstance(vol["total"], int) and vol["total"] > 0
    assert set(vol["by_layer"]) == {"metadata (L1)", "instructions (L2)", "resources (L3)"}
    # every sample_skills JSON matches exactly one layer hint, so layers sum to total
    assert sum(vol["by_layer"].values()) == vol["total"]
    assert vol["lazy_load_saving"] >= 1.0
