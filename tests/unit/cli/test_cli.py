"""Tests for the minimal CLI (`python -m src.cli`)."""

from __future__ import annotations

import pytest

from src.cli.__main__ import main


def test_info_is_default(capsys: pytest.CaptureFixture[str]) -> None:
    """No subcommand → info banner with reproduce instructions."""
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PythonClawRefactorRL" in out
    assert "scripts/train_5seed_isolated.py" in out


def test_graph_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """`graph` prints node/edge counts, the coupling hotspot, and orphan skills."""
    rc = main(["graph"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nodes" in out and "edges" in out
    assert "python_execution" in out  # the fan-in hotspot
    # Orphan skills match BUG_REPORT.md Bug 1.
    assert "json_validator" in out and "web_search" in out


def test_cost_token_volume(capsys: pytest.CaptureFixture[str]) -> None:
    """`cost` prints the cl100k_base token total split by layer."""
    rc = main(["cost"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "9297" in out  # sealed Skills-corpus token volume
    assert "metadata (L1)" in out
