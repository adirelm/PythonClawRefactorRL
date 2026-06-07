"""Phase-4 D7 — execute notebooks/analysis.ipynb end-to-end via nbclient.

Asserts every cell runs without raising and that the SDK-driven output text
appears (e.g. ``cells:`` from ``run_ablation``, the best/worst/baseline row,
and the heatmap+marginals embed cells emit either a display_data or a
``[pending]`` placeholder when the figure isn't yet rendered).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
NB_PATH = REPO_ROOT / "notebooks" / "analysis.ipynb"


def _cell_stream_text(cell: nbformat.NotebookNode) -> str:
    """Concatenate all stream-output text from a single executed cell."""
    return "".join(out.text for out in cell.get("outputs", []) if out.output_type == "stream")


def _cell_output_types(cell: nbformat.NotebookNode) -> set[str]:
    """Return the set of output types emitted by a cell (stream, display_data, ...)."""
    return {out.output_type for out in cell.get("outputs", [])}


@pytest.fixture(scope="module")
def executed_notebook() -> nbformat.NotebookNode:
    """Execute the analysis notebook once and return the populated node tree."""
    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    return nb


def test_no_cell_errors(executed_notebook: nbformat.NotebookNode) -> None:
    """No cell may emit an ``error`` output (would imply a Python exception)."""
    for idx, cell in enumerate(executed_notebook.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            assert out.output_type != "error", f"cell {idx} raised: {out.get('ename')}: {out.get('evalue')}"


def test_cell_count(executed_notebook: nbformat.NotebookNode) -> None:
    """The notebook is exactly 9 cells (4 markdown + 4 code + 1 trailing markdown)."""
    assert len(executed_notebook.cells) == 9


def test_sdk_summary_cell(executed_notebook: nbformat.NotebookNode) -> None:
    """Cell 2 (index 1) prints the SDK ablation summary line."""
    text = _cell_stream_text(executed_notebook.cells[1])
    assert "cells:" in text
    assert "n_ok_total:" in text
    assert "wall_clock_s:" in text


def test_best_worst_baseline_cell(executed_notebook: nbformat.NotebookNode) -> None:
    """Cell 4 (index 3) prints best/worst/baseline rows from the SDK Ablation."""
    text = _cell_stream_text(executed_notebook.cells[3])
    assert "best" in text
    assert "worst" in text
    assert "baseline" in text


def test_heatmap_cell_emits_image_or_pending(executed_notebook: nbformat.NotebookNode) -> None:
    """Cell 6 (index 5) either displays the heatmap PNG or prints a [pending] notice."""
    cell = executed_notebook.cells[5]
    types = _cell_output_types(cell)
    if "display_data" in types:
        return  # figure rendered
    assert "[pending]" in _cell_stream_text(cell)


def test_marginals_cell_emits_image_or_pending(executed_notebook: nbformat.NotebookNode) -> None:
    """Cell 8 (index 7) either displays the marginals PNG or prints a [pending] notice."""
    cell = executed_notebook.cells[7]
    types = _cell_output_types(cell)
    if "display_data" in types:
        return  # figure rendered
    assert "[pending]" in _cell_stream_text(cell)
