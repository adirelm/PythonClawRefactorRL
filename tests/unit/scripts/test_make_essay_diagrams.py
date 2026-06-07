"""Tests for ``scripts/make_essay_diagrams.py`` (Phase-4 ESSAY-DIAGRAMS-D1).

Locks the architecture-diagram contract: PNG exists, is wide enough to read
on a printed page, and the CLI exits 0 in a fresh subprocess. The diagram
content itself (boxes / arrows / labels) is fixed by the script's NODES
table and visually reviewed at submission time — these tests guard the
machine-checkable surface.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "make_essay_diagrams.py"
_SPEC = importlib.util.spec_from_file_location("make_essay_diagrams", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["make_essay_diagrams"] = mod
_SPEC.loader.exec_module(mod)


def test_diagram_outputs_png(tmp_path: Path) -> None:
    """`main()` writes essay_d1_architecture.png > 5 KB to --output-dir."""
    rc = mod.main(["--output-dir", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "essay_d1_architecture.png"
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_diagram_dimensions(tmp_path: Path) -> None:
    """Rendered PNG is at least 1000 px wide so labels stay readable."""
    pil_image = pytest.importorskip("PIL.Image")
    rc = mod.main(["--output-dir", str(tmp_path)])
    assert rc == 0
    with pil_image.open(tmp_path / "essay_d1_architecture.png") as img:
        width, height = img.size
    assert width >= 1000
    assert height >= 500


def test_smoke_renders_without_error(tmp_path: Path) -> None:
    """Subprocess invocation (python scripts/make_essay_diagrams.py) returns 0."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert (tmp_path / "essay_d1_architecture.png").exists()


def test_nodes_cover_pipeline_stages() -> None:
    """NODES table includes every component the brief mandates for D1."""
    labels = [label for label, _ in mod.NODES]
    for required in (
        "PythonClaw source",
        "LocalGraphify",
        "PolicyNet",
        "Action",
        "compute_reward",
        "GAE buffer",
        "PPO update",
    ):
        assert required in labels, f"D1 missing component: {required}"


def test_title_names_closed_loop() -> None:
    """Title surfaces the priors→policy→reward closed-loop thesis."""
    assert "priors" in mod.TITLE.lower()
    assert "reward" in mod.TITLE.lower()
    assert "closed loop" in mod.CAPTION.lower()


def test_render_returns_output_path(tmp_path: Path) -> None:
    """`render(out)` creates parent dirs and returns the written path."""
    target = tmp_path / "nested" / "essay_d1_architecture.png"
    written = mod.render(target)
    assert written == target
    assert target.exists()
    assert target.stat().st_size > 5_000
