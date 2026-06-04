"""Tests for src/services/vault_writer.py — DiGraph → Obsidian Vault."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from src.services.vault_writer import sanitize, write_vault


def _node_attrs(kind: str = "function", layer: str = "L1") -> dict:
    return {
        "kind": kind,
        "LOC": 10,
        "cyclomatic": 2,
        "layer": layer,
        "lazy_load_flag": False,
    }


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("A", **_node_attrs(kind="class"))
    g.add_node("B", **_node_attrs(kind="function"))
    g.add_node("C", **_node_attrs(kind="module"))
    g.add_edge("A", "B", rel_type="call", weight=1.0)
    g.add_edge("A", "C", rel_type="import", weight=1.0)
    g.add_edge("B", "C", rel_type="inheritance", weight=1.0)
    return g


def test_write_vault_creates_markdown_per_node(tmp_path: Path, sample_graph: nx.DiGraph) -> None:
    write_vault(sample_graph, tmp_path)
    md_files = sorted(p.name for p in tmp_path.glob("*.md") if p.name != "README.md")
    assert md_files == ["A.md", "B.md", "C.md"]
    assert len(md_files) == sample_graph.number_of_nodes()


def test_md_files_have_yaml_frontmatter(tmp_path: Path, sample_graph: nx.DiGraph) -> None:
    write_vault(sample_graph, tmp_path)
    for node in sample_graph.nodes:
        text = (tmp_path / f"{node}.md").read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == "---", f"{node}.md missing opening front-matter"
        # find closing fence after the opener
        closing_idx = next(i for i, ln in enumerate(lines[1:], start=1) if ln == "---")
        assert closing_idx >= 2
        body_between = lines[1:closing_idx]
        joined = "\n".join(body_between)
        for key in ("kind", "LOC", "cyclomatic", "layer", "lazy_load_flag"):
            assert f"{key}:" in joined, f"{key} missing from {node}.md front-matter"


def test_wikilinks_match_edges(tmp_path: Path, sample_graph: nx.DiGraph) -> None:
    write_vault(sample_graph, tmp_path)
    a_md = (tmp_path / "A.md").read_text(encoding="utf-8")
    calls_idx = a_md.index("## Calls")
    imports_idx = a_md.index("## Imports")
    # [[B]] appears under "## Calls" (between Calls and Imports sections)
    calls_section = a_md[calls_idx:imports_idx]
    assert "[[B]]" in calls_section
    # [[C]] appears under "## Imports"
    imports_section = a_md[imports_idx:]
    assert "[[C]]" in imports_section
    b_md = (tmp_path / "B.md").read_text(encoding="utf-8")
    assert "## Inherits from" in b_md
    inherits_section = b_md[b_md.index("## Inherits from") :]
    assert "[[C]]" in inherits_section


def test_sanitize_handles_special_chars() -> None:
    assert sanitize("a/b:c") == "a_b_c"
    assert sanitize("pkg.module:Class") == "pkg.module_Class"
    assert sanitize("foo\\bar") == "foo_bar"
    # all-special collapses to a single underscore placeholder
    assert sanitize("///") == "_"


def test_README_written_at_vault_root(tmp_path: Path, sample_graph: nx.DiGraph) -> None:  # noqa: N802
    write_vault(sample_graph, tmp_path)
    readme = tmp_path / "README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert f"Total nodes: {sample_graph.number_of_nodes()}" in text
    assert f"Total edges: {sample_graph.number_of_edges()}" in text
    assert "Top-5 most-depended-on" in text


def test_write_vault_creates_missing_output_dir(tmp_path: Path, sample_graph: nx.DiGraph) -> None:
    target = tmp_path / "nested" / "vault"
    assert not target.exists()
    write_vault(sample_graph, target)
    assert target.is_dir()
    assert (target / "README.md").exists()


def test_empty_graph_writes_only_readme(tmp_path: Path) -> None:
    write_vault(nx.DiGraph(), tmp_path)
    assert (tmp_path / "README.md").exists()
    md_files = [p.name for p in tmp_path.glob("*.md")]
    assert md_files == ["README.md"]
