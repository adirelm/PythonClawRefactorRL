"""Unit tests for src/graphify/ast_visitor.py (ADR-002 node/edge contract)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graphify.ast_visitor import walk_module


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Per-test isolated dir for fixture .py files."""
    d = tmp_path / "graphify_samples"
    d.mkdir()
    return d


def _write(d: Path, name: str, body: str) -> Path:
    p = d / name
    p.write_text(body, encoding="utf-8")
    return p


def test_walk_module_extracts_class_nodes(sample_dir: Path) -> None:
    src = _write(sample_dir, "m.py", "class Foo:\n    x = 1\n\nclass Bar:\n    y = 2\n")
    nodes, _ = walk_module(src)
    class_names = {n["name"] for n in nodes if n["kind"] == "class"}
    assert class_names == {"Foo", "Bar"}
    foo = next(n for n in nodes if n["name"] == "Foo")
    assert foo["qualified_name"] == "m.Foo"
    assert foo["layer"] is None
    assert foo["loc"] >= 1


def test_walk_module_extracts_function_nodes(sample_dir: Path) -> None:
    src = _write(sample_dir, "m.py", "def alpha():\n    return 1\n\ndef beta(x):\n    return x\n")
    nodes, _ = walk_module(src)
    fn_names = {n["name"] for n in nodes if n["kind"] == "function"}
    assert fn_names == {"alpha", "beta"}
    alpha = next(n for n in nodes if n["name"] == "alpha")
    assert alpha["qualified_name"] == "m.alpha"


def test_inheritance_edges_emitted(sample_dir: Path) -> None:
    src = _write(sample_dir, "m.py", "class A:\n    pass\n\nclass B(A):\n    pass\n")
    _, edges = walk_module(src)
    inh = [e for e in edges if e["rel_type"] == "inheritance"]
    assert any(e["src"] == "m.B" and e["dst"] == "A" for e in inh)


def test_import_edges_emitted(sample_dir: Path) -> None:
    src = _write(sample_dir, "m.py", "from x import y\nimport os\n")
    _, edges = walk_module(src)
    imports = [e for e in edges if e["rel_type"] == "import"]
    dsts = {e["dst"] for e in imports}
    assert "x.y" in dsts
    assert "os" in dsts
    for e in imports:
        assert e["src"] == "m"
        assert e["weight"] == 1


def test_function_call_edges_emitted(sample_dir: Path) -> None:
    src = _write(
        sample_dir,
        "m.py",
        "def f():\n    return 1\n\ndef g():\n    f()\n    f()\n    return 0\n",
    )
    _, edges = walk_module(src)
    calls = [e for e in edges if e["rel_type"] == "call"]
    g_to_f = [e for e in calls if e["src"] == "m.g" and e["dst"] == "f"]
    assert len(g_to_f) == 1
    assert g_to_f[0]["weight"] == 2


def test_loc_count_matches(sample_dir: Path) -> None:
    body_lines = ["def big(x):"] + [f"    x = x + {i}" for i in range(19)]
    src = _write(sample_dir, "m.py", "\n".join(body_lines) + "\n")
    nodes, _ = walk_module(src)
    big = next(n for n in nodes if n["name"] == "big")
    assert 18 <= big["loc"] <= 22


def test_cyclomatic_at_least_1(sample_dir: Path) -> None:
    src = _write(
        sample_dir,
        "m.py",
        "def simple():\n    return 1\n\ndef branchy(x):\n    if x:\n        return 1\n    return 0\n",
    )
    nodes, _ = walk_module(src)
    fns = [n for n in nodes if n["kind"] == "function"]
    for n in fns:
        assert n["cyclomatic"] >= 1
    branchy = next(n for n in nodes if n["name"] == "branchy")
    simple = next(n for n in nodes if n["name"] == "simple")
    assert branchy["cyclomatic"] > simple["cyclomatic"]


def test_async_function_extracted(sample_dir: Path) -> None:
    src = _write(sample_dir, "m.py", "async def fetch():\n    return 1\n")
    nodes, _ = walk_module(src)
    assert any(n["name"] == "fetch" and n["kind"] == "function" for n in nodes)


def test_lazy_flag_underscore_and_property(sample_dir: Path) -> None:
    src = _write(
        sample_dir,
        "m.py",
        "class C:\n"
        "    @property\n"
        "    def val(self):\n        return 1\n\n"
        "    def _hidden(self):\n        return 2\n\n"
        "    def public(self):\n        return 3\n",
    )
    nodes, _ = walk_module(src)
    by_name = {n["name"]: n for n in nodes}
    assert by_name["val"]["lazy_load_flag"] is True
    assert by_name["_hidden"]["lazy_load_flag"] is True
    assert by_name["public"]["lazy_load_flag"] is False


def test_module_node_emitted(sample_dir: Path) -> None:
    src = _write(sample_dir, "mymod.py", "x = 1\n")
    nodes, _ = walk_module(src)
    mods = [n for n in nodes if n["kind"] == "module"]
    assert len(mods) == 1
    assert mods[0]["qualified_name"] == "mymod"
