"""Phase-1 unit tests for LocalGraphify (ADR-002 acceptance evidence)."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pytest

from src.graphify.local_impl import LocalGraphify

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIM_ROOT = REPO_ROOT / "src" / "pythonclaw_shim"
REQUIRED_NODE_ATTRS = {"kind", "LOC", "cyclomatic", "layer", "lazy_load_flag"}
REQUIRED_EDGE_ATTRS = {"rel_type", "weight"}


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@pytest.fixture
def graphify() -> LocalGraphify:
    return LocalGraphify()


def test_build_returns_digraph_type(graphify: LocalGraphify, tmp_path: Path) -> None:
    _write(tmp_path / "m.py", "x = 1\n")
    graph = graphify.build(tmp_path)
    assert isinstance(graph, nx.DiGraph)


def test_build_on_sample_skills_yields_at_least_10_nodes(graphify: LocalGraphify) -> None:
    """Walking the shim Skills dir yields ≥10 skill_layer nodes (L1+L2 JSONs)."""
    graph = graphify.build(SHIM_ROOT / "sample_skills")
    skill_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "skill_layer"]
    assert len(skill_nodes) >= 10, f"expected ≥10 skill_layer nodes, got {len(skill_nodes)}"


def test_build_emits_call_edges(graphify: LocalGraphify, tmp_path: Path) -> None:
    """A function that calls another emits a `rel_type='call'` edge."""
    src = "def helper():\n    return 1\n\ndef caller():\n    helper()\n    helper()\n"
    _write(tmp_path / "registry.py", src)
    graph = graphify.build(tmp_path)
    call_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["rel_type"] == "call"]
    assert call_edges, "expected at least one call edge"


def test_build_emits_inheritance_edges(graphify: LocalGraphify, tmp_path: Path) -> None:
    """`class B(A)` produces a `rel_type='inheritance'` edge from B → A."""
    _write(tmp_path / "inh.py", "class A:\n    pass\n\nclass B(A):\n    pass\n")
    graph = graphify.build(tmp_path)
    inh_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["rel_type"] == "inheritance"]
    assert inh_edges, "expected at least one inheritance edge"
    assert any(v == "A" for _, v in inh_edges), "B should inherit from A"


def test_node_attrs_present(graphify: LocalGraphify, tmp_path: Path) -> None:
    """Every node carries the full required attribute set."""
    _write(tmp_path / "m.py", "def f():\n    return 1\n")
    _write(tmp_path / "s.metadata.json", '{"name": "s", "depends_on": []}')
    graph = graphify.build(tmp_path)
    for nid, attrs in graph.nodes(data=True):
        missing = REQUIRED_NODE_ATTRS - set(attrs.keys())
        assert not missing, f"node {nid!r} missing attrs: {missing}"


def test_edge_attrs_present(graphify: LocalGraphify, tmp_path: Path) -> None:
    """Every edge carries `rel_type` + `weight`."""
    body = "import os\n\ndef g():\n    print(1)\n\nclass C(object):\n    pass\n"
    _write(tmp_path / "m.py", body)
    graph = graphify.build(tmp_path)
    assert graph.number_of_edges() > 0
    for u, v, attrs in graph.edges(data=True):
        missing = REQUIRED_EDGE_ATTRS - set(attrs.keys())
        assert not missing, f"edge {u}->{v} missing attrs: {missing}"
        assert attrs["rel_type"] in {"call", "import", "inheritance"}


def test_load_roundtrip(graphify: LocalGraphify, tmp_path: Path) -> None:
    """build → pickle.dump → LocalGraphify.load yields an isomorphic graph."""
    _write(tmp_path / "m.py", "import os\n\ndef f():\n    return 1\n")
    original = graphify.build(tmp_path)
    pickle_path = tmp_path / "graph.pkl"
    with pickle_path.open("wb") as fh:
        pickle.dump(original, fh)
    reloaded = graphify.load(pickle_path)
    assert isinstance(reloaded, nx.DiGraph)
    assert set(reloaded.nodes()) == set(original.nodes())
    assert set(reloaded.edges()) == set(original.edges())


def test_load_missing_file_raises(graphify: LocalGraphify, tmp_path: Path) -> None:
    """A missing pickle path raises FileNotFoundError with adapter prefix."""
    with pytest.raises(FileNotFoundError, match=r"LocalGraphify\.load"):
        graphify.load(tmp_path / "absent.pkl")


_CODE_KINDS = {"function", "method", "class"}


def test_mccabe_floor_for_code_nodes(graphify: LocalGraphify, tmp_path: Path) -> None:
    """McCabe ≥ 1 invariant: every function/method/class has cyclomatic ≥ 1.

    Regression for MEDIUM finding — ``_attrs`` previously defaulted cyc=0 and
    layer=0, breaking the McCabe ≥1 floor *and* colliding with valid
    skill_layer values {1, 2, 3}.
    """
    body = "def f():\n    return 1\n\nclass C:\n    def m(self):\n        return 2\n"
    _write(tmp_path / "m.py", body)
    graph = graphify.build(tmp_path)
    code_nodes = [(n, d) for n, d in graph.nodes(data=True) if d.get("kind") in _CODE_KINDS]
    assert code_nodes, "expected at least one function/class node"
    for nid, attrs in code_nodes:
        cyc = attrs["cyclomatic"]
        assert cyc is not None and cyc >= 1, f"node {nid!r} violates McCabe ≥1 (got {cyc!r})"


def test_layer_semantics_code_vs_skill(graphify: LocalGraphify, tmp_path: Path) -> None:
    """layer ∈ {1,2,3} only for skill_layer nodes; ``None`` for code nodes.

    Was previously ``0`` for code, which collided with the skill layer
    enum domain and made any consumer keying on ``layer`` ambiguous.
    """
    _write(tmp_path / "m.py", "def f():\n    return 1\n")
    _write(tmp_path / "s.metadata.json", '{"name": "s", "depends_on": []}')
    graph = graphify.build(tmp_path)
    for nid, attrs in graph.nodes(data=True):
        kind = attrs.get("kind")
        layer = attrs["layer"]
        if kind == "skill_layer":
            assert layer in {1, 2, 3}, f"skill_layer {nid!r} bad layer={layer!r}"
        elif kind == "external":
            assert layer == 0, f"external {nid!r} should have layer=0 (got {layer!r})"
        else:
            assert layer is None, f"code node {nid!r} should have layer=None (got {layer!r})"


_VALID_KINDS = {"module", "class", "function", "method", "skill_layer", "external"}


def test_methods_inside_class_visited(graphify: LocalGraphify, tmp_path: Path) -> None:
    """ast.walk traversal must surface methods as ``kind='method'`` with qualified name."""
    _write(tmp_path / "m.py", "class Foo:\n    def bar(self):\n        return 1\n")
    graph = graphify.build(tmp_path)
    assert "m.Foo.bar" in graph.nodes, f"method node missing; got {sorted(graph.nodes)!r}"
    assert graph.nodes["m.Foo.bar"]["kind"] == "method"


def test_nested_function_visited(graphify: LocalGraphify, tmp_path: Path) -> None:
    """Nested ``def`` inside another function must surface as its own node."""
    src = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    _write(tmp_path / "m.py", src)
    graph = graphify.build(tmp_path)
    assert "m.outer" in graph.nodes
    assert "m.inner" in graph.nodes, f"nested function missing; got {sorted(graph.nodes)!r}"


def test_every_node_has_required_attrs(graphify: LocalGraphify, tmp_path: Path) -> None:
    """Contract: every node ∈ {module, class, function, method, skill_layer, external}."""
    src = (
        "import os\n\nclass A:\n    pass\n\nclass B(A):\n"
        "    def m(self):\n        unknown()\n\n"
        "def top():\n    def nested():\n        return 1\n    return nested()\n"
    )
    _write(tmp_path / "m.py", src)
    _write(tmp_path / "s.metadata.json", '{"name": "s", "depends_on": []}')
    graph = graphify.build(tmp_path)
    for nid, attrs in graph.nodes(data=True):
        missing = REQUIRED_NODE_ATTRS - set(attrs.keys())
        assert not missing, f"node {nid!r} missing attrs: {missing}"
        assert attrs["kind"] in _VALID_KINDS, f"node {nid!r} unknown kind={attrs['kind']!r}"


def test_phantom_call_filtered_or_marked_external(graphify: LocalGraphify, tmp_path: Path) -> None:
    """Calls to undefined symbols → edge target either dropped or marked external."""
    _write(tmp_path / "m.py", "def f():\n    undefined_symbol()\n")
    graph = graphify.build(tmp_path)
    call_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["rel_type"] == "call"]
    for _, dst in call_edges:
        if dst in graph.nodes:
            assert graph.nodes[dst]["kind"] in {"function", "method", "class", "external"}, (
                f"phantom call target {dst!r} kept without external marker"
            )
        # else: edge was dropped → also acceptable per spec
