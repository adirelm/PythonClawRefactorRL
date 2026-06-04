"""Local NetworkX-based GraphifyAdapter implementation (Phase 1, ADR-002).

AST-walks .py modules + Skills JSON shim files → nx.DiGraph with node attrs
``{kind, LOC, cyclomatic, layer, lazy_load_flag}`` and edge attrs
``{rel_type ∈ {call, import, inheritance}, weight}``.

``_walk_module`` is inlined (Phase-1 owns only local_impl.py + its unit test);
promote to ``src/graphify/ast_visitor.py`` if a second consumer appears.
"""

from __future__ import annotations

import ast
import json
import logging
import pickle
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

_JSON_LAYER_SUFFIX = {".metadata.json": 1, ".instructions.json": 2, ".resources.json": 3}
_BRANCH_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp, ast.IfExp)


def _json_layer(path: Path) -> int | None:
    name = path.name.lower()
    for suffix, layer in _JSON_LAYER_SUFFIX.items():
        if name.endswith(suffix):
            return layer
    return None


def _cyclomatic(func: ast.AST) -> int:
    """McCabe-lite: 1 + count of branching nodes inside ``func``."""
    return 1 + sum(1 for n in ast.walk(func) if isinstance(n, _BRANCH_NODES))


def _node_loc(node: ast.AST) -> int:
    return getattr(node, "end_lineno", node.lineno) - node.lineno + 1


def _attrs(kind: str, loc: int, cyc: int = 0, layer: int = 0, lazy: bool = False) -> dict:
    return {"kind": kind, "LOC": loc, "cyclomatic": cyc, "layer": layer, "lazy_load_flag": lazy}


def _walk_module(py_path: Path, src_root: Path):
    """Parse one .py file → (nodes, edges) honoring the attribute contract."""
    try:
        text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        logger.debug("skip %s: %s", py_path, exc)
        return [], []
    mod_id = py_path.relative_to(src_root).with_suffix("").as_posix().replace("/", ".")
    nodes: list = [(mod_id, _attrs("module", len(text.splitlines())))]
    edges: list = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append((mod_id, alias.name, {"rel_type": "import", "weight": 1.0}))
        elif isinstance(node, ast.ImportFrom) and node.module:
            edges.append((mod_id, node.module, {"rel_type": "import", "weight": 1.0}))
        elif isinstance(node, ast.ClassDef):
            cls_id = f"{mod_id}.{node.name}"
            nodes.append((cls_id, _attrs("class", _node_loc(node), _cyclomatic(node))))
            for base in node.bases:
                tgt = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if tgt:
                    edges.append((cls_id, tgt, {"rel_type": "inheritance", "weight": 1.0}))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            fn_id = f"{mod_id}.{node.name}"
            nodes.append((fn_id, _attrs("function", _node_loc(node), _cyclomatic(node))))
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    tgt = getattr(sub.func, "id", None) or getattr(sub.func, "attr", None)
                    if tgt:
                        edges.append((fn_id, f"{mod_id}.{tgt}", {"rel_type": "call", "weight": 1.0}))
    return nodes, edges


def _walk_skill_json(json_path: Path):
    """Emit one ``skill_layer`` node + ``depends_on`` edges for a Skills JSON file."""
    layer = _json_layer(json_path)
    if layer is None:
        return [], []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("skip %s: %s", json_path, exc)
        return [], []
    skill_name = json_path.name.split(".")[0]
    node_id = f"skill.{skill_name}.L{layer}"
    nodes = [(node_id, _attrs("skill_layer", len(json.dumps(payload)), layer=layer, lazy=layer in (2, 3)))]
    deps = payload.get("depends_on", []) if isinstance(payload, dict) else []
    edges = [(node_id, f"skill.{dep}.L{layer}", {"rel_type": "import", "weight": 1.0}) for dep in deps]
    return nodes, edges


class LocalGraphify:
    """NetworkX-backed concrete GraphifyAdapter (ADR-002, Phase 1)."""

    def build(self, src_root: Path, *, seed: int = 0) -> nx.DiGraph:
        """Walk ``src_root`` for .py + .json files → nx.DiGraph."""
        logger.info("LocalGraphify.build(src_root=%s, seed=%d)", src_root, seed)
        root = Path(src_root)
        graph: nx.DiGraph = nx.DiGraph()
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            self._absorb(graph, *_walk_module(py, root))
        for js in sorted(root.rglob("*.json")):
            self._absorb(graph, *_walk_skill_json(js))
        return graph

    @staticmethod
    def _absorb(graph: nx.DiGraph, nodes: list, edges: list) -> None:
        for nid, attrs in nodes:
            graph.add_node(nid, **attrs)
        for s, d, attrs in edges:
            graph.add_edge(s, d, **attrs)

    def load(self, pickle_path: Path) -> nx.DiGraph:
        """Read a previously-pickled DiGraph; raise informative FileNotFoundError.

        Security: pickle is intentional — ADR-002 mandates ``.load(pickle_path)``
        round-trip with .build() output (non-JSON-serializable AST dicts). The
        pickle is written by the adapter's own caller within the repo cache and
        never sourced from an untrusted / network location.
        """
        path = Path(pickle_path)
        try:
            with path.open("rb") as fh:
                graph = pickle.load(fh)  # internal-only cache, see docstring
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"LocalGraphify.load: no pickle at {path!s}") from exc
        if not isinstance(graph, nx.DiGraph):
            raise TypeError(f"LocalGraphify.load: pickle at {path!s} is not nx.DiGraph")
        return graph
