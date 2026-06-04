"""AST visitor — extracts code-level dependency facts (GraphifyAdapter, ADR-002).

Node attrs: {kind, name, qualified_name, loc, cyclomatic, layer, lazy_load_flag}
Edge attrs: {src, dst, rel_type ∈ {call, import, inheritance}, weight}
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

try:
    import radon.complexity as _radon_cc

    _HAS_RADON = True
except ImportError:  # pragma: no cover
    _HAS_RADON = False

NodeInfo = dict[str, Any]
EdgeInfo = dict[str, Any]
_BRANCH: tuple[type[ast.AST], ...] = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.BoolOp, ast.IfExp, ast.comprehension, ast.ExceptHandler)  # fmt: skip


def _fallback_cc(node: ast.AST) -> int:
    """Branch-count fallback when radon is unavailable. Always ≥ 1."""
    return 1 + sum(1 for c in ast.walk(node) if isinstance(c, _BRANCH))


def _radon_cc_by_line(source: str) -> dict[int, int]:
    if not _HAS_RADON:
        return {}
    try:
        return {r.lineno: r.complexity for r in _radon_cc.cc_visit(source)}
    except SyntaxError:  # pragma: no cover
        return {}


def _qual(module: str, name: str) -> str:
    return f"{module}.{name}" if module else name


def _loc_of(node: ast.AST) -> int:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start) or start
    return max(1, end - start + 1)


def _lazy(name: str, fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None) -> bool:
    """ADR-005: lazy if name starts with `_` or has @property decorator."""
    decs = fn.decorator_list if fn is not None else ()
    return name.startswith("_") or any(
        (isinstance(d, ast.Name) and d.id == "property")
        or (isinstance(d, ast.Attribute) and d.attr == "property")
        for d in decs
    )


def _node_dict(kind: str, name: str, qname: str, attrs: tuple[int, int, bool]) -> NodeInfo:
    loc, cc, lazy = attrs
    return {"kind": kind, "name": name, "qualified_name": qname,
            "loc": loc, "cyclomatic": max(1, cc),
            "layer": None, "lazy_load_flag": lazy}  # fmt: skip


class CodeEntityVisitor(ast.NodeVisitor):
    """Walks a parsed module, emitting node/edge facts."""

    def __init__(self, module_qname: str, source: str) -> None:
        self.module = module_qname
        self.nodes: list[NodeInfo] = []
        self.edges: list[EdgeInfo] = []
        self._cc_map = _radon_cc_by_line(source)
        self._scope: list[str] = [module_qname] if module_qname else []
        self._calls: dict[tuple[str, str], int] = {}

    def _cur(self) -> str:
        return self._scope[-1] if self._scope else self.module

    def _emit_fn(self, kind: str, name: str, qname: str, n: ast.AST, lazy: bool) -> None:
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
            cc = self._cc_map.get(n.lineno) or _fallback_cc(n)
        else:
            cc = 1
        self.nodes.append(_node_dict(kind, name, qname, (_loc_of(n), cc, lazy)))

    def visit_Module(self, node: ast.Module) -> None:
        loc = max((_loc_of(c) for c in node.body), default=1)
        self.nodes.append(_node_dict("module", self.module, self.module, (loc, 1, False)))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = _qual(self._cur(), node.name)
        self._emit_fn("class", node.name, qname, node, _lazy(node.name))
        for base in node.bases:
            self.edges.append({"src": qname, "dst": ast.unparse(base),
                               "rel_type": "inheritance", "weight": 1})  # fmt: skip
        self._scope.append(qname)
        self.generic_visit(node)
        self._scope.pop()

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qname = _qual(self._cur(), node.name)
        self._emit_fn("function", node.name, qname, node, _lazy(node.name, node))
        self._scope.append(qname)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.edges.append({"src": self.module, "dst": alias.name,
                               "rel_type": "import", "weight": 1})  # fmt: skip

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        for alias in node.names:
            dst = _qual(base, alias.name) if base else alias.name
            self.edges.append({"src": self.module, "dst": dst,
                               "rel_type": "import", "weight": 1})  # fmt: skip

    def visit_Call(self, node: ast.Call) -> None:
        key = (self._cur(), ast.unparse(node.func))
        self._calls[key] = self._calls.get(key, 0) + 1
        self.generic_visit(node)

    def finalize(self) -> None:
        for (src, dst), w in self._calls.items():
            self.edges.append({"src": src, "dst": dst, "rel_type": "call", "weight": w})


def walk_module(src_path: Path) -> tuple[list[NodeInfo], list[EdgeInfo]]:
    """Parse a .py file and return (nodes, edges) extracted from its AST."""
    source = Path(src_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    v = CodeEntityVisitor(Path(src_path).stem, source)
    v.visit(tree)
    v.finalize()
    return v.nodes, v.edges
