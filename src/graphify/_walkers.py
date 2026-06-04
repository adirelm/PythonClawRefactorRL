"""Internal AST/JSON walkers for ``LocalGraphify`` (split from ``local_impl`` for
≤150 LOC). Public surface remains ``LocalGraphify.build``; this module holds the
per-file extraction helpers and node/edge contracts.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

from src.graphify.ast_visitor import _fallback_cc as _cyclomatic
from src.graphify.ast_visitor import _loc_of as _node_loc

logger = logging.getLogger(__name__)
_JSON_LAYER_SUFFIX = {".metadata.json": 1, ".instructions.json": 2, ".resources.json": 3}
_CODE_KINDS = {"function", "method", "class"}
_FN = (ast.FunctionDef, ast.AsyncFunctionDef)
EXT_ATTRS = {"kind": "external", "LOC": 0, "cyclomatic": 0, "layer": 0, "lazy_load_flag": False}


def _json_layer(path: Path) -> int | None:
    name = path.name.lower()
    return next((v for k, v in _JSON_LAYER_SUFFIX.items() if name.endswith(k)), None)


def _attrs(
    kind: str, loc: int, cyc: int | None = None, layer: int | None = None, lazy: bool = False
) -> dict:  # McCabe ≥1 floor for code
    c = max(1, cyc) if kind in _CODE_KINDS and cyc is not None else cyc
    return {"kind": kind, "LOC": loc, "cyclomatic": c, "layer": layer, "lazy_load_flag": lazy}


def _collect_defs(tree: ast.AST, mod: str):  # pass 1: classes / methods / top-level + nested fns
    defs, by_class, methods = {}, {}, set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        cq = f"{mod}.{cls.name}"
        defs[cq] = cls
        by_class[cq] = {s.name for s in ast.iter_child_nodes(cls) if isinstance(s, _FN)}
        for s in (x for x in ast.iter_child_nodes(cls) if isinstance(x, _FN)):
            defs[f"{cq}.{s.name}"] = s
            methods.add(id(s))
    for fn in (n for n in ast.walk(tree) if isinstance(n, _FN) and id(n) not in methods):
        defs.setdefault(f"{mod}.{fn.name}", fn)
    return defs, by_class


def _kind_of(q: str, n: ast.AST, by_class: dict[str, set[str]]) -> str:
    if isinstance(n, ast.ClassDef):
        return "class"
    cq, leaf = q.rsplit(".", 1) if "." in q else ("", q)
    return "method" if cq in by_class and leaf in by_class[cq] else "function"


def _resolve(name: str, owner: str, mod: str,
             defs: dict[str, ast.AST], by_class: dict[str, set[str]]) -> str:  # fmt: skip
    cq = owner.rsplit(".", 1)[0] if "." in owner else ""
    if cq in by_class and name in by_class[cq]:
        return f"{cq}.{name}"
    cand = f"{mod}.{name}"
    return cand if cand in defs else f"external:{name}"


def _imports(tree: ast.AST, mod: str, edges: list, ext: set[str]) -> None:
    for n in ast.iter_child_nodes(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                edges.append((mod, a.name, {"rel_type": "import", "weight": 1.0}))
                ext.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module:
            edges.append((mod, n.module, {"rel_type": "import", "weight": 1.0}))
            ext.add(n.module)


def _inheritance(tree: ast.AST, mod: str, defs: dict[str, ast.AST],
                 edges: list, ext: set[str]) -> None:  # fmt: skip
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        cq = f"{mod}.{cls.name}"
        for b in cls.bases:
            tgt = b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
            if tgt:
                edges.append((cq, tgt, {"rel_type": "inheritance", "weight": 1.0}))
                if f"{mod}.{tgt}" not in defs:
                    ext.add(tgt)


def _calls(defs: dict[str, ast.AST], by_class: dict[str, set[str]], mod: str,
           edges: list, ext: set[str]) -> None:  # fmt: skip
    for owner, fn in defs.items():
        if not isinstance(fn, _FN):
            continue
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Call) and (
                nm := getattr(sub.func, "id", None) or getattr(sub.func, "attr", None)
            ):
                tgt = _resolve(nm, owner, mod, defs, by_class)
                edges.append((owner, tgt, {"rel_type": "call", "weight": 1.0}))
                if tgt.startswith("external:"):
                    ext.add(tgt)


def walk_py(py: Path, root: Path):
    try:
        text = py.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        logger.debug("skip %s: %s", py, exc)
        return [], []
    mod = py.relative_to(root).with_suffix("").as_posix().replace("/", ".")
    defs, by_class = _collect_defs(tree, mod)
    nodes: list = [(mod, _attrs("module", len(text.splitlines())))]
    nodes.extend((q, _attrs(_kind_of(q, n, by_class), _node_loc(n), _cyclomatic(n)))
                 for q, n in defs.items())  # fmt: skip
    edges: list = []
    ext: set[str] = set()
    _imports(tree, mod, edges, ext)
    _inheritance(tree, mod, defs, edges, ext)
    _calls(defs, by_class, mod, edges, ext)
    nodes.extend((e, dict(EXT_ATTRS)) for e in ext)
    return nodes, edges


def walk_skill_json(p: Path):  # one skill_layer node + depends_on edges
    layer = _json_layer(p)
    if layer is None:
        return [], []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("skip %s: %s", p, exc)
        return [], []
    name = p.name.split(".")[0]
    nid = f"skill.{name}.L{layer}"
    nodes = [(nid, _attrs("skill_layer", len(json.dumps(payload)), layer=layer, lazy=layer in (2, 3)))]
    deps = payload.get("depends_on", []) if isinstance(payload, dict) else []
    edges = [(nid, f"skill.{d}.L{layer}", {"rel_type": "import", "weight": 1.0}) for d in deps]
    return nodes, edges
