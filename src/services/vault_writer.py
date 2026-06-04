"""DiGraph → Obsidian Vault Markdown writer.

Renders each node as a markdown note with YAML front-matter (graph
attrs) and groups outgoing edges by ``rel_type`` into ``## Calls``,
``## Imports``, and ``## Inherits from`` sections of Obsidian-style
``[[wiki-links]]``. A vault-level ``README.md`` summarises ``|V|``,
``|E|``, and the top-5 most-depended-on nodes (by in-degree).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import networkx as nx

_REL_HEADERS: dict[str, str] = {
    "call": "## Calls",
    "import": "## Imports",
    "inheritance": "## Inherits from",
}
_REL_ORDER: tuple[str, ...] = ("call", "import", "inheritance")
_FRONT_MATTER_KEYS: tuple[str, ...] = ("kind", "LOC", "cyclomatic", "layer", "lazy_load_flag")
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(name: str) -> str:
    """Replace path separators and special chars with ``_`` for safe filenames."""
    cleaned = _SANITIZE_RE.sub("_", str(name)).strip("_")
    return cleaned or "_"


def _yaml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _front_matter(node_attrs: dict) -> str:
    lines = ["---"]
    for key in _FRONT_MATTER_KEYS:
        if key in node_attrs:
            lines.append(f"{key}: {_yaml_value(node_attrs[key])}")
    lines.append("---")
    return "\n".join(lines)


def render_node(node_attrs: dict, neighbors_by_rel: dict[str, list[str]]) -> str:
    """Return the markdown body for a single node.

    Args:
        node_attrs: Mapping from {kind, LOC, cyclomatic, layer, lazy_load_flag}.
        neighbors_by_rel: Mapping rel_type → list of neighbor display names.
    """
    parts: list[str] = [_front_matter(node_attrs), ""]
    for rel in _REL_ORDER:
        neighbors = neighbors_by_rel.get(rel, [])
        if not neighbors:
            continue
        parts.append(_REL_HEADERS[rel])
        parts.extend(f"- [[{sanitize(n)}]]" for n in neighbors)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _group_outgoing(graph: nx.DiGraph, node: object) -> dict[str, list[str]]:
    by_rel: dict[str, list[str]] = {rel: [] for rel in _REL_ORDER}
    for _, dst, data in graph.out_edges(node, data=True):
        rel = data.get("rel_type", "call")
        by_rel.setdefault(rel, []).append(str(dst))
    return by_rel


def _render_readme(graph: nx.DiGraph) -> str:
    in_deg = Counter({str(n): int(d) for n, d in graph.in_degree()})
    top5 = in_deg.most_common(5)
    lines = [
        "# Vault Summary",
        "",
        f"Total nodes: {graph.number_of_nodes()}",
        f"Total edges: {graph.number_of_edges()}",
        "",
        "## Top-5 most-depended-on",
        "",
    ]
    if not top5:
        lines.append("- (none)")
    else:
        lines.extend(f"- [[{sanitize(name)}]] (in-degree: {deg})" for name, deg in top5)
    lines.append("")
    return "\n".join(lines)


def write_vault(graph: nx.DiGraph, output_dir: Path) -> None:
    """Write one ``.md`` per node + a vault-level ``README.md``.

    Args:
        graph: NetworkX DiGraph with node attrs and edge ``rel_type``.
        output_dir: Destination vault directory (created if missing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for node, attrs in graph.nodes(data=True):
        neighbors = _group_outgoing(graph, node)
        body = render_node(dict(attrs), neighbors)
        (output_dir / f"{sanitize(node)}.md").write_text(body, encoding="utf-8")
    (output_dir / "README.md").write_text(_render_readme(graph), encoding="utf-8")
