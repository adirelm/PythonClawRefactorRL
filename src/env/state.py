"""State dataclass per docs/STATE_DESIGN.md §2.

State = (A, X, edge_attrs) where:
- A: weighted CSR adjacency (|V|, |V|), rel-weighted sum (no self-loops).
- X: torch.FloatTensor (|V|, 16), columns per STATE_DESIGN §3 contract.
- edge_type_counts: {"call": N, "import": M, "inheritance": K}.

Variable-|V| primary path (ADR-004) — no padding here; the policy/value
head consumes variable-sized graphs natively via PyG DataLoader. Padding
to V_max=512 is the ADR-008 fallback, owned by the env, not this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from src.utils.config_loader import load_config

# Column indices for X (frozen contract — STATE_DESIGN §3).
_C_LOC, _C_CYC, _C_DIN, _C_DOUT, _C_BTW = 0, 1, 2, 3, 4
_C_L1, _C_L2, _C_L3 = 5, 6, 7
_C_LAZY = 8
_C_KCLS, _C_KMOD, _C_KFN = 9, 10, 11
_C_INSK, _C_COMP, _C_AGE, _C_RES = 12, 13, 14, 15

_FEATURE_DIM = 16
_REL_TYPES = ("call", "import", "inheritance")
_FN_KINDS = {"function", "method"}
_CYC_CLIP = 20.0  # STATE_DESIGN §3 — cyclomatic /20 clipped to [0,1]
_LAYER_COLS = {1: _C_L1, 2: _C_L2, 3: _C_L3}  # skill layer one-hot map


@dataclass(frozen=True)
class State:
    """Immutable observation tuple consumed by the policy/value head."""

    A: sp.csr_matrix
    X: torch.Tensor
    node_ids: tuple[str, ...]
    edge_type_counts: dict[str, int]

    @property
    def num_nodes(self) -> int:
        """Number of graph nodes (rows of ``A`` / length of ``node_ids``)."""
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        """Number of directed edges (non-zero entries of adjacency ``A``)."""
        return int(self.A.nnz)

    @classmethod
    def from_digraph(cls, graph: nx.DiGraph) -> State:
        """Build State from an nx.DiGraph emitted by GraphifyAdapter (ADR-002)."""
        node_ids = tuple(sorted(graph.nodes()))
        idx = {nid: i for i, nid in enumerate(node_ids)}
        cfg = load_config()
        weights = cfg["state"]["relation_weights"]
        loc_log1p = bool(cfg["state"]["normalization"].get("loc_log1p", True))

        adj, counts = _build_adjacency(graph, idx, weights)
        features = _build_features(graph, node_ids, adj, loc_log1p)
        return cls(A=adj, X=features, node_ids=node_ids, edge_type_counts=counts)

    def to_pyg_data(self) -> dict:
        """Variable-|V| primary path (ADR-004) — emit edge_index + x for PyG."""
        coo = self.A.tocoo()
        edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        edge_weight = torch.tensor(coo.data, dtype=torch.float32)
        return {
            "x": self.X,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "num_nodes": self.num_nodes,
            "edge_type_counts": dict(self.edge_type_counts),
        }


def _build_adjacency(
    graph: nx.DiGraph, idx: dict[str, int], weights: dict[str, float]
) -> tuple[sp.csr_matrix, dict[str, int]]:
    n = len(idx)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    counts: dict[str, int] = dict.fromkeys(_REL_TYPES, 0)
    for u, v, attrs in graph.edges(data=True):
        if u == v or u not in idx or v not in idx:
            continue
        rel = attrs.get("rel_type", "call")
        if rel not in weights:
            continue
        counts[rel] = counts.get(rel, 0) + 1
        rows.append(idx[u])
        cols.append(idx[v])
        data.append(float(weights[rel]) * float(attrs.get("weight", 1.0)))
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    adj.sum_duplicates()
    return adj, counts


def _build_features(
    graph: nx.DiGraph, node_ids: tuple[str, ...], adj: sp.csr_matrix, loc_log1p: bool
) -> torch.Tensor:
    n = len(node_ids)
    x = np.zeros((n, _FEATURE_DIM), dtype=np.float32)
    deg_out = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_in = np.asarray(adj.sum(axis=0)).reshape(-1)
    max_deg = float(max(deg_in.max(initial=0.0), deg_out.max(initial=0.0), 1.0))
    locs = np.array([float(graph.nodes[nid].get("LOC", 0) or 0) for nid in node_ids], dtype=np.float32)
    loc_col = np.log1p(locs) if loc_log1p else locs
    max_loc = float(loc_col.max(initial=0.0) or 1.0)
    cycs = np.array([float(graph.nodes[nid].get("cyclomatic") or 0) for nid in node_ids], dtype=np.float32)
    max_comp = float((cycs * locs).max(initial=0.0) or 1.0)
    for i, nid in enumerate(node_ids):
        attrs = graph.nodes[nid]
        x[i, _C_LOC] = loc_col[i] / max_loc
        x[i, _C_CYC] = float(min(cycs[i], _CYC_CLIP)) / _CYC_CLIP
        x[i, _C_DIN] = deg_in[i] / max_deg
        x[i, _C_DOUT] = deg_out[i] / max_deg
        # betweenness_cached (col 4) starts at 0; CentralityScheduler fills it.
        layer_col = _LAYER_COLS.get(attrs.get("layer"))
        if layer_col is not None:
            x[i, layer_col] = 1.0
        x[i, _C_LAZY] = 1.0 if attrs.get("lazy_load_flag") else 0.0
        kind = attrs.get("kind")
        if kind == "class":
            x[i, _C_KCLS] = 1.0
        elif kind == "module":
            x[i, _C_KMOD] = 1.0
        elif kind in _FN_KINDS:
            x[i, _C_KFN] = 1.0
        x[i, _C_INSK] = 1.0 if kind == "skill_layer" else 0.0
        x[i, _C_COMP] = float(cycs[i] * locs[i]) / max_comp
        # age_episodes (col 14) starts at 0; env mutates per-step on edits.
        # reserved (col 15) always 0.0.
    return torch.from_numpy(x)
