"""GraphEncoder — variable-|V| state encoder per ADR-004.

Primary path: 2-layer GraphSAGE via PyTorch Geometric (``torch_geometric``)
when installed. Fallback path (this CI / wheel-matrix-constrained
environments): a 2-layer masked MLP over the padded ``(B, V_max, in_dim)``
node-feature tensor.

Both paths satisfy the same contract:

- input  ``x_padded`` shape ``(B, V, in_dim)``   (V may be V_max=512)
- input  ``mask``      shape ``(B, V)``          1 for real nodes, 0 for padding
- input  ``edge_index`` (PyG path only)          ``(2, E)`` LongTensor
- output                  shape ``(B, out_dim)`` graph-level embedding

Per ADR-004 §V_max reconciliation: padded rows enter as zero-feature
nodes and are zeroed out by the mask before mean-pool readout, so the
fallback MLP path produces the same shape contract as the GraphSAGE path.
"""

from __future__ import annotations

import torch
from torch import nn

try:  # pragma: no cover - import guard exercised via monkeypatch test
    from torch_geometric.nn import SAGEConv  # type: ignore[import-not-found]

    _HAS_PYG = True
except ImportError:  # pragma: no cover - fallback path is the tested branch in CI
    SAGEConv = None  # type: ignore[assignment,misc]
    _HAS_PYG = False

_DEFAULT_SEED = 0
_RANK_PADDED = 3  # (B, V, in_dim)
_RANK_MASK = 2  # (B, V)


class GraphEncoder(nn.Module):
    """ADR-004 encoder; PyG SAGEConv primary, padded-MLP fallback."""

    def __init__(
        self,
        in_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 128,
        use_graphsage: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_graphsage = bool(use_graphsage and _HAS_PYG)
        if self.use_graphsage:
            self._build_graphsage()
        else:
            # Determinism in the fallback path so two encoders with the same
            # seed produce identical embeddings (ADR-004 fallback reproducibility).
            torch.manual_seed(_DEFAULT_SEED)
            self._build_mlp()

    def _build_graphsage(self) -> None:
        assert SAGEConv is not None  # pragma: no cover - guarded by use_graphsage flag
        self.conv1 = SAGEConv(self.in_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.out_dim)
        self.act = nn.ReLU()

    def _build_mlp(self) -> None:
        self.lin1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.act = nn.ReLU()

    def forward(
        self,
        x_padded: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return graph-level embedding ``(B, out_dim)`` via masked mean-pool."""
        if x_padded.dim() != _RANK_PADDED:
            raise ValueError(f"x_padded must be (B, V, in_dim); got {tuple(x_padded.shape)}")
        if mask.dim() != _RANK_MASK or mask.shape != x_padded.shape[:2]:
            raise ValueError(f"mask must be (B, V) matching x_padded; got mask={tuple(mask.shape)}")
        if self.use_graphsage and edge_index is not None:  # pragma: no cover - PyG path
            return self._forward_graphsage(x_padded, mask, edge_index)
        return self._forward_mlp(x_padded, mask)

    def _forward_mlp(self, x_padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Zero out padded rows BEFORE the projection so padded zeros stay zero
        # through the (biased) Linear — they only re-acquire bias mass at non-pad
        # rows. Masked mean-pool over the (B, V, out_dim) tensor gives (B, out_dim).
        mask_f = mask.to(dtype=x_padded.dtype).unsqueeze(-1)  # (B, V, 1)
        h = self.act(self.lin1(x_padded * mask_f))
        h = self.lin2(h * mask_f)
        h = h * mask_f  # re-zero after second projection (bias contamination guard)
        denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1) — guard empty graphs
        return h.sum(dim=1) / denom  # (B, out_dim)

    def _forward_graphsage(
        self, x_padded: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover - exercised only when PyG installed
        batch, v_max, _ = x_padded.shape
        x_flat = x_padded.reshape(batch * v_max, -1)
        h = self.act(self.conv1(x_flat, edge_index))
        h = self.conv2(h, edge_index)
        h = h.reshape(batch, v_max, -1)
        mask_f = mask.to(dtype=h.dtype).unsqueeze(-1)
        h = h * mask_f
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return h.sum(dim=1) / denom
