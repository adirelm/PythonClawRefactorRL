"""Unit tests for src/model/encoder.py — GraphEncoder per ADR-004.

The CI / wheel-constrained environment does not ship ``torch_geometric``,
so the runtime path under test is the **padded-MLP fallback**. The
GraphSAGE primary path is exercised only when PyG is installed; the
fallback branch is what these tests pin.

Contracts being pinned:
- Output shape ``(B, out_dim)`` independent of V (padded V_max=512).
- Variable V tolerated as long as the mask zeros padded rows.
- Padded zero rows do not contribute to the graph-level embedding.
- Canonical hidden dim = 64 stays a hard constant.
- Falling back to MLP when PyG import fails is the documented path.
- Two encoders constructed with the same fallback seed are bit-identical.
"""

from __future__ import annotations

import builtins
import importlib

import pytest
import torch

import src.model.encoder as encoder_mod
from src.model.encoder import GraphEncoder

_V_MAX = 512
_IN_DIM = 16
_HIDDEN_DIM = 64
_OUT_DIM = 128


def _padded_batch(batch: int, real_v: int, in_dim: int = _IN_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (B, V_max, in_dim) padded tensor + (B, V_max) mask."""
    torch.manual_seed(123)
    x = torch.zeros(batch, _V_MAX, in_dim)
    mask = torch.zeros(batch, _V_MAX)
    x[:, :real_v, :] = torch.randn(batch, real_v, in_dim)
    mask[:, :real_v] = 1.0
    return x, mask


def test_encoder_output_shape() -> None:
    enc = GraphEncoder(in_dim=_IN_DIM, hidden_dim=_HIDDEN_DIM, out_dim=_OUT_DIM, use_graphsage=False)
    x, mask = _padded_batch(batch=2, real_v=20)
    out = enc(x, mask)
    assert out.shape == (2, _OUT_DIM)


def test_encoder_handles_variable_v() -> None:
    enc = GraphEncoder(in_dim=_IN_DIM, hidden_dim=_HIDDEN_DIM, out_dim=_OUT_DIM, use_graphsage=False)
    x10, m10 = _padded_batch(batch=1, real_v=10)
    x20, m20 = _padded_batch(batch=1, real_v=20)
    out10 = enc(x10, m10)
    out20 = enc(x20, m20)
    # Same fixed output width regardless of real |V| — that is the whole point.
    assert out10.shape == out20.shape == (1, _OUT_DIM)


def test_encoder_mask_zeros_padded_positions() -> None:
    """Padded slots must not change the embedding — fill padded rows with noise
    and confirm the masked embedding equals the clean-pad embedding."""
    enc = GraphEncoder(in_dim=_IN_DIM, hidden_dim=_HIDDEN_DIM, out_dim=_OUT_DIM, use_graphsage=False)
    real_v = 12
    x_clean, mask = _padded_batch(batch=1, real_v=real_v)
    x_noisy = x_clean.clone()
    # Stuff garbage into padded rows; mask should make this invisible.
    x_noisy[:, real_v:, :] = torch.randn(1, _V_MAX - real_v, _IN_DIM) * 10.0
    out_clean = enc(x_clean, mask)
    out_noisy = enc(x_noisy, mask)
    assert torch.allclose(out_clean, out_noisy, atol=1e-6)


def test_encoder_uses_canonical_hidden_dim() -> None:
    enc = GraphEncoder(use_graphsage=False)
    assert enc.hidden_dim == _HIDDEN_DIM
    assert enc.in_dim == _IN_DIM
    assert enc.out_dim == _OUT_DIM
    # The hidden Linear width is the contract surface most likely to drift.
    assert enc.lin1.out_features == _HIDDEN_DIM
    assert enc.lin2.in_features == _HIDDEN_DIM
    assert enc.lin2.out_features == _OUT_DIM


def test_fallback_mlp_when_pyg_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch the import system so ``torch_geometric`` cannot resolve;
    the reloaded module must report ``_HAS_PYG == False`` and use the MLP path."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch_geometric" or name.startswith("torch_geometric."):
            raise ImportError("simulated: torch_geometric not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    reloaded = importlib.reload(encoder_mod)
    try:
        assert reloaded._HAS_PYG is False
        enc = reloaded.GraphEncoder(use_graphsage=True)  # asked for SAGE; should downgrade
        assert enc.use_graphsage is False
        assert hasattr(enc, "lin1") and hasattr(enc, "lin2")
    finally:
        # Restore the real module so other tests aren't poisoned.
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(encoder_mod)


def test_forward_rejects_wrong_x_rank() -> None:
    enc = GraphEncoder(use_graphsage=False)
    bad_x = torch.zeros(_V_MAX, _IN_DIM)  # (V, F) instead of (B, V, F)
    mask = torch.ones(1, _V_MAX)
    with pytest.raises(ValueError, match="x_padded must be"):
        enc(bad_x, mask)


def test_forward_rejects_mismatched_mask() -> None:
    enc = GraphEncoder(use_graphsage=False)
    x = torch.zeros(1, _V_MAX, _IN_DIM)
    bad_mask = torch.ones(1, _V_MAX // 2)  # wrong V dimension
    with pytest.raises(ValueError, match="mask must be"):
        enc(x, bad_mask)


def test_deterministic_with_seed() -> None:
    """Two fallback encoders built back-to-back must produce identical weights
    because __init__ seeds torch before building the Linears."""
    enc_a = GraphEncoder(use_graphsage=False)
    enc_b = GraphEncoder(use_graphsage=False)
    x, mask = _padded_batch(batch=1, real_v=8)
    out_a = enc_a(x, mask)
    out_b = enc_b(x, mask)
    assert torch.allclose(out_a, out_b, atol=1e-7)
    # Weight-level identity is the stronger claim that justifies the embedding match.
    assert torch.allclose(enc_a.lin1.weight, enc_b.lin1.weight)
    assert torch.allclose(enc_a.lin2.weight, enc_b.lin2.weight)
