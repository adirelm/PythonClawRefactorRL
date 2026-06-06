"""Unit tests for src.model.policy_net.PolicyNet (Phase 3).

Pin the contract of the actor-critic head: shape stability against the
frozen ``A_max = 45057`` action space, Huang & Ontañón pre-softmax masking
correctness (illegal actions can never be sampled), critic output shape,
log-prob finiteness on legal actions, and seeded determinism for the
``get_action`` path.
"""

from __future__ import annotations

import torch

from src.model.policy_net import PolicyNet

_A_MAX = 45057
_FEATURE_DIM = 16
_V_MAX = 8  # tiny V_max per test for speed; PolicyNet is V-shape-agnostic
_BATCH = 2


def _make_inputs(batch: int = _BATCH, v_max: int = _V_MAX, real_nodes: int = 4):
    """Synthesize ``(x_padded, node_mask)`` with the first ``real_nodes`` real."""
    x = torch.randn(batch, v_max, _FEATURE_DIM)
    mask = torch.zeros(batch, v_max, dtype=torch.bool)
    mask[:, :real_nodes] = True
    return x, mask


def _legal_action_mask(batch: int = _BATCH, *, legal: list[int]) -> torch.Tensor:
    """Build an ``(B, A_max)`` bool mask with True only at the given indices."""
    am = torch.zeros(batch, _A_MAX, dtype=torch.bool)
    am[:, legal] = True
    return am


def test_forward_returns_logits_and_value() -> None:
    """forward() yields a (logits, value) pair with non-NaN entries."""
    net = PolicyNet()
    x, mask = _make_inputs()
    logits, value = net(x, mask)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(value).any()


def test_logits_shape_matches_a_max() -> None:
    """Logits are (B, A_max=45057); value is (B, 1)."""
    net = PolicyNet()
    x, mask = _make_inputs()
    logits, value = net(x, mask)
    assert logits.shape == (_BATCH, _A_MAX)
    assert value.shape == (_BATCH, 1)


def test_action_mask_pre_softmax_zeros_illegal() -> None:
    """Sampled actions are always within the legal set, over many samples.

    Pins the Huang & Ontañón (2022) guarantee: illegal logits become -inf
    pre-softmax, so ``Categorical`` assigns them zero probability.
    """
    torch.manual_seed(0)
    net = PolicyNet()
    x, mask = _make_inputs()
    logits, _ = net(x, mask)
    legal = [0, 17, 1234, _A_MAX - 1]  # arbitrary spread incl. NOOP slot
    action_mask = _legal_action_mask(legal=legal)
    legal_set = set(legal)
    for _ in range(50):
        action_idx, _ = net.get_action(logits, action_mask)
        for a in action_idx.tolist():
            assert a in legal_set, f"sampled illegal action {a} (legal={legal_set})"


def test_value_is_scalar() -> None:
    """Critic returns a scalar (last-dim 1) per batch element."""
    net = PolicyNet()
    x, mask = _make_inputs()
    _, value = net(x, mask)
    assert value.shape[-1] == 1
    assert value.ndim == 2  # (B, 1)


def test_log_prob_finite() -> None:
    """log_prob on the chosen legal action is finite (no -inf or NaN)."""
    torch.manual_seed(1)
    net = PolicyNet()
    x, mask = _make_inputs()
    logits, _ = net(x, mask)
    action_mask = _legal_action_mask(legal=[3, 42, 9999, _A_MAX - 1])
    _, log_prob = net.get_action(logits, action_mask)
    assert torch.isfinite(log_prob).all()


def test_deterministic_with_seed() -> None:
    """Same weights + same x + same seed → same sampled action_idx."""
    x, mask = _make_inputs()
    action_mask = _legal_action_mask(legal=[0, 1, 2, 3, 4, 5, _A_MAX - 1])

    torch.manual_seed(123)
    net_a = PolicyNet()
    logits_a, _ = net_a(x, mask)
    torch.manual_seed(999)
    action_a, _ = net_a.get_action(logits_a, action_mask)

    torch.manual_seed(123)
    net_b = PolicyNet()
    logits_b, _ = net_b(x, mask)
    torch.manual_seed(999)
    action_b, _ = net_b.get_action(logits_b, action_mask)

    assert torch.equal(action_a, action_b)
    assert torch.allclose(logits_a, logits_b)
