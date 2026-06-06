"""Architectural guarantee: PolicyNet.get_action is Categorical(-inf)-safe.

Pins the fix for the 5-seed PPO hang (seeds 123 / 314 / 271): if any batch
row has an all-False ``action_mask``, the pre-softmax mask becomes all
``-inf`` and ``Categorical(logits=...)`` produces NaN probabilities, which
deadlocks ``dist.sample()``. Per Huang & Ontañón (2022) §3 invalid action
masking, NOOP is the always-legal escape slot (ACTION_DESIGN §2.4); we
force it True on any empty row so training degenerates gracefully instead
of hanging.

These tests also guarantee that single-True and normal masks keep their
existing semantics — the defensive guard is a no-op on healthy inputs.
"""

from __future__ import annotations

import torch

from src.env.actions import NOOP_INDEX
from src.model.policy_net import PolicyNet

_A_MAX = 45057
_BATCH = 4


def _zero_logits(batch: int = 1) -> torch.Tensor:
    """Logits = 0 ⇒ uniform over whatever the mask leaves legal."""
    return torch.zeros(batch, _A_MAX)


def test_all_false_mask_does_not_hang_and_returns_noop() -> None:
    """All-False mask must NOT hang the sampler; NOOP slot is the escape hatch."""
    torch.manual_seed(42)
    policy = PolicyNet()
    logits = _zero_logits(batch=1)
    mask = torch.zeros(1, _A_MAX, dtype=torch.bool)  # entirely illegal — the failure mode
    action_idx, log_prob = policy.get_action(logits, mask)
    assert action_idx.shape == (1,)
    assert int(action_idx.item()) == NOOP_INDEX
    assert torch.isfinite(log_prob).all(), "log_prob must be finite under NOOP fallback"


def test_single_true_mask_is_deterministic() -> None:
    """Mask with exactly one True ⇒ that slot is sampled deterministically."""
    torch.manual_seed(0)
    policy = PolicyNet()
    chosen = 12345
    logits = _zero_logits(batch=1)
    mask = torch.zeros(1, _A_MAX, dtype=torch.bool)
    mask[0, chosen] = True
    for _ in range(10):
        action_idx, log_prob = policy.get_action(logits, mask)
        assert int(action_idx.item()) == chosen
        assert torch.isfinite(log_prob).all()


def test_normal_mask_distribution_is_finite_and_sums_to_one() -> None:
    """Healthy mask ⇒ softmax over legal slots is finite + normalised."""
    torch.manual_seed(1)
    policy = PolicyNet()
    logits = torch.randn(_BATCH, _A_MAX)
    mask = torch.zeros(_BATCH, _A_MAX, dtype=torch.bool)
    legal = [0, 17, 1234, _A_MAX - 1]  # spread incl. NOOP
    for col in legal:
        mask[:, col] = True
    action_idx, log_prob = policy.get_action(logits, mask)
    assert action_idx.shape == (_BATCH,)
    assert torch.isfinite(log_prob).all()
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked_logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(_BATCH), atol=1e-5)
    assert torch.isfinite(probs).all()


def test_mixed_batch_with_one_empty_row_still_samples_finitely() -> None:
    """Batch where one row is empty + others healthy: NaN must not contaminate."""
    torch.manual_seed(7)
    policy = PolicyNet()
    logits = _zero_logits(batch=3)
    mask = torch.zeros(3, _A_MAX, dtype=torch.bool)
    mask[0, 100] = True  # healthy
    # mask[1] stays all-False — the failure-mode row
    mask[2, 200] = True  # healthy
    action_idx, log_prob = policy.get_action(logits, mask)
    assert int(action_idx[0].item()) == 100
    assert int(action_idx[1].item()) == NOOP_INDEX
    assert int(action_idx[2].item()) == 200
    assert torch.isfinite(log_prob).all()
