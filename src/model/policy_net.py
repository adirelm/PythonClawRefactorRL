"""Actor-critic policy network with pre-softmax action masking (Phase 3).

Implements the Huang & Ontañón (2022) "invalid action masking" pattern: the
illegal logits are set to ``-inf`` *before* the softmax so the resulting
categorical distribution has zero probability mass on illegal actions and
their gradients are detached. The pooling stage is intentionally a simple
mean-pool over node features — Phase 4 swaps this for a GraphSAGE encoder
behind the same forward signature.

Frozen constants (CLAUDE.md sealed values):
- feature_dim = 16     (node feature width)
- A_max       = 45057  (SPLIT 4096 + MERGE 8192 + REWIRE 32768 + NOOP 1)
- V_max       = 512    (fallback padding bound; used only by callers)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

_FEATURE_DIM_DEFAULT = 16
_HIDDEN_DIM_DEFAULT = 64
_ACTION_DIM_DEFAULT = 45057  # A_max — kept here to avoid src.env import cycle


class PolicyNet(nn.Module):
    """Two-headed actor-critic MLP over mean-pooled node features.

    Forward pass
    ------------
    ``x_padded`` is a ``(B, V_max, feature_dim)`` float tensor and ``mask`` is
    a ``(B, V_max)`` bool tensor with True at *real* node slots. We mean-pool
    over the True rows to get a ``(B, feature_dim)`` graph embedding, then
    branch into the actor (logits over ``A_max``) and critic (scalar value).
    """

    def __init__(
        self,
        feature_dim: int = _FEATURE_DIM_DEFAULT,
        hidden_dim: int = _HIDDEN_DIM_DEFAULT,
        action_dim: int = _ACTION_DIM_DEFAULT,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, x_padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool real (mask=True) node rows over the V_max axis.

        Empty graphs (mask all-False) divide by clamp(min=1) → produce zeros,
        which keeps the forward pass numerically safe at the cost of an
        uninformative embedding (acceptable: action mask will leave NOOP only).
        """
        mask_f = mask.to(x_padded.dtype).unsqueeze(-1)  # (B, V_max, 1)
        summed = (x_padded * mask_f).sum(dim=1)  # (B, feature_dim)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x_padded.dtype)
        return summed / denom

    def forward(self, x_padded: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)`` for a batch of padded graphs.

        Shapes
        ------
        - ``x_padded``: ``(B, V_max, feature_dim)``  float32
        - ``mask``:     ``(B, V_max)``               bool
        - returns ``logits`` ``(B, A_max)``, ``value`` ``(B, 1)``
        """
        embed = self._pool(x_padded, mask)
        logits = self.actor(embed)
        value = self.critic(embed)
        return logits, value

    def get_action(
        self, logits: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a legal action from ``logits`` under ``action_mask``.

        Implements the pre-softmax mask from Huang & Ontañón (2022): every
        ``False`` slot in ``action_mask`` has its logit overwritten with
        ``-inf`` *before* the softmax, so ``Categorical`` cannot ever sample
        an illegal action and gradients on illegal logits are zero.

        Parameters
        ----------
        logits : ``(B, A_max)`` float tensor — raw actor head output.
        action_mask : ``(B, A_max)`` bool tensor — True = legal.

        Returns
        -------
        ``(action_idx, log_prob)`` both shaped ``(B,)``.
        """
        if logits.shape != action_mask.shape:
            raise ValueError(f"logits {tuple(logits.shape)} != action_mask {tuple(action_mask.shape)}")
        masked = logits.masked_fill(~action_mask, float("-inf"))
        dist = Categorical(logits=masked)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx, log_prob
