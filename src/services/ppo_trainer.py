"""Custom PPO trainer — no SB3, no gymnasium (brief §2.2). Schulman et al.
(2017) PPO clipped surrogate against our 4-tuple ``SkillsGraphEnv.step``.
Canonical (``clip_eps=0.2``, ``gae_lambda=0.95``, ``gamma=0.99``) flow in
from ``config.ppo`` and are asserted at ``__init__`` so drift fails fast."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.distributions import Categorical

from src.env.action_mask import compute_mask
from src.env.actions import global_index_to_action
from src.env.skills_graph_env import SkillsGraphEnv
from src.model.policy_net import PolicyNet
from src.services._ppo_helpers import pad_state
from src.services.gae_buffer import Trajectory, compute_gae_advantages, compute_returns

_CLIP, _LAM, _GAMMA = 0.2, 0.95, 0.99


@dataclass
class PPOConfig:
    """Hyperparameter bundle (keeps ``PPOTrainer.__init__`` argument count sane)."""

    clip_eps: float = _CLIP
    gae_lambda: float = _LAM
    gamma: float = _GAMMA
    lr: float = 3e-4
    n_steps: int = 128
    n_epochs: int = 4
    batch_size: int = 64
    vf_coef: float = 0.5


class PPOTrainer:
    """PPO with clipped surrogate + GAE — wraps our 4-tuple env directly."""

    def __init__(self, env: SkillsGraphEnv, policy: PolicyNet, **kwargs) -> None:
        cfg = PPOConfig(**kwargs)
        if float(cfg.clip_eps) != _CLIP or float(cfg.gae_lambda) != _LAM:
            raise ValueError(f"clip_eps/gae_lambda sealed at {_CLIP}/{_LAM}; got {cfg}")
        self.env, self.policy, self.cfg = env, policy, cfg
        self.clip_eps, self.gae_lambda, self.gamma = cfg.clip_eps, cfg.gae_lambda, cfg.gamma
        self.n_steps, self.n_epochs, self.batch_size = cfg.n_steps, cfg.n_epochs, cfg.batch_size
        self.vf_coef = cfg.vf_coef
        self.optimizer = optim.Adam(self.policy.parameters(), lr=float(cfg.lr))

    def _fwd(
        self, state, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward the policy over ``state`` with an explicit ``action_mask``.
        ``action_mask`` defaults to the live env mask (rollout path). During
        ``update()`` we replay historical states; the env's current mask
        does not match those states, so callers pass the per-state mask
        computed from ``compute_mask(state)`` to avoid -inf log-prob spikes.
        """
        x, m = pad_state(state)
        logits, value = self.policy(x, m)
        if action_mask is None:
            action_mask = self.env.get_action_mask()
        return logits, value, action_mask.unsqueeze(0)

    def collect_rollout(self) -> Trajectory:
        """Run ``self.n_steps`` env steps, building a Trajectory."""
        s_, a_, lp_, r_, v_, d_ = [], [], [], [], [], []
        state, _ = self.env.reset()
        for _ in range(self.n_steps):
            with torch.no_grad():
                logits, value, amask = self._fwd(state)
                aidx, log_prob = self.policy.get_action(logits, amask)
            idx = int(aidx.item())
            nxt, reward, done, _ = self.env.step(global_index_to_action(idx))
            s_.append(state)
            a_.append(idx)
            lp_.append(float(log_prob.item()))
            r_.append(float(reward))
            v_.append(float(value.squeeze().item()))
            d_.append(bool(done))
            state = self.env.reset()[0] if done else nxt
        f32 = torch.float32
        return Trajectory(
            states=s_,
            actions=a_,
            log_probs=torch.tensor(lp_, dtype=f32),
            rewards=torch.tensor(r_, dtype=f32),
            values=torch.tensor(v_, dtype=f32),
            dones=torch.tensor(d_, dtype=torch.bool),
        )

    def _eval(self, traj: Trajectory, idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        lps, vs = [], []
        for i in idxs:
            # Per-historical-state mask: env state may have advanced past traj.states[i].
            amask_state = compute_mask(traj.states[i])
            logits, value, amask = self._fwd(traj.states[i], action_mask=amask_state)
            masked = logits.masked_fill(~amask, float("-inf"))
            lps.append(Categorical(logits=masked).log_prob(torch.tensor([traj.actions[i]])))
            vs.append(value.squeeze(0).squeeze(-1))
        return torch.cat(lps), torch.stack(vs)

    def compute_loss(
        self,
        trajectory: Trajectory,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        idxs: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PPO clipped surrogate + MSE value loss over indices ``idxs``."""
        if idxs is None:
            idxs = list(range(len(trajectory.states)))
        new_lps, new_vs = self._eval(trajectory, idxs)
        adv, ret = advantages[idxs], returns[idxs]
        ratio = torch.exp(new_lps - trajectory.log_probs[idxs])
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        return -torch.min(ratio * adv, clipped * adv).mean(), nn.functional.mse_loss(new_vs, ret)

    def update(self, trajectory: Trajectory) -> dict[str, float]:
        """Run ``n_epochs`` of mini-batch SGD; return PPO metrics."""
        adv = compute_gae_advantages(trajectory, gamma=self.gamma, gae_lambda=self.gae_lambda, last_value=0.0)
        ret = compute_returns(adv, trajectory.values)
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        t = len(trajectory.states)
        idxs = list(range(t))
        lp = lv = cf = kl = 0.0
        for _ in range(self.n_epochs):
            for s in range(0, t, self.batch_size):
                batch = idxs[s : s + self.batch_size]
                p_loss, v_loss = self.compute_loss(trajectory, adv, ret, batch)
                self.optimizer.zero_grad()
                (p_loss + self.vf_coef * v_loss).backward()
                self.optimizer.step()
                lp, lv = float(p_loss.item()), float(v_loss.item())
                with torch.no_grad():
                    diff = self._eval(trajectory, batch)[0] - trajectory.log_probs[batch]
                    cf = float(((torch.exp(diff) - 1.0).abs() > self.clip_eps).float().mean().item())
                    kl = float((-diff).mean().item())
        return {"policy_loss": lp, "value_loss": lv, "clip_fraction": cf, "approx_kl": kl}

    def train(self, total_steps: int, *, log_every: int = 1000) -> list[dict]:
        """Outer loop: collect → update; return per-iter metrics. ``log_every`` is logger-agnostic — caller subscribes to ``history``."""
        _ = log_every
        history, steps = [], 0
        while steps < total_steps:
            steps += self.n_steps
            history.append({**self.update(self.collect_rollout()), "steps": steps})
        return history
