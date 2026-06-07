# PPO + GAE Math Audit (Schulman 2017/2016)

**Auditor:** Claude-default rigor (Codex CLI unavailable in environment).
**Scope:** `src/services/ppo_trainer.py`, `src/services/gae_buffer.py`, `src/model/policy_net.py`.
**Method:** Verbatim code quotation against Schulman 2017 (PPO), Schulman 2016 (GAE),
Huang & Ontañón 2022 (action masking), Engstrom 2020 (implementation matters).

---

## 1. PPO Clipped Surrogate (Schulman et al. 2017, Eq. 7)

**Equation 7:**
> L^CLIP(θ) = Ê_t[ min( r_t(θ) Â_t, clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
> where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log π_new − log π_old)

**Code (ppo_trainer.py:128–130, `compute_loss`):**
```python
ratio = torch.exp(new_lps - trajectory.log_probs[idxs])
clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
return -torch.min(ratio * adv, clipped * adv).mean(), nn.functional.mse_loss(new_vs, ret)
```

**Clip-eps seal (ppo_trainer.py:54–55):**
```python
if float(cfg.clip_eps) != _CLIP or float(cfg.gae_lambda) != _LAM:
    raise ValueError(f"clip_eps/gae_lambda sealed at {_CLIP}/{_LAM}; got {cfg}")
```
where `_CLIP, _LAM, _GAMMA = 0.2, 0.95, 0.99` (line 21).

**KL + clip-fraction logging (ppo_trainer.py:150–152):**
```python
diff = self._eval(trajectory, batch)[0] - trajectory.log_probs[batch]
cf = float(((torch.exp(diff) - 1.0).abs() > self.clip_eps).float().mean().item())
kl = float((-diff).mean().item())
```

| Check | Result |
|---|---|
| `ratio = exp(new_log_prob − old_log_prob)` | PASS |
| `clip(ratio, 1−ε, 1+ε)` with ε=0.2 (sealed) | PASS |
| `min(surr1, surr2)` before negation | PASS |
| `policy_loss = -mean(min(...))` | PASS |
| `approx_kl` + `clip_fraction` logged | PASS |
| Value clipping (Engstrom 2020) | **FAIL** — plain MSE only |

**Section verdict: WARN.** Core surrogate matches Eq. 7 exactly. Value clipping
(`v_clipped = v_old + clip(v_new − v_old, −ε, +ε)`) is absent; Engstrom 2020
flags this as a common implementation detail, not a Schulman 2017 requirement.
Acceptable for assignment scope; flag in `BUG_REPORT.md` follow-ups.

---

## 2. Generalized Advantage Estimator (Schulman et al. 2016, Eq. 11 + 16)

**Eq. 11 (TD residual):** δ_t^V = r_t + γV(s_{t+1}) − V(s_t)
**Eq. 16 (GAE recursion):** Â_t = δ_t + γλ Â_{t+1}
**Terminal mask:** Â_t = δ_t + γλ(1−done_t) Â_{t+1}; boundary Â_T = 0.

**Code (gae_buffer.py:74–82, `compute_gae_advantages`):**
```python
advantages = torch.zeros_like(rewards)
next_value = float(last_value)
next_advantage = 0.0
for step in range(t - 1, -1, -1):
    mask = 1.0 - float(dones[step].item())
    delta = float(rewards[step].item()) + gamma * mask * next_value - float(values[step].item())
    next_advantage = delta + gamma * gae_lambda * mask * next_advantage
    advantages[step] = next_advantage
    next_value = float(values[step].item())
```

**Returns (gae_buffer.py:90):**
```python
return advantages + values
```

**Trainer wiring (ppo_trainer.py:134):**
```python
adv = compute_gae_advantages(trajectory, gamma=self.gamma, gae_lambda=self.gae_lambda, last_value=0.0)
```

| Check | Result |
|---|---|
| δ_t formula with `(1−done_t)` terminal mask | PASS |
| Recursion walks BACKWARDS T-1 → 0 (`range(t-1, -1, -1)`) | PASS |
| Boundary `Â_T` via `next_advantage = 0.0` init + `next_value = last_value` | PASS |
| `returns = advantages + values` | PASS |
| λ=0.95, γ=0.99 sealed (line 21 + init check) | PASS |

**Note on boundary.** Spec asked: `Â_T = last_value · γ · (1−done_T)`. Code does
something equivalent but cleaner: it initializes `next_advantage = 0.0` (Â_{T+1} = 0)
and `next_value = last_value` (V(s_T) = bootstrap), so the first iteration computes
`δ_{T-1} = r_{T-1} + γ(1−done_{T-1})·last_value − V(s_{T-1})` and
`Â_{T-1} = δ_{T-1} + γλ(1−done_{T-1})·0 = δ_{T-1}`. Matches Schulman 2016 Eq. 16
under the standard "last advantage is zero" boundary convention. PASS.

**Section verdict: PASS.**

---

## 3. Action Masking (Huang & Ontañón 2022, §3)

**Pattern:** logits[~mask] := −∞ **before** Categorical(logits=...) sampling.

**Code (policy_net.py:113–121, `get_action`):**
```python
safe_mask = action_mask
empty_rows = (~action_mask).all(dim=-1)
if bool(empty_rows.any()):
    safe_mask = action_mask.clone()
    safe_mask[..., _NOOP_IDX_DEFAULT] = safe_mask[..., _NOOP_IDX_DEFAULT] | empty_rows
masked = logits.masked_fill(~safe_mask, float("-inf"))
dist = Categorical(logits=masked)
action_idx = dist.sample()
log_prob = dist.log_prob(action_idx)
```

**Update-path mask replay (ppo_trainer.py:109–112):**
```python
amask_state = compute_mask(traj.states[i])
logits, value, amask = self._fwd(traj.states[i], action_mask=amask_state)
masked = logits.masked_fill(~amask, float("-inf"))
lps.append(Categorical(logits=masked).log_prob(torch.tensor([traj.actions[i]])))
```

| Check | Result |
|---|---|
| `masked_fill` applied BEFORE `Categorical` | PASS |
| `Categorical(logits=...)` (logsumexp-safe, not probs) | PASS |
| `log_prob` returned for ratio computation | PASS |
| All-False-row defensive NOOP fallback (avoids `Categorical(-inf)` NaN hang on seeds 123/314/271) | PASS (extra safety) |

**Section verdict: PASS.** The all-False-row NOOP fallback is *extra-spec*: it
preserves Huang & Ontañón §3 semantics (NOOP is the always-legal escape per
`ACTION_DESIGN §2.4`) while plugging the Categorical(-inf) hang documented in
TRACE.md F10 / TODO.md T3.7.

---

## 4. Verdict per Section

| Section | Verdict | Reason |
|---|---|---|
| 1. PPO clipped surrogate | **WARN** | Core Eq. 7 exact; value clipping (Engstrom 2020) absent |
| 2. GAE | **PASS** | Eq. 11 + 16 with terminal mask, sealed hyperparams |
| 3. Action masking | **PASS** | Huang & Ontañón §3 + defensive NOOP for degenerate graphs |

**Overall: PASS-with-WARN.** The PPO + GAE math is correct against Schulman
2017/2016. The single WARN (no value clipping) is an implementation detail,
not a Schulman 2017 requirement, and does not affect the canonical sealed
values (ε=0.2, λ=0.95, γ=0.99).

---

## 5. Found Issues

1. **Value clipping absent (WARN, §1).**
   Engstrom et al. 2020 §4.2 recommends `v_clipped = v_old + clip(v_new − v_old, −ε, +ε)`
   and `value_loss = max(MSE(v_new, ret), MSE(v_clipped, ret))`. Current code
   (`ppo_trainer.py:130`) uses plain `nn.functional.mse_loss(new_vs, ret)`. Not a
   Schulman 2017 PPO requirement; flag for `BUG_REPORT.md` follow-ups.

2. **No advantage normalization across full batch (INFO, §1).**
   `ppo_trainer.py:137` normalizes `adv` *once* over the full trajectory before
   mini-batching: `adv = (adv - adv.mean()) / (adv.std() + 1e-8)`. Some
   implementations renormalize per-minibatch (Engstrom 2020 §4.3). Either is
   defensible; current choice is cheaper and stable. No action.

3. **`next_value` reassigned from in-buffer V (INFO, §2).**
   `gae_buffer.py:82`: after computing Â_t, `next_value = values[step]` (the
   in-buffer V(s_t) used as V(s_{t+1}) for step-1). This is the standard GAE
   bootstrap chain and matches Schulman 2016. No action.

---

## 6. References

- Schulman et al. 2017, "Proximal Policy Optimization Algorithms," arXiv:1707.06347 (Eq. 7).
- Schulman et al. 2016, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438 (Eq. 11, 16).
- Huang & Ontañón 2022, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms," arXiv:2006.14171 (§3).
- Engstrom et al. 2020, "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO," arXiv:2005.12729 (§4.2, §4.3).
