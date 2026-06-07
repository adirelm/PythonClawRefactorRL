# THEORY.md — PPO + GAE + Graph-RL Mathematical Foundations

> Cross-references each equation to its source paper, the config key that
> seals its hyperparameter, and the `src/` module that implements it.

---

## 1  PPO Clipped Surrogate Objective (Schulman et al. 2017, arXiv:1707.06347)

### 1.1  Probability ratio

Let $\theta$ be the current policy parameters and $\theta_{\text{old}}$ the
parameters used during rollout collection. The importance-sampling ratio for
timestep $t$ is:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

In log-space (numerically stable):
$r_t = \exp\!\bigl(\log\pi_\theta - \log\pi_{\theta_{\text{old}}}\bigr)$

Implementation: `src/services/ppo_trainer.py` `compute_loss()`, line
`ratio = torch.exp(new_lps - trajectory.log_probs[idxs])`.

### 1.2  Clipped surrogate loss (Eq. 7 of Schulman 2017)

$$L^{\text{CLIP}}(\theta)
  = \hat{\mathbb{E}}_t\!\Bigl[
      \min\!\bigl(r_t(\theta)\,\hat{A}_t,\;
                  \operatorname{clip}(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon)\,\hat{A}_t\bigr)
    \Bigr]$$

**Sealed hyperparameter:** $\varepsilon = 0.2$ (`config.ppo.clip_eps`; asserted
at `PPOTrainer.__init__`; architecture test `tests/architecture/test_reward_formula.py`
checks the sealed value).

Implementation: `src/services/ppo_trainer.py`
```
clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
return -torch.min(ratio * adv, clipped * adv).mean(), ...
```

### 1.3  Full PPO objective with value loss and entropy bonus

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H[\pi_\theta]$$

where $c_1 = 0.5$ (`config.ppo.vf_coef`) and $H$ is the policy entropy. In
this project the entropy bonus is implicit (the Categorical sampler preserves
entropy; no explicit $c_2$ term is added — see ADR-008). The update step:

```
(p_loss + self.vf_coef * v_loss).backward()
```

**Cross-ref:** `src/services/ppo_trainer.py` `update()`; PRD `docs/prd/PRD-PPO.md`.

---

## 2  Generalised Advantage Estimation — GAE(λ) (Schulman et al. 2016, arXiv:1506.02438)

### 2.1  TD residual with terminal mask (Eq. 11)

$$\delta_t = r_t + \gamma\,(1 - d_t)\,V(s_{t+1}) - V(s_t)$$

where $d_t \in \{0,1\}$ is the done flag (terminal mask). The $(1-d_t)$ factor
zeroes out the bootstrap value on episode boundaries so that $V(s_{t+1})$ is
not added across episode resets.

**Sealed hyperparameters:** $\gamma = 0.99$ (`config.ppo.gamma`).

### 2.2  GAE recurrence (Eq. 16) — backward pass

$$\hat{A}_t = \delta_t + (\gamma\lambda)(1 - d_t)\,\hat{A}_{t+1}$$

with boundary condition $\hat{A}_T = 0$. Walking from $t = T-1$ down to $t=0$:

$$\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta_{t+l}
              \quad(\text{truncated at episode termination})$$

**Sealed hyperparameter:** $\lambda = 0.95$ (`config.ppo.gae_lambda`; asserted
at `PPOTrainer.__init__`; test `tests/unit/services/test_gae_buffer.py::test_gae_uses_canonical_lambda_0_95`).

Implementation: `src/services/gae_buffer.py` `compute_gae_advantages()`:
```python
delta = r_t + gamma * mask * next_value - V_t
next_advantage = delta + gamma * gae_lambda * mask * next_advantage
```

### 2.3  Advantage normalisation

Before the policy update, advantages are normalised per mini-batch:

$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + 10^{-8}}$$

This is standard PPO practice (Engstrom et al. 2020) to reduce gradient
variance without changing the optimal policy.

Implementation: `src/services/ppo_trainer.py` `update()`:
```python
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

---

## 3  Canonical Reward Function (ADR-007, brief §2.2)

$$R_t = \alpha\,\Delta\text{Modularity}_t
      + \beta\,\Delta\text{Cohesion}_t
      - \gamma\,\text{CouplingPenalty}_t
      + P_{\text{skills},t}$$

**Sealed defaults** (`config.reward`):

| Symbol | Default | Config key | Meaning |
|--------|---------|-----------|---------|
| $\alpha$ | 1.0 | `reward.alpha` | ΔModularity weight |
| $\beta$ | 1.0 | `reward.beta` | ΔCohesion weight |
| $\gamma$ | 0.5 | `reward.gamma` | Coupling-penalty weight |
| $P_{\text{skills}}$ | −5.0 | `reward.p_skills` | Lazy-load-break penalty |

Architecture test `tests/architecture/test_reward_formula.py` parses the AST
of `src/env/reward.py` and asserts all four terms and signs are present.

**Newman-Girvan modularity** (Newman & Girvan 2004):

$$Q = \frac{1}{2m}\sum_{ij}\Bigl[A_{ij} - \frac{k_i k_j}{2m}\Bigr]\delta(c_i, c_j)$$

where $m$ = number of edges, $k_i$ = degree of node $i$, $c_i$ = community
assignment, $\delta$ = Kronecker delta. Computed via Louvain
(`networkx.algorithms.community.louvain_communities`) with a 0.5 s watchdog
and greedy fallback (`greedy_modularity_communities`).

Implementation: `src/services/metrics/modularity.py`; `src/env/reward.py`.

---

## 4  Action Masking (Huang & Ontañón 2022)

Pre-softmax logit masking prevents the policy from sampling illegal actions
without biasing the gradient estimate (Huang & Ontañón 2022, "A Closer Look at
Invalid Action Masking"):

$$\tilde{\ell}_a = \begin{cases} \ell_a & \text{if } a \in \mathcal{A}_{\text{legal}} \\ -\infty & \text{otherwise} \end{cases}$$

The action distribution is then $\pi_\theta(a \mid s) = \text{softmax}(\tilde{\ell})$.

Implementation: `src/env/action_mask.py` `compute_mask()` + `src/model/policy_net.py`
`get_action()`. NOOP (action index 45056) is always legal (ACTION_DESIGN §2.4).

---

## 5  Equation–Module Cross-Reference

| Equation | Paper | `src/` module | Config key |
|----------|-------|---------------|-----------|
| $r_t(\theta)$ — ratio | Schulman 2017 | `ppo_trainer.compute_loss` | — |
| $L^{\text{CLIP}}$ — clipped surrogate | Schulman 2017, Eq. 7 | `ppo_trainer.compute_loss` | `ppo.clip_eps=0.2` |
| $\delta_t$ — TD residual | Schulman 2016, Eq. 11 | `gae_buffer.compute_gae_advantages` | `ppo.gamma=0.99` |
| $\hat{A}_t$ — GAE advantage | Schulman 2016, Eq. 16 | `gae_buffer.compute_gae_advantages` | `ppo.gae_lambda=0.95` |
| $R_t$ — reward | ADR-007 / brief §2.2 | `env.reward.compute_reward` | `reward.{alpha,beta,gamma,p_skills}` |
| $Q$ — Newman-Girvan modularity | Newman & Girvan 2004 | `services.metrics.modularity` | — |
| $\tilde{\ell}_a$ — logit masking | Huang & Ontañón 2022 | `env.action_mask.compute_mask` | — |

---

## 6  References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
  *High-Dimensional Continuous Control Using Generalized Advantage Estimation*.
  arXiv:1506.02438.
- Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community
  structure in networks. *Physical Review E*, 69(2), 026113.
- Huang, S., & Ontañón, S. (2022). A Closer Look at Invalid Action Masking in
  Policy Gradient Algorithms. *FLAIRS-35*. arXiv:2006.14171.
- Engstrom, L., et al. (2020). Implementation Matters in Deep RL: A Case Study
  on PPO and TRPO. *ICLR 2020*.
