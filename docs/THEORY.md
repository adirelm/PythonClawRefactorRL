# THEORY.md — A4 RL theory write-up (Phase 1 fill pending)

This document captures the brief §2.3 PPO + GAE math, the reward shaping
theory (Ng et al. 1999), and the graph-RL methodology. It will be filled
out during Phase 1+. For now, equation pointers:

## PPO clipped objective (Schulman 2017 Eq.7)
See docs/prd/PRD-PPO.md §3 for L^CLIP, hyperparameter table, and SB3 wrapper plan.

## GAE advantage estimation (Schulman 2016 Eq.11+16)
See docs/prd/PRD-GAE.md §2.1 for δ_t (terminal-mask) and Â_t recurrence.

## Reward shaping (Ng et al. 1999)
See docs/adr/ADR-007-reward-upgrade-MUST.md for weighted-sum vs potential-based discussion + canonical reward formula.

## Action masking (Huang & Ontañón 2022)
See docs/ACTION_DESIGN.md §5 for pre-softmax logit→−∞ masking.
