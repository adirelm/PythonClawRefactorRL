"""Dual-criterion convergence checker (ADR-010).

Stub for Phase 1+ implementation. Real impl will:
- Compute non-overlapping window means (A=[t-200:t-100], B=[t-100:t], step=100)
- Check reward criterion: |mean(B)-mean(A)|/|mean(A)| ≤ reward_rel_tol (default 0.02)
- Check entropy slope: |dH/dt| < entropy_slope_threshold (default 0.01 nats/ep)
- Emit ConvergenceVerdict per seed; PARTIAL_CONVERGENCE supported
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvergenceVerdict:
    converged: bool
    reward_delta: float
    entropy_slope: float
    partial: bool
    reason: str


PARTIAL_CONVERGENCE = "PARTIAL_CONVERGENCE"  # marker


def check_convergence(rewards: list[float], entropies: list[float], *, config: dict) -> ConvergenceVerdict:
    """Phase 1+ will implement. Stub raises NotImplementedError."""
    raise NotImplementedError("Phase 1+ implementation pending; see ADR-010")
