"""Metrics services — closes the Codex gap so ``src.env.reward`` resolves.

Re-exports the primitive metrics + delta helpers consumed by
``src.env.reward.compute_reward``. Cohesion / coupling land in sibling
commits this same workflow; ``__getattr__`` defers their imports so
this commit alone unblocks modularity. ``compute_coupling`` is exposed
as an alias of ``compute_coupling_penalty`` to match the late import
in ``src.env.reward`` unchanged.
"""

from __future__ import annotations

from src.services.metrics.modularity import compute_modularity, delta_modularity

__all__ = [
    "compute_cohesion",
    "compute_coupling",
    "compute_coupling_penalty",
    "compute_modularity",
    "delta_cohesion",
    "delta_coupling",
    "delta_modularity",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy dispatch
    if name in {"compute_cohesion", "delta_cohesion"}:
        from src.services.metrics import cohesion  # noqa: PLC0415

        return getattr(cohesion, name)
    if name in {"compute_coupling_penalty", "delta_coupling", "compute_coupling"}:
        from src.services.metrics import coupling  # noqa: PLC0415

        attr = "compute_coupling_penalty" if name == "compute_coupling" else name
        return getattr(coupling, attr)
    raise AttributeError(f"module 'src.services.metrics' has no attribute {name!r}")
