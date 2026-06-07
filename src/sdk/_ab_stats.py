"""t-CI helpers for SDK ablation (split from ``ablation.py`` to keep ≤150 LOC).

Mirrors ``scripts/_ablation_lib._T_CRIT_975`` so the SDK consumer path
stays import-cheap (no scipy).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import mean as _mean
from statistics import stdev as _stdev

# Student-t 97.5% critical values for dof = 1..30; >=30 falls back to 1.96.
# fmt: off
_T_CRIT_975: dict[int, float] = dict(enumerate([
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
    2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
], start=1))
# fmt: on
_T_INF = 1.9600
_MIN_N_FOR_CI = 2


def t_ci95(values: Iterable[float]) -> tuple[float, float]:
    """(mean, t95 half-width) on finite values; (nan, 0.0) if empty, (m, 0.0) if n<2."""
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    n = len(finite)
    if n == 0:
        return (float("nan"), 0.0)
    m = _mean(finite)
    if n < _MIN_N_FOR_CI:
        return (m, 0.0)
    s = _stdev(finite)
    half = _T_CRIT_975.get(n - 1, _T_INF) * s / math.sqrt(n)
    return (m, half)
