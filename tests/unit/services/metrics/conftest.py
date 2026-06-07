"""Autouse fixture: clear the modularity partition memo between tests.

The Phase 4 RC-1 ``safe_louvain`` cache (``src.services.metrics.modularity.
_partition_cache``) memoizes by structural graph key so NOOPs and failed
refactor ops within a rollout don't re-run Louvain. Without a between-tests
reset the watchdog tests would see stale partitions from earlier monkey-
patched runs and false-positive on the wedge path.
"""

from __future__ import annotations

import pytest

from src.services.metrics.modularity import clear_partition_cache


@pytest.fixture(autouse=True)
def _clear_louvain_cache():
    clear_partition_cache()
    yield
    clear_partition_cache()
