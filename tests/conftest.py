"""Shared pytest fixtures — populated in Phase 1+.

Kept minimal at bootstrap so the empty test tree still collects cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent
