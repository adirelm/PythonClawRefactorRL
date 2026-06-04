"""Architectural contract: GraphifyAdapter Protocol shape (ADR-002).

Asserts:
1. `GraphifyAdapter` is a typing.Protocol (not a regular class).
2. It exposes `build()` and `load()` callables.
3. `LocalGraphify` structurally satisfies the Protocol via
   `runtime_checkable` isinstance check.

Keeps the F2 acceptance pointer real: any future Phase-1 implementation
that drifts from this contract will fail this test before merging.
"""

from __future__ import annotations

from typing import Protocol, get_type_hints  # noqa: F401

from src.graphify import GraphifyAdapter
from src.graphify.local_impl import LocalGraphify


def test_graphify_adapter_is_protocol() -> None:
    """GraphifyAdapter must be a typing.Protocol subclass marker."""
    # Protocol classes carry the _is_protocol sentinel set by typing machinery.
    assert getattr(GraphifyAdapter, "_is_protocol", False), (
        "GraphifyAdapter must be declared as `class GraphifyAdapter(Protocol)`"
    )


def test_graphify_adapter_has_build_method() -> None:
    """Contract: .build(src_root, *, seed) must be defined."""
    assert hasattr(GraphifyAdapter, "build"), "GraphifyAdapter must declare .build()"
    assert callable(GraphifyAdapter.build), ".build must be callable"


def test_graphify_adapter_has_load_method() -> None:
    """Contract: .load(pickle_path) must be defined."""
    assert hasattr(GraphifyAdapter, "load"), "GraphifyAdapter must declare .load()"
    assert callable(GraphifyAdapter.load), ".load must be callable"


def test_local_graphify_satisfies_protocol() -> None:
    """LocalGraphify must structurally satisfy GraphifyAdapter (runtime_checkable)."""
    instance = LocalGraphify()
    assert isinstance(instance, GraphifyAdapter), (
        "LocalGraphify must satisfy GraphifyAdapter Protocol (check method names/signatures match adapter.py)"
    )
