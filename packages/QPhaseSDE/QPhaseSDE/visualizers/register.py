from __future__ import annotations

"""Domain-level registration helpers for visualizers.

Thin wrappers around the central registry to improve readability.
"""

from typing import Any, Callable

from ..core.registry import registry as _registry


def register(name: str, builder: Callable[..., Any], **meta: Any) -> None:
    _registry.register("visualization", name, builder, **meta)


def register_lazy(name: str, target: str, **meta: Any) -> None:
    _registry.register_lazy("visualization", name, target, **meta)
