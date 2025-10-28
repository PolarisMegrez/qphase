from __future__ import annotations

"""Backend factory: instantiate a backend by name/config via the unified registry."""

from typing import Any

from ..core.registry import registry


def get_backend(name: str, **kwargs: Any):
    full = name if ":" in name else f"backend:{name}"
    return registry.create(full, **kwargs)
