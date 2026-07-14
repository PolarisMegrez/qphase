"""Shared CuPy helpers for model-local kernels."""

from __future__ import annotations

from typing import Any


def broadcast_param(value: Any, n: int) -> Any:
    """Return a float64 CuPy parameter array with shape ``(n,)``."""
    import cupy as cp

    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        arr = cp.asarray(value)
        if arr.shape == (n,):
            return arr.astype(cp.float64, copy=False)
        if arr.size == 1:
            return cp.full((n,), float(arr.item()), dtype=cp.float64)
    return cp.full((n,), float(value), dtype=cp.float64)
