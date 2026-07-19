"""Shared CuPy helpers for model-local kernels."""

from __future__ import annotations

from typing import Any


def broadcast_param(value: Any, n: int, dtype: Any = None) -> Any:
    """Return a CuPy parameter array with shape ``(n,)``."""
    import cupy as cp

    dtype = cp.float64 if dtype is None else dtype
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        arr = cp.asarray(value)
        if arr.shape == (n,):
            return arr.astype(dtype, copy=False)
        if arr.size == 1:
            return cp.full((n,), float(arr.item()), dtype=dtype)
    return cp.full((n,), float(value), dtype=dtype)
