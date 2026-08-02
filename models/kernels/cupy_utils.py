"""Shared CuPy helpers for model-local kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

_BufferT = TypeVar("_BufferT")


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


def get_lru_buffer(
    cache: dict[Any, _BufferT],
    keys: list[Any],
    key: Any,
    factory: Callable[[], _BufferT],
    *,
    max_entries: int = 2,
) -> _BufferT:
    """Return a cached device buffer while bounding retained shape variants."""
    if key in cache:
        keys.remove(key)
        keys.append(key)
        return cache[key]
    value = factory()
    cache[key] = value
    keys.append(key)
    while len(keys) > max_entries:
        del cache[keys.pop(0)]
    return value
