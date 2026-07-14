"""qphase_sde: Buffer cache for SDE simulation.
---------------------------------------------------------
Lightweight, backend-agnostic buffer reuse inside the SDE engine. Keeping a
small pool of pre-allocated arrays reduces the number of ``empty``/``zeros``
calls per integration step, which is especially beneficial for GPU backends
where frequent CUDA allocations can become a bottleneck.

Public API
----------
``SDEBufferCache`` : Simple LRU-style cache keyed by ``(shape, dtype)``.
"""

from typing import Any

from qphase.backend.base import BackendBase

__all__ = ["SDEBufferCache"]


class SDEBufferCache:
    """Cache for reusing temporary arrays during SDE integration.

    The cache is keyed by ``(shape, dtype)``. Buffers returned by ``get`` are
    not zeroed; callers must overwrite the entire array before use.

    Parameters
    ----------
    backend : BackendBase
        Backend used to allocate new buffers when the cache misses.
    max_entries_per_key : int
        Maximum number of buffers kept for each ``(shape, dtype)`` key.
    """

    def __init__(
        self, backend: BackendBase, max_entries_per_key: int = 2
    ) -> None:
        self.backend = backend
        self.max_entries_per_key = max(max_entries_per_key, 1)
        self._cache: dict[tuple[tuple[int, ...], str], list[Any]] = {}

    def _key(self, shape: tuple[int, ...], dtype: Any) -> tuple[tuple[int, ...], str]:
        dtype_key = str(dtype)
        return (shape, dtype_key)

    def get(self, shape: tuple[int, ...], dtype: Any) -> Any:
        """Return a buffer of the requested shape and dtype.

        If a matching buffer is in the cache, it is returned. Otherwise a new
        buffer is allocated with ``backend.empty``.
        """
        key = self._key(shape, dtype)
        pool = self._cache.get(key)
        if pool:
            return pool.pop()
        return self.backend.empty(shape, dtype=dtype)

    def put(self, buf: Any) -> None:
        """Return a buffer to the cache for reuse.

        The buffer is only retained if the pool for its ``(shape, dtype)`` key
        has not reached ``max_entries_per_key``.
        """
        if buf is None:
            return
        shape = getattr(buf, "shape", None)
        dtype = getattr(buf, "dtype", None)
        if shape is None or dtype is None:
            return
        key = self._key(shape, dtype)
        pool = self._cache.setdefault(key, [])
        if len(pool) < self.max_entries_per_key:
            pool.append(buf)

    def clear(self) -> None:
        """Drop all cached buffers."""
        self._cache.clear()
