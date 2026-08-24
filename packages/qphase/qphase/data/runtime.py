"""qphase: Runtime Data Handle Implementations
---------------------------------------------------------
Minimal in-process handle/lease implementations backing the Phase 0 frozen
protocols (:mod:`qphase.data.handles`). Phase 1 provides single-process
owner/lease bookkeeping and explicit ``materialize`` only — no LRU, eviction,
pinning or cross-job caching.

- :class:`HostArrayHandle` wraps a NumPy array on the host.
- :class:`BackendArrayHandle` wraps a NumPy-API-compatible device array
  (e.g. CuPy); explicit device-to-host copies go through
  :func:`qphase.backend.xputil.convert_to_numpy` and only with
  ``copy_policy="allow"``.
- :class:`ReadOnlyArrayView` is a delegating read-only view of another
  handle.
- :class:`DataLease` is the default idempotent lease (execution/session
  scope).

Ownership rules follow the frozen contract: only the owner closes a handle;
closing invalidates outstanding leases, whose consumers then observe an error
on next access rather than silently reading freed memory.

Public API
----------
HostArrayHandle
    Owner-managed host array handle.
BackendArrayHandle
    Owner-managed device array handle (NumPy-API-compatible backends).
ReadOnlyArrayView
    Read-only delegating view of another handle.
DataLease
    Default idempotent lease implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend.xputil import convert_to_numpy
from .handles import CopyPolicy, LeaseScope
from .schema import VariableSchema

__all__ = [
    "BackendArrayHandle",
    "DataLease",
    "HostArrayHandle",
    "ReadOnlyArrayView",
]


class DataLease:
    """Default lease over a handle; ``release()`` is idempotent."""

    def __init__(
        self,
        handle: Any,
        consumer: str,
        scope: LeaseScope = "execution",
    ) -> None:
        self._handle = handle
        self._consumer = consumer
        self._scope = scope
        self._released = False

    @property
    def handle(self) -> Any:
        """The leased handle."""
        return self._handle

    @property
    def consumer(self) -> str:
        """Identifier of the consuming component."""
        return self._consumer

    @property
    def scope(self) -> LeaseScope:
        """Lifetime scope of the lease."""
        return self._scope

    def release(self) -> None:
        """Release the lease; idempotent."""
        if not self._released:
            self._released = True
            self._handle._release_lease(self)

    def __enter__(self) -> DataLease:
        """Return the lease itself on context entry."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the lease on context exit."""
        self.release()


class _ArrayHandleBase:
    """Shared owner/lease bookkeeping of the concrete handles."""

    def __init__(
        self,
        variable_schema: VariableSchema,
        *,
        owner: str,
        read_only: bool,
    ) -> None:
        self._variable_schema = variable_schema
        self._owner = owner
        self._read_only = read_only
        self._leases: set[DataLease] = set()
        self._closed = False

    # -- protocol surface ------------------------------------------------

    @property
    def variable_schema(self) -> VariableSchema:
        """Schema of the single variable this handle backs."""
        return self._variable_schema

    @property
    def read_only(self) -> bool:
        """Whether consumers may mutate the underlying buffer."""
        return self._read_only

    @property
    def owner(self) -> str:
        """Identifier of the component owning the buffer's lifetime."""
        return self._owner

    def acquire(
        self, consumer: str, scope: LeaseScope = "execution"
    ) -> DataLease:
        """Acquire a lease on this handle for one consumer."""
        self._check_live()
        lease = DataLease(self, consumer, scope)
        self._leases.add(lease)
        return lease

    # -- owner operations -------------------------------------------------

    @property
    def lease_count(self) -> int:
        """Number of outstanding leases (diagnostics/tests)."""
        return len(self._leases)

    @property
    def closed(self) -> bool:
        """Whether the owner has closed this handle."""
        return self._closed

    def close(self) -> None:
        """Close the buffer; owner-only. Outstanding leases are invalidated.

        Consumers holding a lease observe an error on next access through the
        handle. ``close()`` itself is idempotent.
        """
        self._closed = True

    # -- internals ----------------------------------------------------------

    def _release_lease(self, lease: DataLease) -> None:
        self._leases.discard(lease)

    def _check_live(self) -> None:
        if self._closed:
            raise RuntimeError(
                f"handle for variable {self._variable_schema.name!r} owned by "
                f"{self._owner!r} has been closed"
            )

    def _check_variable(self, array: Any) -> None:
        dtype = np.dtype(array.dtype)
        if dtype != np.dtype(self._variable_schema.dtype):
            raise ValueError(
                f"array dtype {dtype} does not match variable "
                f"{self._variable_schema.name!r} dtype "
                f"{self._variable_schema.dtype}"
            )
        if len(array.shape) != len(self._variable_schema.dims):
            raise ValueError(
                f"array shape {tuple(array.shape)} does not match variable "
                f"{self._variable_schema.name!r} dims "
                f"{self._variable_schema.dims}"
            )


class HostArrayHandle(_ArrayHandleBase):
    """Owner-managed handle over a host (NumPy) array."""

    def __init__(
        self,
        array: np.ndarray,
        variable_schema: VariableSchema,
        *,
        owner: str,
        read_only: bool = False,
    ) -> None:
        super().__init__(variable_schema, owner=owner, read_only=read_only)
        self._check_variable(array)
        self._array = array

    @property
    def device(self) -> str:
        """Device identifier; always 'cpu' for host handles."""
        return "cpu"

    @property
    def dtype(self) -> str:
        """Element dtype name of the payload."""
        return np.dtype(self._array.dtype).str

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the payload, matching the variable's dims."""
        return tuple(self._array.shape)

    @property
    def nbytes(self) -> int:
        """Payload size in bytes (no synchronization needed on host)."""
        return int(self._array.nbytes)

    def materialize(
        self,
        target_device: str | None = None,
        copy_policy: CopyPolicy = "allow",
    ) -> np.ndarray:
        """Materialize the payload on the requested device.

        Host payloads are already on ``cpu``; other devices require a backend
        transfer and must go through :class:`BackendArrayHandle` creation by
        the backend — host handles never perform implicit host-to-device
        copies, regardless of the copy policy. Read-only handles return a
        non-writeable view.
        """
        self._check_live()
        if target_device in (None, "cpu"):
            if self._read_only:
                view = self._array[:]
                view.flags.writeable = False
                return view
            return self._array
        raise RuntimeError(
            f"host handle cannot materialize on {target_device!r}; device "
            "transfers are performed by backends creating backend handles"
        )


class BackendArrayHandle(_ArrayHandleBase):
    """Owner-managed handle over a NumPy-API-compatible device array.

    The wrapped array must expose ``dtype``/``shape``/``nbytes`` with NumPy
    semantics (CuPy arrays qualify). ``device`` is an explicit identifier such
    as ``"cuda:0"``; metadata queries never synchronize with the device.
    """

    def __init__(
        self,
        array: Any,
        variable_schema: VariableSchema,
        *,
        owner: str,
        device: str,
        read_only: bool = False,
    ) -> None:
        super().__init__(variable_schema, owner=owner, read_only=read_only)
        self._check_variable(array)
        self._array = array
        self._device = device

    @property
    def device(self) -> str:
        """Device identifier, e.g. 'cuda:0'."""
        return self._device

    @property
    def dtype(self) -> str:
        """Element dtype name of the payload."""
        return np.dtype(self._array.dtype).str

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the payload, matching the variable's dims."""
        return tuple(self._array.shape)

    @property
    def nbytes(self) -> int:
        """Payload size in bytes without triggering synchronization."""
        return int(self._array.nbytes)

    def materialize(
        self,
        target_device: str | None = None,
        copy_policy: CopyPolicy = "allow",
    ) -> Any:
        """Materialize the payload on the requested device.

        ``target_device=None`` (or the handle's own device) returns the
        device array itself. ``target_device="cpu"`` performs an *explicit*
        device-to-host copy only under ``copy_policy="allow"``; with
        ``"never"`` it raises. Cross-device copies are a Phase 3 concern.
        """
        self._check_live()
        if target_device in (None, self._device):
            return self._array
        if target_device == "cpu":
            if copy_policy == "never":
                raise RuntimeError(
                    "payload does not reside on 'cpu' and copy policy is "
                    "'never'"
                )
            return convert_to_numpy(self._array)
        raise RuntimeError(
            f"cannot materialize on {target_device!r}: cross-device copies "
            "are not part of the Phase 1 handle contract"
        )


class ReadOnlyArrayView:
    """Read-only delegating view of another handle.

    Shares the inner handle's lifetime: closing the owner handle invalidates
    the view (consumers observe an error on next access). Leases acquired
    through the view reference the view, so consumers never reach the
    writable inner handle. Read-only enforcement of device arrays is
    best-effort (the flag is advisory for backends without write flags).
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._leases: set[DataLease] = set()

    @property
    def variable_schema(self) -> VariableSchema:
        """Schema of the single variable this handle backs."""
        return self._inner.variable_schema

    @property
    def device(self) -> str:
        """Device identifier of the inner handle."""
        return self._inner.device

    @property
    def dtype(self) -> str:
        """Element dtype name of the payload."""
        return self._inner.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the payload, matching the variable's dims."""
        return self._inner.shape

    @property
    def nbytes(self) -> int:
        """Payload size in bytes (without triggering synchronization)."""
        return self._inner.nbytes

    @property
    def read_only(self) -> bool:
        """Views are always read-only."""
        return True

    @property
    def owner(self) -> str:
        """Identifier of the component owning the buffer's lifetime."""
        return self._inner.owner

    def acquire(
        self, consumer: str, scope: LeaseScope = "execution"
    ) -> DataLease:
        """Acquire a lease on this view for one consumer."""
        self._check_live()
        lease = DataLease(self, consumer, scope)
        self._leases.add(lease)
        return lease

    def materialize(
        self,
        target_device: str | None = None,
        copy_policy: CopyPolicy = "allow",
    ) -> Any:
        """Materialize the payload read-only on the requested device."""
        self._check_live()
        result = self._inner.materialize(target_device, copy_policy)
        if isinstance(result, np.ndarray):
            view = result[:]
            view.flags.writeable = False
            return view
        return result

    # -- internals ----------------------------------------------------------

    def _release_lease(self, lease: DataLease) -> None:
        self._leases.discard(lease)

    def _check_live(self) -> None:
        if getattr(self._inner, "closed", False):
            raise RuntimeError(
                f"the owner handle of this read-only view (variable "
                f"{self._inner.variable_schema.name!r}) has been closed"
            )
