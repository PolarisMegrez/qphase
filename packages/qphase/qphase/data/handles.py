"""qphase: Runtime Data Handle Contracts
---------------------------------------------------------
Freezes the interface (not the implementation) of in-process data handles and
leases. Handles may be device-resident; leases carry reference-counted
lifetime so a buffer is reclaimed only after its last consumer releases.

Ownership and failure semantics:

- Every handle has exactly one *owner* (the component that created it and is
  responsible for its eventual disposal). Consumers never dispose buffers.
- Handles obtained by consumers are read-only views; mutation requires the
  owner's explicit writable handle.
- ``export_interface()`` returns a descriptor (DLPack/Array API/host) that
  records whether a copy occurred and which stream must be synchronized.
- ``release()`` is idempotent; using a released handle raises. If the owner
  fails, outstanding leases are invalidated and consumers observe an error on
  next access rather than silently reading freed memory.
- Lease scopes are ``execution`` (one engine invocation) or ``session`` (one
  workflow session). Artifact persistence never depends on a lease.

Public API
----------
DataHandleProtocol
    In-process, possibly device-resident buffer contract.
DataLeaseProtocol
    Reference-counted lifetime contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .schema import ProductSchema

__all__ = [
    "DataHandleProtocol",
    "DataLeaseProtocol",
]

#: Lifetime scopes understood by the lease protocol.
LeaseScope = Literal["execution", "session"]


@runtime_checkable
class DataHandleProtocol(Protocol):
    """Contract of an in-process, possibly device-resident data buffer."""

    @property
    def schema(self) -> ProductSchema:
        """Product schema describing the buffer's semantics."""
        ...

    @property
    def device(self) -> str:
        """Device identifier, e.g. 'cpu' or 'cuda:0'."""
        ...

    @property
    def dtype(self) -> str:
        """Element dtype name of the primary payload."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the primary payload."""
        ...

    @property
    def nbytes(self) -> int:
        """Total payload size in bytes (without triggering synchronization)."""
        ...

    @property
    def read_only(self) -> bool:
        """Whether consumers may mutate the underlying buffer."""
        ...

    @property
    def owner(self) -> str:
        """Identifier of the component owning the buffer's lifetime."""
        ...

    def acquire(self, consumer: str, scope: LeaseScope = "execution") -> Any:
        """Acquire a lease on this handle for one consumer."""
        ...

    def release(self) -> None:
        """Release this handle; idempotent. Further access raises."""
        ...

    def materialize(self, *, device: str | None = None, copy: bool = True) -> Any:
        """Materialize the payload on the requested device.

        Implementations must never perform an implicit device-to-host copy:
        callers choose the target device and copy policy explicitly.
        """
        ...

    def export_interface(self) -> Mapping[str, Any]:
        """Return an exchange descriptor (DLPack/Array API/host).

        The descriptor records whether a copy occurred and which stream must
        be synchronized before use. Private buffers that cannot be shared
        safely must raise instead of exporting an unsafe pointer.
        """
        ...


@runtime_checkable
class DataLeaseProtocol(Protocol):
    """Reference-counted lifetime contract over a data handle."""

    @property
    def handle(self) -> DataHandleProtocol:
        """The leased handle."""
        ...

    @property
    def consumer(self) -> str:
        """Identifier of the consuming component."""
        ...

    @property
    def scope(self) -> LeaseScope:
        """Lifetime scope of the lease."""
        ...

    @property
    def pinned(self) -> bool:
        """Whether the underlying buffer is pinned against eviction."""
        ...

    def pin(self) -> None:
        """Pin the buffer against session-cache eviction."""
        ...

    def release(self) -> None:
        """Release the lease; the buffer is reclaimable after the last one."""
        ...
