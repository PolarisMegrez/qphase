"""qphase: Runtime Data Handle Contracts
---------------------------------------------------------
Freezes the minimal interface (not the implementation) of in-process data
handles and leases. A handle backs exactly one *variable* of a data product
and may be device-resident; leases carry the consumer-facing lifetime.

Ownership and failure semantics:

- Every handle has exactly one *owner* (the component that created it). Only
  the owner closes or invalidates the buffer; consumers only ever release
  their leases. When the last lease is released, the owner may reclaim the
  buffer; the concrete reference-counting implementation is a Phase 1/3
  concern and is not part of this contract.
- Handles handed to consumers are read-only views; mutation requires the
  owner's explicit writable handle.
- ``release()`` on a lease is idempotent; using a released lease raises. If
  the owner fails, outstanding leases are invalidated and consumers observe an
  error on next access rather than silently reading freed memory.
- Lease scopes are ``execution`` (one engine invocation) or ``session`` (one
  workflow session). Artifact persistence never depends on a lease.
- ``materialize()`` is the only frozen exchange operation: callers choose the
  target device and copy policy explicitly, and implementations must never
  perform an implicit device-to-host copy. Zero-copy exchange descriptors
  (DLPack/Array API, stream synchronization) are a Phase 3 design and are
  deliberately absent from this frozen protocol.

Public API
----------
DataHandleProtocol
    In-process, possibly device-resident variable buffer contract.
DataLeaseProtocol
    Consumer-facing lifetime contract.
CopyPolicy
    Copy policy of explicit materialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .schema import VariableSchema

__all__ = [
    "CopyPolicy",
    "DataHandleProtocol",
    "DataLeaseProtocol",
    "LeaseScope",
]

#: Lifetime scopes understood by the lease protocol.
LeaseScope = Literal["execution", "session"]

#: Copy policy of explicit materialization: "allow" permits a copy,
#: "never" requires the payload to already reside on the target device.
CopyPolicy = Literal["allow", "never"]


@runtime_checkable
class DataHandleProtocol(Protocol):
    """Contract of an in-process buffer backing one product variable."""

    @property
    def variable_schema(self) -> VariableSchema:
        """Schema of the single variable this handle backs."""
        ...

    @property
    def device(self) -> str:
        """Device identifier, e.g. 'cpu' or 'cuda:0'."""
        ...

    @property
    def dtype(self) -> str:
        """Element dtype name of the payload."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the payload, matching the variable's dims."""
        ...

    @property
    def nbytes(self) -> int:
        """Payload size in bytes (without triggering synchronization)."""
        ...

    @property
    def read_only(self) -> bool:
        """Whether consumers may mutate the underlying buffer."""
        ...

    @property
    def owner(self) -> str:
        """Identifier of the component owning the buffer's lifetime."""
        ...

    def acquire(
        self, consumer: str, scope: LeaseScope = "execution"
    ) -> DataLeaseProtocol:
        """Acquire a lease on this handle for one consumer."""
        ...

    def materialize(
        self,
        target_device: str | None = None,
        copy_policy: CopyPolicy = "allow",
    ) -> Any:
        """Materialize the payload on the requested device.

        With ``copy_policy="never"``, raise if the payload is not already on
        the target device. Implementations must never perform an implicit
        device-to-host copy.
        """
        ...


@runtime_checkable
class DataLeaseProtocol(Protocol):
    """Consumer-facing lifetime contract over a data handle."""

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

    def release(self) -> None:
        """Release the lease; idempotent.

        The buffer is reclaimable by its owner once the last lease is
        released.
        """
        ...
