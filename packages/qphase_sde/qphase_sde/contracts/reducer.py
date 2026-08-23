"""qphase_sde: Reducer Contract (2.0)
---------------------------------------------------------
Freezes the unified reducer lifecycle for analysers that support trajectory
batching, time streaming or scan tiling. Reducers own sufficient statistics
and their merge semantics; the engine drives the lifecycle and checkpoints
mergeable state between tiles/batches.

Checkpoint rules: state serialized by ``checkpoint()`` must be
JSON/portable-array friendly — never opaque backend objects (no CuPy arrays,
no FFT plans). ``restore()`` rebuilds portable state; execution arenas are
re-created separately.

Public API
----------
ReducerProtocol
    The prepare/consume/merge/finalize/checkpoint lifecycle.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ReducerProtocol",
]


@runtime_checkable
class ReducerProtocol(Protocol):
    """Unified reducer lifecycle for batched and streaming analysers."""

    def prepare(self, schema: Any, context: Any) -> Any:
        """Create the initial reducer state for a product schema."""
        ...

    def consume(self, view: Any, state: Any, reporter: Any) -> Any:
        """Fold one data view (batch/segment/tile) into the state."""
        ...

    def merge(self, left: Any, right: Any) -> Any:
        """Merge two states; must be associative and deterministic."""
        ...

    def finalize(self, state: Any) -> Any:
        """Produce the final data product from a state."""
        ...

    def checkpoint(self, state: Any) -> dict[str, Any]:
        """Serialize a state to portable sufficient statistics."""
        ...

    def restore(self, checkpoint: dict[str, Any]) -> Any:
        """Rebuild a state from portable checkpoint data."""
        ...
