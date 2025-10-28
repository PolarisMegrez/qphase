from __future__ import annotations

"""Minimal cross-domain protocols (no third-party imports).

This module defines the smallest set of abstract interfaces required for
interoperation between domains (backends, states, integrators, noise models,
visualizers, io). It MUST NOT import numpy/torch/jax/cupy/etc.

Design principles
-----------------
- Keep the interface surface area minimal and stable.
- Express only capabilities that higher layers truly depend on.
- Implementation details and third-party deps belong to their domain packages.

Concurrency and mutation
------------------------
- Unless explicitly documented, methods are pure or return new objects.
- Methods with in-place effects must document side-effects clearly.
- data_view() returns a view/alias when possible and MUST NOT copy.
- copy() returns an independent deep copy suitable for mutation.

Copy vs view
------------
- view(...) returns a lightweight view that may alias memory; modifying the
  underlying storage reflects in all views. Not guaranteed to persist across
  backend/device transfers.
- copy(...) returns a deep copy that owns its storage and is safe to mutate.

Thread-safety
-------------
- Unless stated otherwise, instances are not thread-safe. Callers must
  synchronize concurrent writes to underlying arrays.
"""

from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Callable
from dataclasses import dataclass


class RNGBase(Protocol):
    """Abstract random-number generator handle.

    Opaque to callers. Implementations may wrap a concrete RNG object.

    Required behaviors (semantic):
    - seed(value): deterministically seed the stream when supported.
    - spawn(n): optional; return a list of independent RNGBase streams.
    """

    def seed(self, value: Optional[int]) -> None: ...

    def spawn(self, n: int) -> list[RNGBase]: ...


class BackendBase(Protocol):
    """Minimal backend capabilities used by the engine and domains.

    Only declare abstract signatures here; concrete libraries live in backends/.

    Notes:
    - Implementations may expose additional optimized methods; core must rely
      only on methods defined here.
    - dtype/device types are treated as opaque Any at the core layer.
    """

    # Identification/metadata
    def backend_name(self) -> str: ...
    def device(self) -> Optional[str]: ...

    # Array creation / conversion
    def array(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def asarray(self, obj: Any, dtype: Any | None = None) -> Any: ...
    def zeros(self, shape: Tuple[int, ...], dtype: Any) -> Any: ...
    def empty(self, shape: Tuple[int, ...], dtype: Any) -> Any: ...
    def empty_like(self, x: Any) -> Any: ...
    def copy(self, x: Any) -> Any: ...

    # Basic ops / linalg
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def concatenate(self, arrays: Tuple[Any, ...], axis: int = -1) -> Any: ...
    def cholesky(self, a: Any) -> Any: ...

    # Complex helpers
    def real(self, x: Any) -> Any: ...
    def imag(self, x: Any) -> Any: ...

    # RNG
    def rng(self, seed: Optional[int]) -> RNGBase: ...
    def randn(self, rng: RNGBase, shape: Tuple[int, ...], dtype: Any) -> Any: ...
    def spawn_rngs(self, master_seed: int, n: int) -> list[RNGBase]: ...


class Serializable(Protocol):
    """Serialization contract.

    serialize() MUST return a JSON-serializable Mapping. Implementations should
    avoid embedding raw array buffers; store lightweight metadata or references
    suitable for snapshotting.
    """

    def serialize(self) -> Mapping[str, Any]: ...

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> "Serializable": ...


class Snapshotable(Protocol):
    """Snapshot contract for lightweight run manifests."""

    def snapshot_meta(self) -> Mapping[str, Any]: ...


class StateBase(Protocol):
    """Minimal state container for trajectories at a single time.

    Storage is backend-defined; callers must not assume array types.

    Required attributes:
    - y: backend array of shape (n_traj, n_modes), complex-like
    - t: float timestamp
    - attrs: dict of lightweight metadata
    """

    y: Any
    t: float
    attrs: Dict[str, Any]

    # Shapes
    @property
    def n_traj(self) -> int: ...

    @property
    def n_modes(self) -> int: ...

    # Views and copies
    def data_view(self) -> Any:
        """Return a non-copying view/alias of underlying array when possible."""
        ...

    def view(self, *, modes: Optional[Any] = None, trajectories: Optional[Any] = None) -> "StateBase": ...
    def copy(self) -> "StateBase": ...

    # Domain operations
    def select_modes(self, idx: Any) -> "StateBase": ...
    def slice_trajectories(self, idx: Any) -> "StateBase": ...

    # Backend migration
    def to_backend(self, target_backend: BackendBase, *, copy_if_needed: bool = True) -> "StateBase": ...


class TrajectorySetBase(Protocol):
    """Minimal time-series container for multiple trajectories."""

    data: Any
    t0: float
    dt: float
    meta: Dict[str, Any]

    @property
    def n_traj(self) -> int: ...

    @property
    def n_steps(self) -> int: ...

    @property
    def n_modes(self) -> int: ...

    def times(self) -> Any: ...


# -------------------------- Models and Specs --------------------------

DriftFn = Callable[[Any, float, Dict], Any]
DiffusionFn = Callable[[Any, float, Dict], Any]
JacobianFn = Callable[[Any, float, Dict], Any]


@dataclass
class SDEModel:
    """Minimal contract for SDE models consumed by the engine.

    The model supplies drift and diffusion evaluated on batches of states.
    noise_basis determines whether diffusion is defined in the real or
    complex basis (the engine may expand complex to real noise channels).
    """

    name: str
    n_modes: int
    noise_basis: str  # "real" | "complex"
    noise_dim: int
    params: Dict[str, Any]
    drift: DriftFn
    diffusion: DiffusionFn
    diffusion_jacobian: Optional[JacobianFn] = None


@dataclass
class NoiseSpec:
    """Specification of real-valued noise channels used by the engine.

    kind: 'independent' | 'correlated'
    dim: number of real channels
    covariance: optional (M,M) real covariance for correlated noise
    """

    kind: str
    dim: int
    covariance: Optional[Any] = None
