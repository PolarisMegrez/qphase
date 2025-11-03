"""
QPhaseSDE: Core Protocols
-------------------------
Minimal abstract interfaces (Protocols and lightweight data classes) shared
across all domains — backends, states, integrators, noise models, visualizers,
and IO. This module is dependency-light and must not import numpy, torch,
jax, or similar libraries.

Behavior
--------
- (*Design principles*) Keep interface surfaces minimal and stable; express
  only capabilities required by higher layers and place implementation details
  and third-party dependencies in domain packages rather than here.
- (*Mutation and purity*) Unless explicitly documented, methods are pure or
  return new objects; methods that perform in-place mutation must document
  side-effects clearly and unambiguously.
- (*Views and copies*) `data_view()` returns a lightweight alias when possible
  and must not copy, while `copy()` returns an independent deep copy that is
  safe for mutation; views may not persist across backend or device transfers.
- (*Thread-safety*) Instances are not thread-safe by default; callers must
  synchronize concurrent writes to shared arrays or otherwise ensure safe
  access patterns.

Notes
-----
- These contracts are intentionally minimal. Domain subpackages may define
  extended or specialized protocols to enable backend-specific optimizations
  or additional capabilities.
"""

from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Callable
from dataclasses import dataclass

__all__ = [
  "RNGBase",
  "BackendBase",
  "Serializable",
  "Snapshotable",
  "StateBase",
  "TrajectorySetBase",
  "DriftFn",
  "DiffusionFn",
  "JacobianFn",
  "SDEModel",
  "NoiseSpec",
]

class RNGBase(Protocol):
    """Abstract random-number generator handle (public protocol).

    Opaque to callers; implementations may wrap a concrete RNG object while
    providing a consistent seeding and spawning interface.
    """

    def seed(self, value: Optional[int]) -> None: ...

    def spawn(self, n: int) -> list["RNGBase"]: ...

class BackendBase(Protocol):
    """Minimal backend protocol for array ops, linalg, RNG, and helpers.

    Concrete backends live under ``backends/`` and must implement the methods
    below. Core layers rely only on this interface; additional methods may be
    provided but must be guarded by capability checks.

    Methods
    -------
    backend_name() -> str
        Return backend identifier.
    device() -> Optional[str]
        Return device string when applicable (e.g., "cuda:0"), else None.
    capabilities() -> Dict[str, Any]
        Report unified, backend-agnostic features for discovery.
    array/ asarray/ zeros/ empty/ empty_like/ copy
        Array creation and conversion helpers.
    einsum(subscripts, *operands) -> Any
        General tensor contraction.
    concatenate(arrays, axis=-1) -> Any
        Concatenate along a given axis.
    cholesky(a) -> Any
        Cholesky factorization.
    real(x)/imag(x) -> Any
        Complex helpers to access views of real/imag parts.
    rng(seed) -> RNGBase
        Create an RNG handle.
    randn(rng, shape, dtype) -> Any
        Standard normal sampling.
    spawn_rngs(master_seed, n) -> list[RNGBase]
        Spawn independent RNG streams deterministically.
    stack(arrays, axis=0) -> Any
        Optional helper to stack arrays along a new axis.
    to_device(x, device) -> Any
        Optional helper to move/adapt arrays to a device.
    """

    # Identification/metadata
    def backend_name(self) -> str: ...
    def device(self) -> Optional[str]: ...
    def capabilities(self) -> Dict[str, Any]: ...  # minimal, unified capability keys

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

    # Optional convenience (not required by core, but used opportunistically)
    # Implementers may raise AttributeError if not supported; callers must guard.
    def stack(self, arrays: Tuple[Any, ...], axis: int = 0) -> Any: ...  # optional
    def to_device(self, x: Any, device: Optional[str]) -> Any: ...  # optional

class Serializable(Protocol):
    """Serialization contract for lightweight, JSON-friendly payloads.

    Implementations should avoid embedding raw array buffers; prefer metadata
    or external references suitable for snapshotting.
    """

    def serialize(self) -> Mapping[str, Any]: ...

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> "Serializable": ...

class Snapshotable(Protocol):
    """Snapshot contract for lightweight run manifests (public protocol)."""

    def snapshot_meta(self) -> Mapping[str, Any]: ...

class StateBase(Protocol):
    """Minimal state container for trajectories at a single time step.

    Encapsulates backend-defined storage of trajectory values and associated
    metadata. Callers must not assume specific array types.

    Attributes
    ----------
    y : Any
        Backend array of shape ``(n_traj, n_modes)``, complex-like.
    t : float
        Time stamp of the state.
    attrs : dict
        Lightweight metadata associated with the state.

    Methods
    -------
    n_traj -> int
        Number of trajectories.
    n_modes -> int
        Number of modes (state dimension).
    data_view() -> Any
        Return a non-copying view/alias of the underlying array when possible.
    view(modes=None, trajectories=None) -> StateBase
        Lightweight slicing/view into modes and/or trajectories.
    copy() -> StateBase
        Deep copy safe for mutation.
    select_modes(idx) -> StateBase
        Return a state with selected mode indices.
    slice_trajectories(idx) -> StateBase
        Return a state with selected trajectory indices.
    to_backend(target_backend, copy_if_needed=True) -> StateBase
        Migrate storage to a target backend when supported.
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
    """Minimal time-series container for multiple trajectories.

    Represents a sampled trajectory set produced by the engine. Storage is
    backend-defined and must be treated as opaque by callers.

    Attributes
    ----------
    data : Any
        Backend array shaped ``(n_traj, n_steps, n_modes)``.
    t0 : float
        Initial time.
    dt : float
        Time step between consecutive samples.
    meta : dict
        Lightweight metadata (e.g., backend info, solver tags).

    Methods
    -------
    n_traj -> int
        Number of trajectories.
    n_steps -> int
        Number of stored steps.
    n_modes -> int
        State dimension per trajectory.
    times() -> Any
        Backend array of sample times derived from ``t0`` and ``dt``.
    """

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

    Provides drift and diffusion evaluated on batches of states. ``noise_basis``
    determines whether diffusion is specified in the real or complex basis; the
    engine may expand complex diffusion into real noise channels as needed.

    Attributes
    ----------
    name : str
        Human-readable model name.
    n_modes : int
        State dimension per trajectory.
    noise_basis : str
        Either ``"real"`` or ``"complex"``.
    noise_dim : int
        Number of real noise channels (M).
    params : dict
        Model parameters consumed by drift/diffusion functions.
    drift : Callable[[Any, float, Dict], Any]
        Drift function f(y, t, params) evaluated on batches.
    diffusion : Callable[[Any, float, Dict], Any]
        Diffusion function L(y, t, params) evaluated on batches.
    diffusion_jacobian : Callable[[Any, float, Dict], Any], optional
        Optional Jacobian of diffusion for higher-order schemes.
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
    """Specification of real-valued noise channels for the engine.

    Attributes
    ----------
    kind : str
        Either ``'independent'`` or ``'correlated'``.
    dim : int
        Number of real channels (M).
    covariance : Any, optional
        Real symmetric covariance matrix with shape ``(M, M)`` used when
        ``kind='correlated'``.
    """

    kind: str
    dim: int
    covariance: Optional[Any] = None
