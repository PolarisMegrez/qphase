"""qphase_sde: Models Base Protocols
----------------------------------

Core contracts for SDE models and noise specifications.
These protocols define the mathematical interface for stochastic differential equations.

This module is dependency-light and safe to import in any environment.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "SDEModel",
    "FunctionalSDEModel",
    "NoiseSpec",
    "DriftFn",
    "DiffusionFn",
    "JacobianFn",
]


DriftFn = Callable[[Any, float, dict], Any]
"""Type for drift function f(y, t, params).

Parameters
----------
y : Any
    State array with shape (n_traj, n_modes)
t : float
    Current time
params : dict
    Model parameters

Returns
-------
Any
    Drift vector with shape (n_traj, n_modes)
"""


DiffusionFn = Callable[[Any, float, dict], Any]
"""Type for diffusion function L(y, t, params).

Parameters
----------
y : Any
    State array with shape (n_traj, n_modes)
t : float
    Current time
params : dict
    Model parameters

Returns
-------
Any
    Diffusion matrix with shape (n_traj, n_modes, noise_dim) or (n_traj, n_modes)
    depending on noise basis
"""


JacobianFn = Callable[[Any, float, dict], Any]
"""Type for diffusion Jacobian function.

Parameters
----------
y : Any
    State array with shape (n_traj, n_modes)
t : float
    Current time
params : dict
    Model parameters

Returns
-------
Any
    Jacobian of diffusion with respect to state
"""


@runtime_checkable
class SDEModel(Protocol):
    """Protocol for SDE models consumed by the engine.

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

    """

    name: str
    n_modes: int
    noise_basis: str
    noise_dim: int
    params: dict[str, Any]

    def drift(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        """Compute drift vector."""
        ...

    def diffusion(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        """Compute diffusion matrix."""
        ...


@dataclass
class FunctionalSDEModel:
    """Concrete implementation of SDEModel using functions (Legacy/Functional).

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
    params: dict[str, Any]
    drift: DriftFn
    diffusion: DiffusionFn
    diffusion_jacobian: JacobianFn | None = None


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
    covariance: Any | None = None
