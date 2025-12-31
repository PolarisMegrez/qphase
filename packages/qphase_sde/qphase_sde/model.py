"""qphase_sde: Model Protocols
---------------------------------------------------------
Core contracts for SDE models and noise specifications.
These protocols define the mathematical interface for stochastic differential equations.

This module is dependency-light and safe to import in any environment.

Public API
----------
`SDEModel` : Protocol for SDE models consumed by the engine.
`DiffusiveSDEModel` : Alias for SDEModel (for backward compatibility).
`FunctionalSDEModel` : Concrete implementation of SDEModel using functions.
`PhaseSpaceModel` : Model defined by phase space drift and diffusion coefficients.
`NoiseSpec` : Specification of real-valued noise channels.
`DriftFn` : Type for drift function.
`DiffusionFn` : Type for diffusion function.
`JacobianFn` : Type for diffusion Jacobian function.
`fpe_to_sde` : Convert a PhaseSpaceModel to a FunctionalSDEModel.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "SDEModel",
    "DiffusiveSDEModel",
    "FunctionalSDEModel",
    "PhaseSpaceModel",
    "NoiseSpec",
    "DriftFn",
    "DiffusionFn",
    "JacobianFn",
    "fpe_to_sde",
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


# Alias for backward compatibility
DiffusiveSDEModel = SDEModel


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


@dataclass
class PhaseSpaceModel:
    """Model defined by phase space drift and diffusion coefficients (FPE).

    Attributes
    ----------
    name : str
        Model name.
    n_modes : int
        Number of modes.
    terms : dict[int, Callable]
        Dictionary of terms: {1: drift_fn, 2: diffusion_fn}.
        drift_fn(y, t, params) -> drift vector
        diffusion_fn(y, t, params) -> diffusion coefficients (diagonal) or matrix
    params : dict[str, Any]
        Model parameters.

    """

    name: str
    n_modes: int
    terms: dict[int, Callable]
    params: dict[str, Any]


def fpe_to_sde(model: PhaseSpaceModel) -> FunctionalSDEModel:
    """Convert a PhaseSpaceModel (FPE) to a FunctionalSDEModel (SDE).

    Automatically constructs the diffusion matrix B from the diffusion coefficient D
    using B = sqrt(D) (assuming D is diagonal/vector of coefficients).

    Parameters
    ----------
    model : PhaseSpaceModel
        The phase space model to convert.

    Returns
    -------
    FunctionalSDEModel
        The equivalent SDE model.

    """
    drift_fn = model.terms[1]
    diff_coeff_fn = model.terms[2]

    def diffusion_wrapper(y: Any, t: float, params: dict[str, Any]) -> Any:
        # Get diffusion coefficients D2
        d2 = diff_coeff_fn(y, t, params)

        # Handle numpy/cupy agnostic
        try:
            import cupy as cp

            xp = cp.get_array_module(y)
        except ImportError:
            xp = np

        # Return sqrt(d2) directly.
        # If d2 is (n_traj, n_modes), this returns (n_traj, n_modes).
        # The engine/integrator should handle this as diagonal noise.
        return xp.sqrt(d2)

    return FunctionalSDEModel(
        name=model.name,
        n_modes=model.n_modes,
        noise_basis="complex",  # Default assumption for FPE usually
        noise_dim=model.n_modes,  # Diagonal noise
        params=model.params,
        drift=drift_fn,
        diffusion=diffusion_wrapper,
    )
