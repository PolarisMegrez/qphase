"""qphase_sde: Integrator Subpackage
---------------------------------------------------------
This subpackage defines and provides integrator plugin implementations.
Integrators solve stochastic differential equations using numerical methods.

Each integrator implementation must adhere to the Integrator protocol and can have
its own configuration schema.

Public API
----------
``Integrator`` : Base protocol for integrators.
``CayleyMaruyama`` : Cayley-Maruyama matrix-drift implementation.
``EulerMaruyama`` : Euler-Maruyama integrator implementation.
``Milstein`` : Milstein integrator implementation.
``GenericSRK`` : Generic Stochastic Runge-Kutta integrator implementation.
"""

from .base import ChunkIntegrator, ChunkStepResult, Integrator
from .cayley_maruyama import CayleyMaruyama
from .euler_maruyama import EulerMaruyama
from .milstein import Milstein
from .srk import GenericSRK

__all__ = [
    # Base protocol
    "ChunkIntegrator",
    "ChunkStepResult",
    "Integrator",
    # Implementations
    "CayleyMaruyama",
    "EulerMaruyama",
    "Milstein",
    "GenericSRK",
]
