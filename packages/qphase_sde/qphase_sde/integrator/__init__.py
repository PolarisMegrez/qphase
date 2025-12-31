"""qphase_sde: Integrator Subpackage
---------------------------------

This subpackage defines and provides integrator plugin implementations.
Integrators solve stochastic differential equations using numerical methods.

Each integrator implementation must adhere to the Integrator protocol and can have
its own configuration schema.
"""

from .base import Integrator
from .euler_maruyama import EulerMaruyama
from .milstein import Milstein
from .srk import GenericSRK

__all__ = [
    # Base protocol
    "Integrator",
    # Implementations
    "EulerMaruyama",
    "Milstein",
    "GenericSRK",
]
