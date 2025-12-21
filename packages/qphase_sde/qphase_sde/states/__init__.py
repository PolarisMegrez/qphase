"""qphase_sde: States Subpackage
---------------------------

This subpackage defines and provides state plugin implementations.
States encapsulate trajectory data at single time steps and trajectory sets
for entire simulations.

Each state implementation is backend-specific but implements common protocols.
"""

from .base import StateBase, TrajectorySetBase
from .cupy_state import State as CuPyState
from .numpy_state import State as NumpyState
from .torch_state import State as TorchState

__all__ = [
    # Base protocols
    "StateBase",
    "TrajectorySetBase",
    # Implementations
    "NumpyState",
    "CuPyState",
    "TorchState",
]
