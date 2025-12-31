"""qphase-sde - SDE Simulation Framework
====================================
A lightweight, extensible framework for phase-space stochastic differential
equation (SDE) simulation and analysis, designed for quantum optics research.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 1.0.0 (Jan 2026)
"""

# Import protocols from CLI
# Trigger self-registration for built-in modules.
# Keep imports lightweight and avoid importing heavy submodules here; rely on
# per-package __init__ to perform lazy registration as needed.
from . import integrator as _qps_integrators  # noqa: F401

# Import Engine class (v0.2 OO interface)
from .engine import Engine  # noqa: F401
from .model import NoiseSpec, SDEModel  # noqa: F401
from .state import State, TrajectorySet  # noqa: F401

# Public version string
__version__ = "1.0.0"

__all__ = [
    "Engine",
    "SDEModel",
    "NoiseSpec",
    "State",
    "TrajectorySet",
    "__version__",
]
