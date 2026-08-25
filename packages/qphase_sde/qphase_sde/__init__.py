"""qphase-sde - SDE Simulation Framework
====================================
A lightweight, extensible framework for phase-space stochastic differential
equation (SDE) simulation and analysis, designed for quantum optics research.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 1.0.1 (Jan 2026)
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import Engine
    from .model import MatrixDriftSDEModel, NoiseSpec, SDEModel
    from .result import SDEDataBundle
    from .state import State, TrajectorySet

# Public version string
__version__ = "1.0.1"

__all__ = [
    "Engine",
    "SDEModel",
    "MatrixDriftSDEModel",
    "NoiseSpec",
    "SDEDataBundle",
    "State",
    "TrajectorySet",
    "__version__",
]

# Root-level re-exports are resolved lazily (PEP 562) so that importing
# declaration-only modules such as ``qphase_sde.manifest`` or
# ``qphase_sde.contracts`` never pulls the engine, concrete plugins or
# backends. Plugin registration is entry-point based; no eager submodule
# import is required here.
_LAZY_EXPORTS = {
    "Engine": ".engine",
    "MatrixDriftSDEModel": ".model",
    "NoiseSpec": ".model",
    "SDEDataBundle": ".result",
    "SDEModel": ".model",
    "State": ".state",
    "TrajectorySet": ".state",
}


def __getattr__(name: str) -> Any:
    """Resolve public root re-exports on first access."""
    module_suffix = _LAZY_EXPORTS.get(name)
    if module_suffix is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}{module_suffix}")
    return getattr(module, name)
