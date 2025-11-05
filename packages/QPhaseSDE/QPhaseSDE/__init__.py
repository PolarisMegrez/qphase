"""
QPhaseSDE - Phase-Space Stochastic Differential Equation Simulation Framework
=============================================================================
A lightweight, extensible framework for phase-space stochastic differential
equation (SDE) simulation and analysis, designed for quantum optics research.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 0.1.3 (Nov 2025)

Notes
-----
Package initialization wires up the global registry and triggers
self-registration for builtin modules (integrators, noise models, backends).
Visualization is intentionally not imported at top-level to avoid optional
dependencies (install extras 'viz' or use QPhaseSDE_cli for CLI/plots).
"""

# Expose the global registry singleton
from .core.registry import registry  # noqa: F401
from .core.engine import run  # noqa: F401
from .core.protocols import SDEModel, NoiseSpec  # noqa: F401
from .backends.factory import get_backend  # noqa: F401

# Trigger self-registration for built-in modules.
# Keep imports lightweight and avoid importing heavy submodules here; rely on
# per-package __init__ to perform lazy registration as needed.
from . import integrators as _qps_integrators  # noqa: F401
from . import noise_models as _qps_noise_models  # noqa: F401
from . import backends as _qps_backends  # noqa: F401

# Public version string
__version__ = "0.1.3 (Nov 2025)"

__all__ = [
	"registry",
	"run",
	"SDEModel",
	"NoiseSpec",
	"get_backend",
	"__version__",
]

