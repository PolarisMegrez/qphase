"""QPhaseSDE core package.

Provides the core SDE engine, data containers, solvers, IO utilities, and
visualization helpers for complex-valued multi-mode stochastic simulations.

Version: 0.1.1

Package initialization also wires up the global registry and triggers
self-registration for builtin modules.
"""

# Expose the global registry singleton
from .core.registry import registry  # noqa: F401

# Trigger self-registration for built-in modules.
# Keep imports lightweight and avoid importing heavy submodules here; rely on
# per-package __init__ to perform lazy registration as needed.
from . import integrators as _qps_integrators  # noqa: F401
from . import noise_models as _qps_noise_models  # noqa: F401
from . import backends as _qps_backends  # noqa: F401
from . import visualizers as _qps_visualizers  # noqa: F401

# Public version string
__version__ = "0.1.1"

__all__ = [
	"registry",
	"__version__",
]

