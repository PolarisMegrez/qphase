"""
QPhaseSDE: Noise Model Factory
------------------------------
Instantiate concrete noise models from NoiseSpec and backend via the central
registry. Currently provides Gaussian noise (independent/correlated).

Behavior
--------
- Resolve by stable registry keys to decouple callers from implementations.
    Specific constructor semantics and error codes are documented by functions.

Notes
-----
- Additional noise families can be integrated by registering new builders
    under the 'noise_model' namespace.
"""

from typing import Any
from ..core.protocols import BackendBase as Backend
from ..core.protocols import NoiseSpec
from .protocols import NoiseModel
from ..core.registry import registry

__all__ = [
    "make_noise_model",
]

def make_noise_model(spec: NoiseSpec, backend: Backend) -> NoiseModel:
    """Instantiate a concrete noise model from a NoiseSpec and backend.

    Currently supports only Gaussian noise models (independent or correlated)
    via the registry. Future extensions may support other noise types.

    Parameters
    ----------
    spec : NoiseSpec
        Noise specification dataclass describing the noise type, basis, and
        correlation structure.
    backend : BackendBase
        Backend instance providing array operations and random number generation.

    Returns
    -------
    NoiseModel
        Instantiated noise model object matching the specification and backend.

    Raises
    ------
    QPSConfigError
        Raised for invalid configuration parameters:

        - [404] Unknown registry key. (``namespace:name`` not registered)
    QPSRegistryError
        Raised when the noise model import fails:

        - [402] Noise model cannot be imported from its dotted path.

    Examples
    --------
    >>> from QPhaseSDE.noise_models.factory import make_noise_model
    >>> spec = NoiseSpec(kind="gaussian", basis="real", ...)  # fill required fields
    >>> model = make_noise_model(spec, backend)
    """
    # Currently only Gaussian noise is supported; independent/correlated via spec
    return registry.create("noise_model:gaussian", spec=spec, backend=backend)
