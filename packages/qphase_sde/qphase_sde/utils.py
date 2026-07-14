"""qphase_sde: Utilities
---------------------------------------------------------
Common utility functions for SDE simulation.
"""

from typing import Any

from qphase.backend.base import BackendBase


def expand_complex_noise_backend(Lc: Any, backend: BackendBase) -> Any:
    """Expand complex-basis diffusion matrix to an equivalent real basis.

    .. deprecated::
        Use :func:`qphase_sde.ops.expand_complex_noise` instead. This wrapper
        is kept for backward compatibility.

    Parameters
    ----------
    Lc : Any
        Complex diffusion matrix.
    backend : BackendBase
        Backend to use for operations.

    Returns
    -------
    Any
        Expanded diffusion matrix in real basis (but complex dtype).

    """
    from qphase_sde import ops

    return ops.expand_complex_noise(Lc, backend)
