"""qphase_sde: Utilities
---------------------------------------------------------
Common utility functions for SDE simulation.
"""

from typing import Any

from qphase.backend.base import BackendBase


def expand_complex_noise_backend(Lc: Any, backend: BackendBase) -> Any:
    """Expand complex-basis diffusion matrix to an equivalent real basis.

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
    a = backend.real(Lc)
    b = backend.imag(Lc)

    # Ensure s matches the precision of Lc to avoid promotion to float64
    s_val = (2.0) ** 0.5
    if hasattr(a, "dtype"):
        s = backend.asarray(s_val, dtype=a.dtype)
    else:
        s = s_val

    Lr_real = backend.concatenate((a / s, -b / s), axis=-1)
    Lr_imag = backend.concatenate((b / s, a / s), axis=-1)

    # Use backend-compatible imaginary unit
    j = backend.asarray(1j, dtype=Lc.dtype)
    return Lr_real + j * Lr_imag
