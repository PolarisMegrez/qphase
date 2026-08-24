"""qphase_sde: Canonical Phase-Space Coordinates
---------------------------------------------------------
Shared real-coordinate conventions for the density-like matrix
``R = alpha alpha^dagger`` built from a complex mode-amplitude state.

The canonical flattening is used by both offline analysers
(``analyser.trajectory_diagnostics``) and online observers
(``observer.first_passage`` matrix projections) so that schema strings and
numeric values stay consistent across the package. The helpers are
backend-agnostic: they accept NumPy or CuPy arrays and compute with the
matching array namespace.

Public API
----------
``CANONICAL_COORDINATE_LAYOUT`` : Human-readable layout description.
``R_CONVENTION`` : Definition of the R matrix convention.
``canonical_r_coordinates`` : Vectorized real coordinates of R.
``canonical_vector`` : Validate a flat canonical vector.
"""

from typing import Any

import numpy as np
from qphase.backend.xputil import get_xp

__all__ = [
    "CANONICAL_COORDINATE_LAYOUT",
    "R_CONVENTION",
    "canonical_r_coordinates",
    "canonical_vector",
]

# Canonical R = alpha alpha^dagger coordinates (real, length n_modes**2).
CANONICAL_COORDINATE_LAYOUT = (
    "[diag(R)_0..diag(R)_{n-1}, Re(R_ij) for i<j in lexicographic order, "
    "Im(R_ij) for i<j in lexicographic order]"
)
R_CONVENTION = "R[i,j] = alpha_i * conj(alpha_j)"


def canonical_r_coordinates(values: Any) -> Any:
    """Real canonical coordinates of ``R = alpha alpha^dagger``.

    ``R[..., i, j] = alpha[..., i] * conj(alpha[..., j])`` is vectorized along
    a new last axis of length ``n_modes**2`` following
    ``CANONICAL_COORDINATE_LAYOUT``; the diagonal entries are the mode
    populations ``|alpha_i|**2``. Works with any array namespace supported by
    ``qphase.backend.xputil.get_xp`` (NumPy, CuPy).
    """
    xp = get_xp(values)
    n_modes = values.shape[-1]
    r_matrix = values[..., :, None] * xp.conj(values[..., None, :])
    diagonal = xp.real(xp.diagonal(r_matrix, axis1=-2, axis2=-1))
    upper_i, upper_j = xp.triu_indices(n_modes, k=1)
    upper = r_matrix[..., upper_i, upper_j]
    return xp.concatenate([diagonal, xp.real(upper), xp.imag(upper)], axis=-1)


def canonical_vector(raw: list[float], n_coordinates: int, label: str) -> np.ndarray:
    """Validate a user-supplied flat canonical vector against ``n_modes**2``."""
    vector = np.asarray(raw, dtype=float)
    if vector.ndim != 1 or vector.size != n_coordinates:
        raise ValueError(
            f"{label} must be a flat real vector of length n_modes**2 = {n_coordinates}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")
    return vector
