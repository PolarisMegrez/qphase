"""qphase_cam: canonical sample-matrix noise covariance
---------------------------------------------------------
The Wigner/moment diffusion a CAM model provides is written in the mode basis
``(i, j)``; a linear-noise assessment needs it in the canonical Hermitian vector
basis the dynamics Jacobian already acts on. This module owns that transform once,
so the stochastic assessment and the ``stochastic_validity`` postprocessor use the
same covariance instead of each keeping its own version.

The transform is exact and representation-fixed: the diagonal entries map to the
diagonal, and the real/imaginary off-diagonal pairs map to the symmetric and
antisymmetric canonical coordinates of the Hermitian basis (the ``M = 0``
covariance of a factorized sample matrix). It is *not* a physical diffusion: a
non-trivial moment closure is recorded by the caller, never by this matrix.
"""

from __future__ import annotations

import numpy as np

__all__ = ["canonical_hermitian_transform", "canonical_sample_matrix_covariance"]


def canonical_sample_matrix_covariance(
    state: np.ndarray, diffusion: np.ndarray
) -> np.ndarray:
    """Return the M=0 sample-matrix noise covariance in canonical coordinates.

    ``state`` is the square state matrix and ``diffusion`` the model's diffusion
    matrix in the same shape. The result is real, symmetric and of order
    ``modes**2`` — the coordinate count the dynamics Jacobian uses.

    Raises
    ------
    ValueError
        If the state is not square, the diffusion has a different shape, or the
        transformed covariance is not real.

    """
    state = np.asarray(state, dtype=complex)
    diffusion = np.asarray(diffusion, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("CAM state must be a square matrix")
    if diffusion.shape != state.shape:
        raise ValueError("CAM diffusion must have the same shape as the state")
    modes = state.shape[0]
    complex_covariance = np.empty((modes * modes, modes * modes), dtype=complex)
    for i in range(modes):
        for j in range(modes):
            for k in range(modes):
                for right in range(modes):
                    complex_covariance[i * modes + j, k * modes + right] = (
                        diffusion[i, right] * state[k, j]
                        + diffusion[k, j] * state[i, right]
                    )
    transform = canonical_hermitian_transform(modes)
    covariance = np.real_if_close(transform @ complex_covariance @ transform.T)
    if np.iscomplexobj(covariance):
        raise ValueError("canonical CAM noise covariance is not real")
    return np.asarray(covariance, dtype=float)


def canonical_hermitian_transform(modes: int) -> np.ndarray:
    """Return the mode-basis to canonical-Hermitian-basis transform.

    One row per canonical coordinate: the ``modes`` diagonal entries, then the
    symmetric off-diagonal pairs, then the antisymmetric (imaginary) pairs.
    """
    transform = np.zeros((modes * modes, modes * modes), dtype=complex)
    row = 0
    for mode in range(modes):
        transform[row, mode * modes + mode] = 1.0
        row += 1
    for left in range(modes):
        for right in range(left + 1, modes):
            transform[row, left * modes + right] = 0.5
            transform[row, right * modes + left] = 0.5
            row += 1
    for left in range(modes):
        for right in range(left + 1, modes):
            transform[row, left * modes + right] = -0.5j
            transform[row, right * modes + left] = 0.5j
            row += 1
    return transform
