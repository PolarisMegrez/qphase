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

Two batch forms sit beside the single-matrix functions, because a reconstruction
samples thousands of states along one manifold and a per-state Python call is what
made it slow:

* :func:`canonical_sample_matrix_covariances` — the same covariance for a stack of
  states, in canonical coordinates;
* :func:`projected_noise_intensities` — its projection onto a critical-mode left
  eigenvector, one real number per state.

The projection *folds the transform into the left vector* instead of transforming
the covariance first. That is the order the reference evaluation uses, and it is the
order this module keeps: ``transform @ Σ @ transform.T`` and
``(transform.T @ left)ᵀ Σ (transform.T @ left)`` are the same number mathematically
but not in the last bits, and a reconstruction that must reproduce a published
curve cannot choose freely. The single-matrix :func:`projected_intensity` in
:mod:`qphase_cam.validity` keeps its own order for the same reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "canonical_hermitian_transform",
    "canonical_sample_matrix_covariance",
    "canonical_sample_matrix_covariances",
    "projected_noise_intensities",
]

#: Largest imaginary residual of a projected noise intensity, relative to its
#: real scale, that is still rounding error rather than a mixed-coordinate mistake.
PROJECTION_IMAGINARY_TOLERANCE = 1.0e-9


def _check_state_stack(states: Any, diffusions: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the stack and its diffusions as complex arrays, or raise.

    A batch of state matrices is ``(n, modes, modes)`` and its diffusion must have
    the same shape: a mismatched or non-square stack is a caller error, not an
    invalid state to be turned into NaN.
    """
    stack = np.asarray(states, dtype=complex)
    diffusion = np.asarray(diffusions, dtype=complex)
    if stack.ndim != 3:
        raise ValueError(
            "a state-matrix stack must be (n, modes, modes); got shape "
            f"{stack.shape}"
        )
    if stack.shape[1] != stack.shape[2]:
        raise ValueError(
            f"a state-matrix stack must be square in its last two axes; got shape "
            f"{stack.shape}"
        )
    if diffusion.shape != stack.shape:
        raise ValueError(
            f"the diffusion stack has shape {diffusion.shape} for state matrices of "
            f"shape {stack.shape}"
        )
    return stack, diffusion


def _mode_basis_covariances(
    stack: np.ndarray, diffusion: np.ndarray
) -> np.ndarray:
    """Return the ``(n, modes², modes²)`` sample-matrix covariance in the mode basis.

    The entry is ``D[i, r] R[k, j] + D[k, j] R[i, r]`` at ``[i·modes + j, k·modes + r]``
    — the same four nested indices :func:`canonical_sample_matrix_covariance` fills,
    written as two broadcast products. Each entry is one product pair, so the values
    are the loop's values exactly.
    """
    count, modes = stack.shape[0], stack.shape[1]
    # Axis order is (i, k, j, r) while broadcasting, then transposed to the
    # (i, j, k, r) order the flattened index pairs use.
    entries = diffusion[:, :, None, None, :] * stack[:, None, :, :, None] + (
        diffusion[:, None, :, :, None] * stack[:, :, None, None, :]
    )
    ordered = np.ascontiguousarray(np.transpose(entries, (0, 1, 3, 2, 4)))
    return ordered.reshape(count, modes * modes, modes * modes)


def canonical_sample_matrix_covariances(
    states: Any, diffusions: Any
) -> np.ndarray:
    """Return the M=0 sample-matrix noise covariance of a state stack.

    ``states`` and ``diffusions`` are ``(n, modes, modes)``; the result is
    ``(n, modes², modes²)`` real. It is the batch form of
    :func:`canonical_sample_matrix_covariance` and raises where that function does:
    a non-square or mismatched stack, or a transform that does not come out real.

    Raises
    ------
    ValueError
        If the stacks are not ``(n, modes, modes)``, do not share a shape, or the
        transformed covariance is not real.

    """
    stack, diffusion = _check_state_stack(states, diffusions)
    modes = stack.shape[1]
    transform = canonical_hermitian_transform(modes)
    covariance = np.real_if_close(
        transform @ _mode_basis_covariances(stack, diffusion) @ transform.T
    )
    if np.iscomplexobj(covariance):
        raise ValueError("canonical CAM noise covariance is not real")
    return np.asarray(covariance, dtype=float)


def projected_noise_intensities(
    states: Any, diffusions: Any, left: Any
) -> np.ndarray:
    """Return ``leftᵀ Σ_i left`` for every state matrix of a stack.

    ``states`` and ``diffusions`` are ``(n, modes, modes)`` and ``left`` is the
    ``modes²`` critical-mode left eigenvector in canonical coordinates. The result is
    one real intensity per state.

    A row whose state or diffusion is not finite keeps its NaN: an unusable state is a
    domain verdict of the caller's, not an exception here. A row whose projection is
    not real by more than :data:`PROJECTION_IMAGINARY_TOLERANCE` *is* an exception —
    it means the caller projected along a vector that is not a canonical coordinate
    direction.

    Raises
    ------
    ValueError
        If the stacks are malformed, ``left`` does not have ``modes²`` entries, or a
        finite row's projection is not real.

    """
    stack, diffusion = _check_state_stack(states, diffusions)
    modes = stack.shape[1]
    direction = np.asarray(left, dtype=float).reshape(-1)
    if direction.size != modes * modes:
        raise ValueError(
            f"the left vector has {direction.size} entries for {modes * modes} "
            "canonical coordinates"
        )
    covariance = _mode_basis_covariances(stack, diffusion)
    transform = canonical_hermitian_transform(modes)
    projected_left = transform.T @ direction
    projected = np.einsum(
        "i,nij,j->n", projected_left, covariance, projected_left, optimize=True
    )
    real = np.real(projected)
    imaginary = np.abs(np.imag(projected))
    # The scale and the residual are both taken over the rows that are finite: a row
    # that is not finite is the caller's invalid-domain verdict, and it must not turn
    # the reality check of the other rows into a NaN comparison that never fails.
    finite = np.isfinite(real) & np.isfinite(imaginary)
    amplitude = float(np.max(np.abs(real[finite]))) if np.any(finite) else 0.0
    scale = max(amplitude, 1.0)
    if np.any(imaginary[finite] > PROJECTION_IMAGINARY_TOLERANCE * scale):
        residual = float(np.max(imaginary[finite])) / scale
        raise ValueError(
            "batch canonical covariance projection is not real: relative residual "
            f"{residual:.3e}"
        )
    return np.asarray(real, dtype=float)


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
