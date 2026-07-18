"""Canonical real coordinates for Hermitian matrices."""

from __future__ import annotations

from typing import Any

from qphase.backend.xputil import get_xp


def matrix_to_vector(matrix: Any) -> Any:
    """Vectorize Hermitian matrices as diagonal, upper-real, upper-imag."""
    xp = get_xp(matrix)
    matrix = xp.asarray(matrix)
    n = int(matrix.shape[-1])
    diagonal = [xp.real(matrix[..., i, i]) for i in range(n)]
    upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
    real = [xp.real(matrix[..., i, j]) for i, j in upper]
    imag = [xp.imag(matrix[..., i, j]) for i, j in upper]
    return xp.stack(diagonal + real + imag, axis=-1)


def vector_to_matrix(vector: Any, n_modes: int) -> Any:
    """Reconstruct Hermitian matrices from canonical real coordinates."""
    xp = get_xp(vector)
    vector = xp.asarray(vector)
    expected = n_modes * n_modes
    if int(vector.shape[-1]) != expected:
        raise ValueError(
            f"expected {expected} coordinates for {n_modes} modes; "
            f"received {vector.shape[-1]}"
        )
    dtype = xp.complex64 if vector.dtype.itemsize <= 4 else xp.complex128
    matrix = xp.zeros(vector.shape[:-1] + (n_modes, n_modes), dtype=dtype)
    cursor = 0
    for i in range(n_modes):
        matrix[..., i, i] = vector[..., cursor]
        cursor += 1
    upper = [(i, j) for i in range(n_modes) for j in range(i + 1, n_modes)]
    real_start = cursor
    imag_start = cursor + len(upper)
    for offset, (i, j) in enumerate(upper):
        value = vector[..., real_start + offset] + 1j * vector[..., imag_start + offset]
        matrix[..., i, j] = value
        matrix[..., j, i] = xp.conj(value)
    return matrix


def symbolic_hermitian_matrix(n_modes: int) -> tuple[Any, tuple[Any, ...]]:
    """Return a SymPy Hermitian matrix using the canonical coordinate order."""
    import sympy as sp

    diagonal = tuple(sp.symbols(f"R_{i + 1}{i + 1}", real=True) for i in range(n_modes))
    upper = [(i, j) for i in range(n_modes) for j in range(i + 1, n_modes)]
    real = tuple(sp.symbols(f"R_{i + 1}{j + 1}_re", real=True) for i, j in upper)
    imag = tuple(sp.symbols(f"R_{i + 1}{j + 1}_im", real=True) for i, j in upper)
    matrix = sp.zeros(n_modes, n_modes)
    for i, value in enumerate(diagonal):
        matrix[i, i] = value
    for offset, (i, j) in enumerate(upper):
        value = real[offset] + sp.I * imag[offset]
        matrix[i, j] = value
        matrix[j, i] = sp.conjugate(value)
    return matrix, diagonal + real + imag
