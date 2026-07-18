"""Canonical Hermitian coordinate tests."""

from __future__ import annotations

import numpy as np
from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix


def test_two_mode_coordinate_order():
    matrix = np.array([[1.0, 2.0 + 3.0j], [2.0 - 3.0j, 4.0]])
    vector = matrix_to_vector(matrix)
    np.testing.assert_allclose(vector, [1.0, 4.0, 2.0, 3.0])
    np.testing.assert_allclose(vector_to_matrix(vector, 2), matrix)


def test_batched_three_mode_round_trip():
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3))
    matrices = raw + raw.conj().transpose(0, 2, 1)
    np.testing.assert_allclose(
        vector_to_matrix(matrix_to_vector(matrices), 3), matrices
    )
