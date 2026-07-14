"""Tests for the Cayley-Maruyama matrix-drift integrator."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama


class MatrixModel:
    name = "matrix_model"
    noise_basis = "real"

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix)
        self.n_modes = int(self.matrix.shape[-1])
        self.noise_dim = self.n_modes
        self.params: dict[str, object] = {}

    def drift_matrix(self, y, t, params):
        del t, params
        return np.broadcast_to(self.matrix, y.shape[:-1] + self.matrix.shape)

    def drift(self, y, t, params):
        return np.einsum("...ij,...j->...i", self.drift_matrix(y, t, params), y)

    def diffusion(self, y, t, params):
        del t, params
        return np.zeros(y.shape[:-1] + (self.n_modes, self.noise_dim))


BACKEND = NumpyBackend()


@pytest.mark.parametrize("n_modes", [1, 2, 5, 10])
def test_supports_multiple_mode_counts(n_modes):
    frequencies = np.linspace(0.1, 0.5, n_modes)
    model = MatrixModel(np.diag(-1j * frequencies))
    y = np.ones((3, n_modes), dtype=np.complex128)
    d_w = np.zeros((3, n_modes))

    dy = CayleyMaruyama(fused="off").step(y, 0.0, 0.1, model, d_w, BACKEND)
    y_next = y + dy

    assert y_next.shape == y.shape
    np.testing.assert_allclose(np.abs(y_next), 1.0, rtol=1e-13, atol=1e-13)


def test_matches_direct_batched_solve():
    matrix = np.array(
        [[-0.2 - 0.3j, -0.1j], [-0.1j, -0.4 + 0.2j]],
        dtype=np.complex128,
    )
    model = MatrixModel(matrix)
    y = np.array([[1.0 + 0.5j, -0.25j], [0.3j, 0.2 - 0.1j]])
    d_w = np.zeros((2, 2))
    dt = 0.05

    dy = CayleyMaruyama(fused="off").step(y, 0.0, dt, model, d_w, BACKEND)
    lhs = np.eye(2) - 0.5 * dt * matrix
    rhs_op = np.eye(2) + 0.5 * dt * matrix
    expected = np.stack([np.linalg.solve(lhs, rhs_op @ row) for row in y])

    np.testing.assert_allclose(y + dy, expected, rtol=1e-13, atol=1e-13)


def test_phase_error_is_second_order():
    omega = 0.7
    model = MatrixModel(np.array([[-1j * omega]]))
    y = np.ones((1, 1), dtype=np.complex128)
    d_w = np.zeros((1, 1))
    integrator = CayleyMaruyama(fused="off")

    errors = []
    for dt in (0.2, 0.1):
        y_next = y + integrator.step(y, 0.0, dt, model, d_w, BACKEND)
        measured = -np.angle(y_next[0, 0] / y[0, 0]) / dt
        errors.append(abs(measured - omega))

    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.03)


def test_left_point_diffusion_is_solved_with_drift():
    matrix = np.array([[-0.5 + 0.2j]])
    model = MatrixModel(matrix)
    model.diffusion = lambda y, t, params: np.ones(y.shape[:-1] + (1, 1))
    y = np.array([[1.0 + 0.0j]])
    d_w = np.array([[0.25]])
    dt = 0.1

    dy = CayleyMaruyama(fused="off").step(y, 0.0, dt, model, d_w, BACKEND)
    expected = np.linalg.solve(
        np.eye(1) - 0.5 * dt * matrix,
        (np.eye(1) + 0.5 * dt * matrix) @ y[0] + d_w[0],
    )

    np.testing.assert_allclose(y[0] + dy[0], expected)


def test_requires_matrix_drift():
    class DriftOnlyModel(MatrixModel):
        drift_matrix = None

    model = DriftOnlyModel(np.array([[-1.0]]))
    with pytest.raises(TypeError, match="drift_matrix"):
        CayleyMaruyama(fused="off").step(
            np.ones((1, 1)), 0.0, 0.1, model, np.zeros((1, 1)), BACKEND
        )


def test_enforces_configured_mode_limit():
    model = MatrixModel(np.eye(5))
    with pytest.raises(ValueError, match=r"1\.\.4 modes"):
        CayleyMaruyama(fused="off", max_modes=4).step(
            np.ones((1, 5)), 0.0, 0.1, model, np.zeros((1, 5)), BACKEND
        )
