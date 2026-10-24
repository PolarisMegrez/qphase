"""Petermann-spectrum and solution-view contracts."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.data import build_state_product
from qphase_cam.postprocessor.petermann import (
    PetermannSpectrum,
    build_bogoliubov_matrix,
    build_monodromy_matrix,
    eigensystem_petermann,
)
from qphase_cam.result import CAMResult

from models.kerr_2mode import Kerr2ModeModel


def test_normal_matrix_has_unit_petermann_factors():
    matrices = np.asarray([np.diag([1.0 + 0.2j, 2.0 - 0.1j])])
    eigenvalues, factors = eigensystem_petermann(matrices)
    np.testing.assert_allclose(eigenvalues, [[1.0 + 0.2j, 2.0 - 0.1j]])
    np.testing.assert_allclose(factors, 1.0)


def test_monodromy_applies_opposite_rayleigh_shifts():
    hamiltonian = np.asarray([np.diag([1.0, 2.0])], dtype=complex)
    interaction = np.zeros_like(hamiltonian)
    bogoliubov = build_bogoliubov_matrix(hamiltonian, interaction)
    monodromy = build_monodromy_matrix(hamiltonian, interaction, np.asarray([0.5]))
    np.testing.assert_allclose(
        np.diag(monodromy[0]), np.diag(bogoliubov[0]) + [-0.5, -0.5, 0.5, 0.5]
    )


def _script_critical_mode_petermann(jacobian: np.ndarray) -> tuple[complex, float]:
    """Return the script's critical-mode Petermann factor.

    ``scan_petermann_bdg.py`` takes the critical mode as the eigenvalue with the
    largest real part, normalises the right vector to unit norm, normalises the
    left vector so ``left·right = 1`` and reports ``left·left``. Reproduced here as
    the comparison semantics, not as a second implementation.
    """
    values, right_vectors = np.linalg.eig(jacobian)
    index = int(np.argmax(np.real(values)))
    value = values[index]
    right = np.real_if_close(right_vectors[:, index], tol=1000)
    left_values, left_vectors = np.linalg.eig(jacobian.T)
    left_index = int(np.argmin(np.abs(left_values - value)))
    left = np.real_if_close(left_vectors[:, left_index], tol=1000)
    assert not np.iscomplexobj(right) and not np.iscomplexobj(left)
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    right /= np.linalg.norm(right)
    overlap = float(left @ right)
    assert abs(overlap) > 1e-12
    left = left / overlap
    return complex(value), float(left @ left)


def test_petermann_factor_matches_the_analytic_two_by_two_value():
    """``[[-1, k], [0, -2]]`` has K = 1 + k² for both modes."""
    k = 2.0
    matrices = np.asarray([[[-1.0, k], [0.0, -2.0]]])
    eigenvalues, factors = eigensystem_petermann(matrices)
    np.testing.assert_allclose(eigenvalues[0], [-2.0, -1.0])
    np.testing.assert_allclose(factors[0], [1.0 + k**2, 1.0 + k**2], rtol=1e-12)


def test_petermann_factor_matches_the_script_critical_mode_convention():
    """The plugin and ``scan_petermann_bdg.py`` report the same critical K."""
    jacobian = np.asarray([[-1.0, 2.0], [0.3, -2.5]])
    value, script_factor = _script_critical_mode_petermann(jacobian)
    eigenvalues, factors = eigensystem_petermann(jacobian[None])
    index = int(np.argmin(np.abs(eigenvalues[0] - value)))
    assert eigenvalues[0][index] == pytest.approx(value)
    assert factors[0][index] == pytest.approx(script_factor, rel=1e-12)
    # A non-normal matrix has excess noise on every mode.
    assert script_factor > 1.0
    assert min(factors[0]) > 1.0


def test_bogoliubov_and_monodromy_blocks_follow_the_script_definition():
    """``[[H+V, V], [-V*, -H*-V*]]`` plus the opposite Rayleigh shifts."""
    hamiltonian = np.asarray([[[1.0, 0.2], [0.1, 2.0]]], dtype=complex)
    interaction = np.asarray([[[0.3j, 0.0], [0.0, -0.4j]]], dtype=complex)
    bogoliubov = build_bogoliubov_matrix(hamiltonian, interaction)
    np.testing.assert_allclose(bogoliubov[0, :2, :2], hamiltonian[0] + interaction[0])
    np.testing.assert_allclose(bogoliubov[0, :2, 2:], interaction[0])
    np.testing.assert_allclose(bogoliubov[0, 2:, :2], -np.conjugate(interaction[0]))
    np.testing.assert_allclose(
        bogoliubov[0, 2:, 2:], -np.conjugate(hamiltonian[0] + interaction[0])
    )
    monodromy = build_monodromy_matrix(hamiltonian, interaction, np.asarray([0.5]))
    np.testing.assert_allclose(
        monodromy[0, :2, :2], hamiltonian[0] + interaction[0] - 0.5 * np.eye(2)
    )
    np.testing.assert_allclose(
        monodromy[0, 2:, 2:],
        -np.conjugate(hamiltonian[0] + interaction[0]) + 0.5 * np.eye(2),
    )
    # A normal (diagonal) Hamiltonian has unit Petermann factors in every frame.
    diagonal = np.asarray([np.diag([1.0, 2.0])])
    zero = np.zeros_like(diagonal)
    for matrices in (
        diagonal,
        build_bogoliubov_matrix(diagonal, zero),
        build_monodromy_matrix(diagonal, zero, np.asarray([0.25])),
    ):
        _, factors = eigensystem_petermann(matrices)
        np.testing.assert_allclose(factors, 1.0, rtol=1e-12)


def test_petermann_postprocessor_emits_aligned_spectra():
    model = Kerr2ModeModel(
        omega_a=0.0,
        omega_b=-0.1,
        chi=0.01,
        gamma_a=0.5,
        gamma_b=1.0,
        g=0.5,
    )
    product = _state_product(
        np.asarray([[[2.0, 0.1j], [-0.1j, 1.0]]], dtype=complex),
        params=model.params,
    )
    output = PetermannSpectrum(chunk_size=1).process(product, _context(model))
    assert np.asarray(output.handle("hamiltonian_eigenvalues").materialize()).shape == (
        1,
        2,
    )
    assert np.asarray(output.handle("bogoliubov_eigenvalues").materialize()).shape == (
        1,
        4,
    )
    assert np.asarray(output.handle("monodromy_eigenvalues").materialize()).shape == (
        1,
        4,
    )
    assert np.isfinite(
        np.asarray(output.handle("hamiltonian_petermann_factors").materialize())
    ).all()


def test_solution_order_is_a_non_mutating_per_point_view():
    states = np.zeros((2, 3, 2, 2), dtype=complex)
    states[:, :, 0, 0] = [[3.0, 1.0, 2.0], [4.0, 5.0, 6.0]]
    result = CAMResult(
        states=states,
        residuals=np.zeros((2, 3)),
        success=np.ones((2, 3), dtype=bool),
        valid_mask=np.asarray([[True, True, True], [True, False, True]]),
        solution_count=np.asarray([3, 2]),
        params={"offset": np.asarray([0.0, 10.0])},
    )
    np.testing.assert_array_equal(
        result.solution_order(descending=True), [[0, 2, 1], [2, 0, -1]]
    )
    custom = result.solution_order(
        lambda state, params: abs(float(np.real(state[0, 0])) - params["offset"])
    )
    np.testing.assert_array_equal(custom, [[1, 2, 0], [2, 0, -1]])
    np.testing.assert_allclose(result.states, states)


def _state_product(states, *, params=None, residuals=None, success=None):
    """Build the typed state product a postprocessor annotates."""
    array = np.asarray(states, dtype=complex)
    if array.ndim == 3:
        array = array[None]
    scan, capacity = array.shape[0], array.shape[1]
    residual = (
        np.zeros((scan, capacity))
        if residuals is None
        else np.asarray(residuals, dtype=float).reshape(scan, capacity)
    )
    valid = np.ones((scan, capacity), dtype=bool)
    return build_state_product(
        states=array,
        residuals=residual,
        success=(
            np.ones((scan, capacity), dtype=bool)
            if success is None
            else np.asarray(success, dtype=bool).reshape(scan, capacity)
        ),
        valid_mask=valid,
        solution_count=np.full(scan, capacity, dtype=np.int64),
        parameters=dict(params or {}),
    )


def _context(model):
    from qphase_cam.postprocessor.base import CAMAnnotationContext

    return CAMAnnotationContext(model=model, backend=NumpyBackend())
