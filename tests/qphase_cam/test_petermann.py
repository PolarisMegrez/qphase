"""Petermann-spectrum and solution-view contracts."""

from __future__ import annotations

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
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


def test_petermann_postprocessor_emits_aligned_spectra():
    model = Kerr2ModeModel(
        omega_a=0.0,
        omega_b=-0.1,
        chi=0.01,
        gamma_a=0.5,
        gamma_b=1.0,
        g=0.5,
    )
    result = CAMResult(
        states=np.asarray([[[2.0, 0.1j], [-0.1j, 1.0]]], dtype=complex),
        residuals=np.zeros(1),
        success=np.ones(1, dtype=bool),
        valid_mask=np.ones(1, dtype=bool),
        solution_count=np.asarray(1),
        params=model.params,
    )
    output = PetermannSpectrum(chunk_size=1).process(result, model, NumpyBackend())
    assert output["hamiltonian_eigenvalues"].shape == (1, 2)
    assert output["bogoliubov_eigenvalues"].shape == (1, 4)
    assert output["monodromy_eigenvalues"].shape == (1, 4)
    assert np.isfinite(output["hamiltonian_petermann_factors"]).all()


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
