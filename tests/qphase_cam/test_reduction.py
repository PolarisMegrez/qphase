"""Exact condensed scalar-reduction tests."""

from __future__ import annotations

import numpy as np
import sympy as sp
from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.core.reduction import CondensedScalarReduction

from models.kerr_3mode import Kerr3ModeModel
from models.vdp_2mode import VDP2ModeModel


def test_condensed_jets_match_explicit_vdp_reduced_dynamics():
    model = VDP2ModeModel(
        omega_a=0.1,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.8,
        Gamma=0.0001,
        g=0.5,
    )
    adapter = FPGenDynamicsAdapter.from_model(model)
    candidate = adapter.dynamics.find_linear_reductions(
        retained_dimension=1
    )[0]
    plan = adapter.dynamics.linear_reduce(candidate=candidate)
    materialized = plan.materialize()
    condensed = CondensedScalarReduction(
        plan,
        order=4,
        control_names=("omega_a", "gamma_b", "gamma_a"),
        base_params=model.params,
    )
    point = np.asarray([12000.0, 0.15, 0.9, 2.1])
    params = {
        **model.params,
        "omega_a": point[1],
        "gamma_b": point[2],
        "gamma_a": point[3],
    }
    q = plan.retained_symbols[0]
    arguments = (q, *(item.symbol for item in adapter.dynamics.parameter_spec))
    expected_function = sp.lambdify(
        arguments,
        sp.Matrix(
            [
                sp.diff(materialized.reduced_residual[0], q, derivative)
                for derivative in range(5)
            ]
        ),
        modules="numpy",
    )
    expected = np.asarray(
        expected_function(
            point[0],
            *(params[item.name] for item in adapter.dynamics.parameter_spec),
        ),
        dtype=float,
    ).reshape(-1)
    np.testing.assert_allclose(
        condensed.diagnostics(point).reduced_coefficients,
        expected,
        rtol=2e-10,
        atol=2e-10,
    )


def test_kerr_three_mode_condensed_reconstruction_solves_eliminated_block():
    model = Kerr3ModeModel(
        omega_a=0.0,
        omega_b=-0.1,
        omega_c=0.2,
        chi=0.01,
        gamma_a=0.5,
        gamma_b=1.0,
        gamma_c=0.4,
        g_ab=0.5,
        g_ac=0.3,
    )
    adapter = FPGenDynamicsAdapter.from_model(model)
    candidate = adapter.dynamics.find_linear_reductions(
        retained_dimension=1
    )[0]
    reduction = CondensedScalarReduction(
        adapter.dynamics.linear_reduce(candidate=candidate),
        order=3,
        control_names=("omega_b", "gamma_b"),
        base_params=model.params,
    )
    point = np.asarray([1.0, -0.1, 1.0])
    state = reduction.reconstruct(point)
    residual = adapter.rhs(state, model.params)
    np.testing.assert_allclose(
        residual[list(candidate.eliminated_equations)], 0.0, atol=3e-14
    )
    assert reduction.method == "reduced_condensed"
    assert reduction.diagnostics(point).condition_number < 100.0
