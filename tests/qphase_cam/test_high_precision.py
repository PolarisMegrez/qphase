"""The public high-precision solve and candidate-state services.

The local-response postprocessor and the retained scripts must agree on which state
a branch starts from and what one Newton continuation returns, so these tests pin
the public services against the postprocessor they were extracted from: same state,
same residual, same convergence, same restored working precision — plus the input
contracts the extraction adds (grouped prediction arguments, failure reporting).
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from qphase.data import load_bundle
from qphase_cam.bifurcation_result import CAMBifurcationResult
from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.data import build_candidate_product
from qphase_cam.postprocessor.candidates import (
    candidate_parameters,
    candidate_state,
    candidate_view,
)
from qphase_cam.postprocessor.local_response import (
    LocalResponseValidation,
    LocalResponseValidationConfig,
)
from qphase_cam.solver.high_precision import (
    HighPrecisionNewtonConfig,
    HighPrecisionSolution,
    solve_high_precision,
)

from models.kerr_full_3mode import KerrFull3ModeModel

#: The production candidate whose branch the historical scripts continue.
CANDIDATE_JOB = (
    "reports/phase10_verification/sessions_production/2026/10"
    "/2026-10-25T03-03-41_89db7e/kerr_full_3mode_d0_omega_b_gamma_b_gamma_c"
)
PARAMS = {
    "omega_a": 0.0,
    "omega_b": 0.5,
    "omega_c": -0.5,
    "chi": 0.001,
    "gamma_a": 3.0,
    "gamma_b": 0.7993197086840077,
    "gamma_c": 0.9572027428720037,
    "g_ab": 4.0,
    "g_ac": 0.1,
    "g_bc": 2.0,
}


def _candidate_product():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    bundle = load_bundle(root / CANDIDATE_JOB)
    return bundle.products["candidates"].materialize()


def _adapter():
    from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter

    return QPSGenDynamicsAdapter.from_model(KerrFull3ModeModel(**PARAMS))


# --- the public solve matches the postprocessor -----------------------------


def test_the_public_solve_matches_the_local_response_newton():
    product = _candidate_product()
    view = candidate_view(product)
    state = candidate_state(view, 0)
    adapter = _adapter()
    params = dict(PARAMS)
    config = HighPrecisionNewtonConfig(
        precision_digits=80, residual_tolerance=1e-40, max_iterations=40
    )
    plugin = LocalResponseValidation(
        LocalResponseValidationConfig(
            precision_digits=80, residual_tolerance=1e-40, max_iterations=40
        )
    )
    initial = state.vector + 1.0e-6 * np.linspace(-1.0, 1.0, state.vector.size)
    solution = solve_high_precision(adapter, initial, params, config=config)
    expected_state, expected_residual, expected_converged = plugin._newton(
        adapter, initial, params, critical_decimals=None
    )
    np.testing.assert_array_equal(solution.state, np.asarray(expected_state))
    assert solution.residual == expected_residual
    assert solution.converged == expected_converged


def test_the_predicted_initial_state_uses_the_verified_decimals():
    product = _candidate_product()
    view = candidate_view(product)
    state = candidate_state(view, 0)
    assert state.has_decimals, "the production candidate carries verified decimals"
    adapter = _adapter()
    params = dict(PARAMS)
    coefficient = np.linspace(1.0e-3, 2.0e-3, len(state.decimals or ()))
    epsilon, exponent = 1.0e-6, 1.0 / 3.0
    solution = solve_high_precision(
        adapter,
        state.chart_vector,
        params,
        critical_decimals=state.decimals,
        leading_coefficient=coefficient,
        epsilon_magnitude=epsilon,
        exponent=exponent,
    )
    # The same prediction through the postprocessor: identical numbers.
    plugin = LocalResponseValidation(
        LocalResponseValidationConfig(precision_digits=80, residual_tolerance=1e-30)
    )
    expected = plugin._newton(
        adapter,
        state.chart_vector,
        params,
        critical_decimals=state.decimals,
        leading_coefficient=coefficient,
        epsilon_magnitude=epsilon,
        exponent=exponent,
    )
    np.testing.assert_array_equal(solution.state, np.asarray(expected[0]))
    assert solution.residual == expected[1]
    assert solution.converged == expected[2]
    # The decimals differ from the chart vector: the solve really started from them.
    assert not np.array_equal(
        np.asarray([float(item) for item in state.decimals or ()]), state.chart_vector
    )


def test_the_working_precision_is_restored():
    adapter = _adapter()
    product = _candidate_product()
    state = candidate_state(candidate_view(product), 0)
    before = mp.mp.dps
    solve_high_precision(
        adapter,
        state.vector,
        dict(PARAMS),
        config=HighPrecisionNewtonConfig(
            precision_digits=60, residual_tolerance=1e-30, max_iterations=5
        ),
    )
    assert mp.mp.dps == before
    with pytest.raises(ValueError, match="together"):
        solve_high_precision(
            adapter,
            state.vector,
            dict(PARAMS),
            critical_decimals=state.decimals,
            epsilon_magnitude=1.0e-6,
        )
    assert mp.mp.dps == before


def test_the_prediction_arguments_must_appear_together():
    adapter = _adapter()
    product = _candidate_product()
    state = candidate_state(candidate_view(product), 0)
    cases = (
        {"critical_decimals": state.decimals},
        {"leading_coefficient": np.zeros_like(state.vector)},
        {"epsilon_magnitude": 1.0e-6},
        {"exponent": 0.5},
        {
            "critical_decimals": state.decimals,
            "leading_coefficient": np.zeros_like(state.vector),
        },
        {
            "critical_decimals": state.decimals,
            "leading_coefficient": np.zeros_like(state.vector),
            "epsilon_magnitude": 1.0e-6,
        },
    )
    for kwargs in cases:
        with pytest.raises(ValueError, match="together"):
            solve_high_precision(adapter, state.vector, dict(PARAMS), **kwargs)


def test_a_solve_that_cannot_converge_reports_it():
    adapter = _adapter()
    product = _candidate_product()
    state = candidate_state(candidate_view(product), 0)
    solution = solve_high_precision(
        adapter,
        state.vector + 1.0e3,
        dict(PARAMS),
        config=HighPrecisionNewtonConfig(
            precision_digits=40, residual_tolerance=1e-60, max_iterations=1
        ),
    )
    assert isinstance(solution, HighPrecisionSolution)
    assert not solution.converged
    assert solution.iterations <= 1
    assert np.isfinite(solution.state).all()


# --- the candidate-state service -------------------------------------------


def test_the_candidate_state_reads_the_typed_view_and_the_legacy_result():
    product = _candidate_product()
    view = candidate_view(product)
    typed = candidate_state(view, 0)
    legacy = CAMBifurcationResult(
        states=np.asarray([view.rows[0].state]),
        state_vectors=np.asarray([view.rows[0].state_vector]),
        control_values=np.asarray([[0.0]]),
        control_names=("gamma_b",),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=np.asarray(["verified"]),
        method=np.asarray(["test"]),
        verification_digits=np.zeros(1, dtype=np.int64),
        verification_status=np.asarray(["not_run"]),
        diagnostics={
            "verified_full_state_decimal_values": np.asarray(
                [list(typed.decimals or ())], dtype=object
            )
        },
    )
    historical = candidate_state(legacy, 0)
    np.testing.assert_array_equal(historical.chart_vector, typed.chart_vector)
    assert historical.decimals == typed.decimals
    np.testing.assert_array_equal(historical.vector, typed.vector)
    with pytest.raises(IndexError, match="out of bounds"):
        candidate_state(legacy, 1)
    with pytest.raises(TypeError, match="CandidateView or a CAMBifurcationResult"):
        candidate_state("not a result", 0)


def test_a_product_without_verified_decimals_falls_back_to_the_chart_vector():
    signature = {
        "state_order": 3,
        "perturbation_order": 1,
        "coupling_state_order": 0,
        "exponent_numerator": 1,
        "exponent_denominator": 3,
        "exponent": 1.0 / 3.0,
        "sublinear": True,
        "branches": (),
    }
    product = build_candidate_product(
        states=np.asarray([np.eye(2)], dtype=complex),
        state_vectors=np.asarray([matrix_to_vector(np.eye(2))]),
        control_values=np.full((1, 1), -0.5),
        control_names=("gamma",),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=["verified"],
        method=["test"],
        verification_digits=np.zeros(1, dtype=np.int64),
        verification_status=["not_run"],
        classification=[{"scaling_signatures": [signature]}],
        base_audit=[{}],
        candidate_identity=["cam:candidate:nodecimals"],
        candidate_offsets=np.asarray([0, 1]),
        case_shape=(1,),
    )
    view = candidate_view(product)
    state = candidate_state(view, 0)
    assert state.decimals is None
    np.testing.assert_array_equal(state.vector, view.rows[0].state_vector)
    np.testing.assert_array_equal(state.vector, state.chart_vector)


def test_the_production_candidate_resolves_its_own_case_parameters():
    """Candidate 0 of the accepted job belongs to case 0 and keeps its controls."""
    product = _candidate_product()
    view = candidate_view(product)
    state = candidate_state(view, 0)
    params = candidate_parameters(view, 0, KerrFull3ModeModel(**PARAMS))
    assert params["omega_b"] == pytest.approx(0.5)
    assert params["gamma_b"] == pytest.approx(0.7993197086840077)
    assert params["gamma_c"] == pytest.approx(0.9572027428720037)
    assert state.chart_vector.size == 9
