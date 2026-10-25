"""The public high-precision solve and candidate-state services.

The local-response postprocessor and the retained scripts must agree on which state
a branch starts from and what one Newton continuation returns, so these tests pin the
public services against the postprocessor they were extracted from: same state, same
residual, same convergence, same restored working precision — plus the input
contracts the extraction adds (grouped prediction arguments, failure reporting).

The fixture is a real candidate of the tracked two-mode fold model, searched once per
module through the public solver and the typed candidate builder. It is built from
the model and the solver only, so the tests carry no dependency on any local evidence
directory: the verified decimal expansion the fold model's search records is what the
decimal-initial tests use.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.bifurcation_result import CAMBifurcationResult
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
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
from qphase_cam.solver.bifurcation import ControlRange
from qphase_cam.solver.execution import candidate_product_from_output
from qphase_cam.solver.high_precision import (
    HighPrecisionNewtonConfig,
    HighPrecisionSolution,
    solve_high_precision,
)

from .test_bifurcation import TwoModeFoldModel, _solver


@pytest.fixture(scope="module")
def fold_candidate() -> dict:
    """Run one real fold search and return the candidate product and its model."""
    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    product = candidate_product_from_output(model, output)
    view = candidate_view(product)
    return {
        "model": model,
        "product": product,
        "view": view,
        "params": dict(model.params),
        "adapter": QPSGenDynamicsAdapter.from_model(model),
    }


def _config() -> HighPrecisionNewtonConfig:
    return HighPrecisionNewtonConfig(
        precision_digits=80, residual_tolerance=1.0e-40, max_iterations=40
    )


def _postprocessor(config: HighPrecisionNewtonConfig | None = None):
    resolved = config or _config()
    return LocalResponseValidation(
        LocalResponseValidationConfig(
            precision_digits=resolved.precision_digits,
            residual_tolerance=resolved.residual_tolerance,
            max_iterations=resolved.max_iterations,
        )
    )


# --- the fixture is a real verified candidate -------------------------------


def test_the_fold_fixture_carries_a_verified_decimal_expansion(fold_candidate):
    view = fold_candidate["view"]
    state = candidate_state(view, 0)
    assert state.has_decimals
    assert len(state.decimals or ()) == view.rows[0].state_vector.size
    # The expansion keeps more digits than the chart vector's doubles, so the same
    # state is described more precisely; ``float`` of it is the chart double again.
    long_items = [
        item
        for item, value in zip(state.decimals or (), state.chart_vector, strict=True)
        if float(value) != 0.0
    ]
    assert long_items and all(len(item) > 20 for item in long_items)
    # Read at working precision, the expansion and the double are different numbers;
    # at the default 15 digits they would look equal.
    with mp.workdps(60):
        assert mp.mpf(state.decimals[0]) != mp.mpf(str(state.chart_vector[0]))
    np.testing.assert_array_equal(
        np.asarray([float(item) for item in state.decimals or ()]),
        state.chart_vector,
    )
    np.testing.assert_array_equal(
        state.vector, np.asarray([float(item) for item in state.decimals])
    )


def test_the_fold_candidate_resolves_its_own_parameters(fold_candidate):
    view = fold_candidate["view"]
    model = fold_candidate["model"]
    row = view.rows[0]
    params = candidate_parameters(view, 0, model)
    # The row's control wins over the model value it perturbs; the untouched model
    # parameters survive, and nothing from another case leaks in.
    assert set(row.controls) == {"gamma"}
    assert params["gamma"] == pytest.approx(row.controls["gamma"])
    assert params["gamma"] != pytest.approx(float(model.params["gamma"]))
    assert params["Gamma"] == pytest.approx(float(model.params["Gamma"]))
    assert params["kappa"] == pytest.approx(float(model.params["kappa"]))
    assert set(params) == set(model.params)


# --- the public solve matches the postprocessor -----------------------------


def test_the_public_solve_matches_the_local_response_newton(fold_candidate):
    view = fold_candidate["view"]
    state = candidate_state(view, 0)
    adapter = fold_candidate["adapter"]
    params = candidate_parameters(view, 0, fold_candidate["model"])
    initial = state.vector + 1.0e-6 * np.linspace(-1.0, 1.0, state.vector.size)
    solution = solve_high_precision(adapter, initial, params, config=_config())
    expected_state, expected_residual, expected_converged = _postprocessor()._newton(
        adapter, initial, params, critical_decimals=None
    )
    np.testing.assert_array_equal(solution.state, np.asarray(expected_state))
    assert solution.residual == expected_residual
    assert solution.converged == expected_converged


def test_the_predicted_initial_state_uses_the_verified_decimals(fold_candidate):
    view = fold_candidate["view"]
    state = candidate_state(view, 0)
    adapter = fold_candidate["adapter"]
    params = candidate_parameters(view, 0, fold_candidate["model"])
    coefficient = np.linspace(1.0e-3, 2.0e-3, len(state.decimals or ()))
    epsilon, exponent = 1.0e-6, 1.0 / 3.0
    solution = solve_high_precision(
        adapter,
        state.chart_vector,
        params,
        config=_config(),
        critical_decimals=state.decimals,
        leading_coefficient=coefficient,
        epsilon_magnitude=epsilon,
        exponent=exponent,
    )
    expected = _postprocessor()._newton(
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
    # The prediction differs from the plain initial state, so the decimals and the
    # coefficient really entered the solve.
    plain = solve_high_precision(adapter, state.chart_vector, params, config=_config())
    assert not np.array_equal(solution.state, plain.state)


# --- input contracts --------------------------------------------------------


def test_the_working_precision_is_restored(fold_candidate):
    adapter = fold_candidate["adapter"]
    view = fold_candidate["view"]
    state = candidate_state(view, 0)
    params = candidate_parameters(view, 0, fold_candidate["model"])
    before = mp.mp.dps
    solve_high_precision(
        adapter,
        state.vector,
        params,
        config=HighPrecisionNewtonConfig(
            precision_digits=60, residual_tolerance=1e-30, max_iterations=5
        ),
    )
    assert mp.mp.dps == before
    with pytest.raises(ValueError, match="together"):
        solve_high_precision(
            adapter,
            state.vector,
            params,
            critical_decimals=state.decimals,
            epsilon_magnitude=1.0e-6,
        )
    assert mp.mp.dps == before


def test_the_prediction_arguments_must_appear_together(fold_candidate):
    adapter = fold_candidate["adapter"]
    view = fold_candidate["view"]
    state = candidate_state(view, 0)
    params = candidate_parameters(view, 0, fold_candidate["model"])
    zero = np.zeros_like(state.vector)
    cases = (
        {"critical_decimals": state.decimals},
        {"leading_coefficient": zero},
        {"epsilon_magnitude": 1.0e-6},
        {"exponent": 0.5},
        {"critical_decimals": state.decimals, "leading_coefficient": zero},
        {
            "critical_decimals": state.decimals,
            "leading_coefficient": zero,
            "epsilon_magnitude": 1.0e-6,
        },
        {
            "leading_coefficient": zero,
            "epsilon_magnitude": 1.0e-6,
            "exponent": 0.5,
        },
    )
    for kwargs in cases:
        with pytest.raises(ValueError, match="together"):
            solve_high_precision(adapter, state.vector, params, **kwargs)


def test_a_solve_that_cannot_converge_says_so(fold_candidate):
    adapter = fold_candidate["adapter"]
    view = fold_candidate["view"]
    state = candidate_state(view, 0)
    params = candidate_parameters(view, 0, fold_candidate["model"])
    solution = solve_high_precision(
        adapter,
        state.vector + 1.0e3,
        params,
        config=HighPrecisionNewtonConfig(
            precision_digits=40, residual_tolerance=1e-60, max_iterations=1
        ),
    )
    assert isinstance(solution, HighPrecisionSolution)
    assert not solution.converged
    assert solution.iterations <= 1
    assert np.isfinite(solution.state).all()


# --- the candidate-state service -------------------------------------------


def test_the_candidate_state_reads_the_typed_view_and_the_legacy_result(
    fold_candidate,
):
    view = fold_candidate["view"]
    typed = candidate_state(view, 0)
    # The same candidate as a legacy result: one case, one candidate per row.
    legacy = CAMBifurcationResult(
        states=np.asarray([view.rows[0].state]),
        state_vectors=np.asarray([view.rows[0].state_vector]),
        control_values=np.asarray([list(view.rows[0].controls.values())]),
        control_names=tuple(view.rows[0].controls),
        full_residual_norm=np.asarray([view.rows[0].full_residual_norm]),
        search_residual_norm=np.asarray([view.rows[0].search_residual_norm]),
        success=np.ones(1, dtype=bool),
        status=np.asarray([view.rows[0].status]),
        method=np.asarray([view.rows[0].method]),
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
    with pytest.raises(IndexError, match="out of bounds"):
        candidate_state(view, len(view.rows))
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
        state_vectors=np.asarray([[1.0, 0.0, 0.0, 1.0]]),
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
