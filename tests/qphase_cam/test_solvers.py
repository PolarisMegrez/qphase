"""CAM solver capability and batched execution tests."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.engine import Engine
from qphase_cam.errors import JacobianUnavailableError
from qphase_cam.postprocessor.frequency import RayleighFrequency
from qphase_cam.solver.batched_newton import BatchedNewtonSolver
from qphase_cam.solver.continuation import ContinuationSolver
from qphase_cam.solver.guess_bounds import GuessBoundsConfig
from qphase_cam.solver.multistability import MultistabilitySolver
from qphase_cam.solver.steady_state import SteadyStateSolver

from models.vdp_2mode import VDP2ModeModel

VDP_PARAMS = {
    "omega_a": 0.0,
    "omega_b": 0.0,
    "gamma_a": 2.0,
    "gamma_b": 0.5,
    "Gamma": 0.0001,
    "g": 0.5,
}
VDP_GUESS = [[20000.0, 20000.0j], [-20000.0j, 20000.0]]


def _vdp_root() -> np.ndarray:
    output = SteadyStateSolver(
        method="cholesky", initial_guess=VDP_GUESS, tolerance=1e-8
    ).solve(VDP2ModeModel(**VDP_PARAMS), NumpyBackend())
    assert output.solutions[0].success
    return output.solutions[0].state


def test_jacobian_required_solvers_reject_missing_capability(no_jacobian_model):
    no_jacobian_model.params["unused"] = 0.0
    with pytest.raises(JacobianUnavailableError):
        BatchedNewtonSolver(max_iterations=1).solve(
            no_jacobian_model, NumpyBackend()
        )
    with pytest.raises(JacobianUnavailableError):
        ContinuationSolver(
            parameter="unused",
            start=0.0,
            stop=1.0,
            initial_guess=np.eye(2),
        ).solve(no_jacobian_model, NumpyBackend())


def test_batched_engine_uses_each_grid_points_parameters():
    root = _vdp_root()
    params = dict(VDP_PARAMS)
    params["omega_a"] = [0.0, 0.001]
    result = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": VDP2ModeModel(**params),
            "cam_solver": BatchedNewtonSolver(
                initial_guesses=[root], max_iterations=20, tolerance=1e-8
            ),
            "cam_postprocessor": RayleighFrequency(),
        }
    ).run()
    assert result.states.shape == (2, 4, 2, 2)
    assert result.solution_count.tolist() == [1, 1]
    assert result.params_at((1, 0))["omega_a"] == 0.001
    assert np.isfinite(result.postprocess["rayleigh_frequency"][:, 0]).all()


def test_multistability_accepts_explicit_and_automatic_bounds():
    root = _vdp_root()
    explicit = {
        "diag_lower": [-10000.0, -10000.0],
        "diag_upper": [30000.0, 30000.0],
        "offdiag_scale": 10000.0,
    }
    solver = MultistabilitySolver(
        n_guesses=1,
        initial_guesses=[root],
        guess_bounds=explicit,
        tolerance=1e-8,
        residual_tolerance=1e-7,
    )
    output = solver.solve(VDP2ModeModel(**VDP_PARAMS), NumpyBackend())
    assert len(output.solutions) == 1
    assert output.solutions[0].success
    inferred = MultistabilitySolver(
        n_guesses=1,
        initial_guesses=[root],
        guess_bounds="auto",
        tolerance=1e-8,
        residual_tolerance=1e-7,
    ).solve(VDP2ModeModel(**VDP_PARAMS), NumpyBackend())
    assert len(inferred.solutions) == 1


def test_guess_bounds_schema_rejects_non_positive_scale():
    with pytest.raises(ValidationError):
        GuessBoundsConfig(
            diag_lower=[-1.0, -1.0],
            diag_upper=[1.0, 1.0],
            offdiag_scale=0.0,
        )


def test_continuation_has_fixed_slots_without_branch_id():
    output = ContinuationSolver(
        parameter="gamma_b",
        start=0.5,
        stop=0.50001,
        step=0.0002,
        max_steps=3,
        tolerance=1e-7,
        initial_guess=VDP_GUESS,
    ).solve(VDP2ModeModel(**VDP_PARAMS), NumpyBackend())
    assert len(output.solutions) >= 2
    assert "gamma_b" in output.axes
    assert "branch_id" not in output.axes
    assert output.metadata["continuation"] is True


@pytest.mark.slow
def test_multistability_tile_processes_preserve_point_order():
    root = _vdp_root()
    params = dict(VDP_PARAMS)
    params["omega_a"] = [0.0, 0.001]
    output = MultistabilitySolver(
        n_guesses=1,
        initial_guesses=[root],
        tile_workers=2,
        tile_size=1,
        tolerance=1e-8,
        residual_tolerance=1e-7,
    ).solve(VDP2ModeModel(**params), NumpyBackend())
    assert [len(row) for row in output.solutions] == [1, 1]
    assert output.metadata["tile_workers"] == 2
