"""CAM solver, engine, capacity, postprocessing, and persistence tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.engine import Engine
from qphase_cam.errors import SolutionCapacityError
from qphase_cam.postprocessor.frequency import (
    HamiltonianSpectrum,
    RayleighFrequency,
)
from qphase_cam.postprocessor.jacobian import JacobianSpectrum
from qphase_cam.postprocessor.physicality import Physicality
from qphase_cam.result import CAMResult
from qphase_cam.solver.steady_state import SteadyStateSolver
from qphase_cam.state import CAMSolution, CAMSolverOutput

from models.vdp_2mode import VDP2ModeModel


def test_no_jacobian_model_solves_without_jacobian(no_jacobian_model):
    result = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": SteadyStateSolver(method="root"),
        }
    ).run()
    assert result.success[0]
    np.testing.assert_allclose(result.states[0], np.eye(2), atol=1e-8)


def test_vdp_engine_postprocess_and_round_trip(tmp_path):
    model = VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )
    result = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": model,
            "cam_solver": SteadyStateSolver(
                method="cholesky",
                initial_guess=[[20000.0, 20000.0j], [-20000.0j, 20000.0]],
                tolerance=1e-8,
            ),
            "cam_postprocessor": {
                "rayleigh": RayleighFrequency(),
                "spectrum": HamiltonianSpectrum(),
                "jacobian": JacobianSpectrum(),
                "physicality": Physicality(),
            },
        }
    ).run()
    assert result.solution_count == 1
    assert result.success[0]
    assert result.postprocess["jacobian_source"][0] == "analytic"
    assert result.postprocess["is_hermitian"][0]
    target = tmp_path / "cam_result.npz"
    result.save(target)
    loaded = CAMResult.load(target)
    np.testing.assert_allclose(loaded.states, result.states, equal_nan=True)
    assert target.with_suffix(".csv").exists()


def test_rayleigh_zero_trace_has_no_division_warning(no_jacobian_model):
    result = CAMResult(
        states=np.zeros((1, 2, 2), dtype=complex),
        residuals=np.zeros(1),
        success=np.ones(1, dtype=bool),
        valid_mask=np.ones(1, dtype=bool),
        solution_count=np.array(1),
        params={},
    )
    with warnings.catch_warnings(record=True) as caught:
        output = RayleighFrequency().process(result, no_jacobian_model, NumpyBackend())
    assert np.isnan(output["rayleigh_frequency"][0])
    assert not caught


def test_finite_difference_jacobian_records_source_and_step(no_jacobian_model):
    result = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": SteadyStateSolver(method="root"),
            "cam_postprocessor": JacobianSpectrum(
                allow_finite_difference=True,
                finite_difference_epsilon=2e-6,
            ),
        }
    ).run()
    assert result.postprocess["jacobian_source"][0] == "finite_difference"
    metadata = result.meta["postprocessor_metadata"]["jacobian_spectrum"]
    assert metadata["jacobian_sources"] == ["finite_difference"]
    assert metadata["finite_difference_epsilon"] == 2e-6


def test_capacity_overflow_is_an_error(no_jacobian_model):
    class OverflowSolver:
        name = "overflow"

        def solve(self, model, backend):
            del model, backend
            solutions = [
                CAMSolution(np.eye(2) * value, 0.0, True, "test")
                for value in (1.0, 2.0)
            ]
            return CAMSolverOutput(solutions)

    engine = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": OverflowSolver(),
        }
    )
    with pytest.raises(SolutionCapacityError):
        engine.run()
