"""NumPy/CuPy consistency tests for CAM batched Newton."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _cupy_available() -> bool:
    try:
        import cupy as cp

        value = cp.asarray([1.0])
        return bool(cp.asnumpy(value)[0] == 1.0)
    except Exception:
        return False


@pytest.mark.skipif(not _cupy_available(), reason="CuPy/CUDA is unavailable")
@pytest.mark.parametrize("model_name", ["vdp_2mode", "kerr_3mode"])
def test_batched_newton_numpy_cupy_agree(model_name):
    from qphase.backend.cupy_backend import CuPyBackend
    from qphase.backend.numpy_backend import NumpyBackend
    from qphase_cam.solver.batched_newton import BatchedNewtonSolver
    from qphase_cam.solver.steady_state import SteadyStateSolver

    if model_name == "vdp_2mode":
        from models.vdp_2mode import VDP2ModeModel as Model

        base = {
            "omega_a": 0.0,
            "omega_b": 0.0,
            "gamma_a": 2.0,
            "gamma_b": 0.5,
            "Gamma": 0.0001,
            "g": 0.5,
        }
        varying, second = "omega_a", 0.001
        initial = [[20000.0, 20000.0j], [-20000.0j, 20000.0]]
        steady_method = "cholesky"
    else:
        from models.kerr_3mode import Kerr3ModeModel as Model

        base = {
            "omega_a": 0.0,
            "omega_b": -0.01,
            "omega_c": 0.01,
            "chi": 0.01,
            "gamma_a": 0.5,
            "gamma_b": 1.0,
            "gamma_c": 0.5,
            "g_ab": 0.5,
            "g_ac": 0.3,
        }
        varying, second = "omega_b", -0.011
        initial = np.eye(3)
        steady_method = "root"
    root = SteadyStateSolver(
        method=steady_method, initial_guess=initial, tolerance=1e-8
    ).solve(Model(**base), NumpyBackend()).solutions[0]
    assert root.success
    batched_params = dict(base)
    batched_params[varying] = [base[varying], second]
    options = {
        "initial_guesses": [root.state],
        "max_iterations": 20,
        "tolerance": 1e-7,
    }
    numpy_rows = BatchedNewtonSolver(**options).solve(
        Model(**batched_params), NumpyBackend()
    ).solutions
    cupy_rows = BatchedNewtonSolver(**options).solve(
        Model(**batched_params), CuPyBackend()
    ).solutions
    for numpy_row, cupy_row in zip(numpy_rows, cupy_rows, strict=True):
        assert numpy_row and cupy_row
        np.testing.assert_allclose(
            cupy_row[0].state, numpy_row[0].state, rtol=1e-8, atol=1e-9
        )
