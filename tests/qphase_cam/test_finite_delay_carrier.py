import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.postprocessor.finite_delay_carrier import (
    CAMFiniteDelayCarrier,
    finite_delay_carrier_from_poles,
)
from qphase_cam.result import CAMResult


def test_cam_finite_delay_instantaneous_limit_is_rayleigh():
    eigenvalues = np.asarray([1.0 - 0.2j, 2.0 - 0.01j])
    residues = np.asarray([4.0, 1.0])
    result = finite_delay_carrier_from_poles(
        eigenvalues, residues, np.asarray([0.01, 1.0, 1e6])
    )

    assert result["instantaneous_frequency"] == pytest.approx(1.2, abs=1e-14)
    assert result["frequency"][-1] == pytest.approx(1.2, abs=5e-8)
    assert abs(result["frequency"][0] - 1.2) > 1e-3


def test_cam_finite_delay_postprocessor_matches_generalized_rayleigh():
    class DiagonalModel:
        n_modes = 2

        @staticmethod
        def cam_hamiltonian(state, params):
            del state, params
            return np.diag([1.0 - 0.2j, 2.0 - 0.01j])

    result = CAMResult(
        states=np.asarray([np.diag([4.0, 1.0])], dtype=complex),
        residuals=np.zeros(1),
        success=np.ones(1, dtype=bool),
        valid_mask=np.ones(1, dtype=bool),
        solution_count=np.asarray(1),
        params={},
    )
    postprocessor = CAMFiniteDelayCarrier(
        modes=[0], include_trace=True, detector_rates=[0.1, 1.0]
    )

    output = postprocessor.process(result, DiagonalModel(), NumpyBackend())

    np.testing.assert_allclose(
        output["finite_delay_carrier_instantaneous_frequency"],
        output["finite_delay_carrier_rayleigh_frequency"],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        output["finite_delay_carrier_instantaneous_rayleigh_residual"],
        0.0,
        atol=1e-14,
    )
    assert postprocessor.result_metadata["measurement_names"] == ["mode_0", "trace"]
