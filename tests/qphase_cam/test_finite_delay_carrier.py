import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.data import build_state_product
from qphase_cam.postprocessor.finite_delay_carrier import (
    CAMFiniteDelayCarrier,
    finite_delay_carrier_from_poles,
)


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

    product = _state_product(np.asarray([np.diag([4.0, 1.0])], dtype=complex))
    postprocessor = CAMFiniteDelayCarrier(
        modes=[0], include_trace=True, detector_rates=[0.1, 1.0]
    )

    output = postprocessor.process(product, _context(DiagonalModel()))

    instantaneous = np.asarray(
        output.handle("finite_delay_carrier_instantaneous_frequency").materialize()
    )
    rayleigh = np.asarray(
        output.handle("finite_delay_carrier_rayleigh_frequency").materialize()
    )
    residual = np.asarray(
        output.handle(
            "finite_delay_carrier_instantaneous_rayleigh_residual"
        ).materialize()
    )
    carrier = np.asarray(output.handle("finite_delay_carrier_frequency").materialize())
    assert carrier.shape == (1, 2, 2)
    np.testing.assert_allclose(instantaneous, rayleigh, atol=1e-14)
    np.testing.assert_allclose(residual, 0.0, atol=1e-14)
    assert postprocessor.result_metadata["measurement_names"] == ["mode_0", "trace"]


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
