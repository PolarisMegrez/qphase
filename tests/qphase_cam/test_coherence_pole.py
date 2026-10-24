"""Channel-visible CAM coherence-pole contracts."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.data import build_state_product
from qphase_cam.postprocessor.coherence_pole import (
    CoherencePoleConfig,
    CoherencePoleSpectrum,
    channel_coherence_poles,
)


def test_channel_coherence_poles_select_slowest_visible_pole() -> None:
    hamiltonian = np.diag([1.0 - 0.2j, 2.0 - 0.01j])
    state = np.diag([4.0, 1.0]).astype(complex)
    mode_0 = np.diag([1.0, 0.0]).astype(complex)
    trace = np.eye(2, dtype=complex)
    coherent = np.outer([1.0, 1.0], [1.0, 1.0]) / 2.0

    result = channel_coherence_poles(
        hamiltonian,
        state,
        np.asarray([mode_0, trace, coherent]),
        relative_residue_floor=0.1,
    )

    np.testing.assert_allclose(result["frequency"], [1.0, 2.0, 2.0])
    np.testing.assert_allclose(result["decay_rate"], [0.2, 0.01, 0.01])
    np.testing.assert_array_equal(result["valid"], [True, True, True])


def test_channel_coherence_poles_apply_relative_visibility_floor() -> None:
    hamiltonian = np.diag([1.0 - 0.2j, 2.0 - 0.01j])
    state = np.diag([4.0, 1.0]).astype(complex)
    trace = np.eye(2, dtype=complex)

    result = channel_coherence_poles(
        hamiltonian,
        state,
        np.asarray([trace]),
        relative_residue_floor=0.3,
    )

    np.testing.assert_allclose(result["frequency"], [1.0])
    np.testing.assert_allclose(result["relative_residue"], [1.0])


def test_coherence_pole_config_accepts_complex_channels() -> None:
    config = CoherencePoleConfig(
        modes=[0],
        channels={"quadrature": ["1+0j", "0+1j"]},
        include_trace=False,
    )

    assert config.channels["quadrature"] == [1.0 + 0.0j, 1.0j]
    assert config.model_dump()["channels"] == {"quadrature": ["(1+0j)", "1j"]}


def test_coherence_pole_config_rejects_duplicate_measurement_names() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        CoherencePoleConfig(modes=[0], channels={"mode_0": [1.0, 0.0]})


def test_coherence_pole_postprocessor_records_measurement_axis() -> None:
    class DiagonalModel:
        n_modes = 2

        @staticmethod
        def cam_hamiltonian(state, params):
            del state, params
            return np.diag([1.0 - 0.2j, 2.0 - 0.01j])

    product = _state_product(np.asarray([np.diag([4.0, 1.0])], dtype=complex))
    postprocessor = CoherencePoleSpectrum(
        modes=[0],
        channels={"balanced": [1.0, 1.0]},
    )

    output = postprocessor.process(product, _context(DiagonalModel()))

    frequency = np.asarray(output.handle("coherence_pole_frequency").materialize())
    # One candidate row, one entry per configured measurement.
    assert frequency.shape == (1, 3)
    assert postprocessor.result_metadata["measurement_names"] == [
        "mode_0",
        "trace",
        "balanced",
    ]
    assert postprocessor.result_metadata["frequency_orientation"] == (
        "phase_decreasing"
    )


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
