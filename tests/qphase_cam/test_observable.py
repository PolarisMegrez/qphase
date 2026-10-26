"""The public diagonal Rayleigh readout: one mode's amplitude frequency.

The toy is two modes with a fixed Hermitian Hamiltonian, so ``Re (H R)[m,m] / R[m,m]``
is known in closed form and the denominator rules can be exercised on purpose: a node
of the mode amplitude (``R[1,1] = 0``) is a real state, a tiny amplitude is below the
floor, and ``denominator_floor = 0`` leaves the plain division alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qphase_cam.core.coordinates import vector_to_matrix
from qphase_cam.observable import (
    ChargeSector,
    DiagonalRayleighObservable,
    ObservableKind,
)

HAMILTONIAN = np.asarray([[1.0, 0.25], [0.25, -0.5]], dtype=complex)
#: ``R = diag(1, 0.4)`` with a real off-diagonal coupling in the ``R[0,1]`` slot.
STATE = np.asarray([1.0, 0.4, 0.3, 0.0])


class _Model:
    """A fixed Hermitian Hamiltonian, broadcast over a state stack."""

    supports_batched_matrices = True

    @staticmethod
    def cam_hamiltonian(matrix: Any, params: Any = None) -> np.ndarray:
        del params
        array = np.asarray(matrix, dtype=complex)
        return np.broadcast_to(HAMILTONIAN, array.shape).copy()


class _BrokenModel(_Model):
    """A Hamiltonian whose shape does not match the state matrix."""

    @staticmethod
    def cam_hamiltonian(matrix: Any, params: Any = None) -> np.ndarray:
        del matrix, params
        return np.eye(3, dtype=complex)


class _Adapter:
    supports_batched_states = True

    @staticmethod
    def state_matrix(vector: Any, params: Any = None) -> np.ndarray:
        del params
        return np.asarray(vector_to_matrix(np.asarray(vector, dtype=float), 2))


def _expected(state: np.ndarray, mode: int) -> float:
    matrix = np.asarray(vector_to_matrix(state, 2), dtype=complex)
    return float(np.real((HAMILTONIAN @ matrix)[mode, mode] / matrix[mode, mode]))


def test_the_readout_is_the_diagonal_matrix_element_ratio():
    observable = DiagonalRayleighObservable(_Model(), mode=1)
    assert observable.value(_Adapter(), STATE, {}) == _expected(STATE, 1)
    assert observable.value(_Adapter(), STATE, {}) != _expected(STATE, 0)
    # The trace readout is a different number: the two readouts are not the same object.
    trace = float(np.real(np.trace(HAMILTONIAN @ vector_to_matrix(STATE, 2))))
    assert observable.value(_Adapter(), STATE, {}) != trace


def test_the_batch_form_is_the_point_form_and_checks_its_shape():
    observable = DiagonalRayleighObservable(_Model(), mode=1)
    stack = np.asarray(
        [STATE, STATE + 0.1 * np.asarray([0.0, 0.0, 1.0, 0.0]), 0.5 * STATE]
    )
    batched = observable.values(_Adapter(), stack, {})
    assert batched.shape == (3,)
    np.testing.assert_allclose(
        batched, [_expected(row, 1) for row in stack], rtol=0.0, atol=0.0
    )
    with pytest.raises(ValueError, match="state stack must be"):
        observable.values(_Adapter(), np.asarray([1.0, 0.4, 0.3, 0.0]), {})


def test_the_identity_states_the_mode_and_the_mapping():
    default = DiagonalRayleighObservable(_Model(), mode=2)
    assert default.identity.kind is ObservableKind.CARRIER
    assert default.identity.sector is ChargeSector.CHARGE_ZERO
    assert default.identity.name == "mode_frequency_2"
    assert default.identity.measurement_mapping == "Re (H R)[2,2] / R[2,2]"
    named = DiagonalRayleighObservable(_Model(), mode=0, name="my_readout")
    assert named.identity.name == "my_readout"
    assert named.identity.measurement_mapping == "Re (H R)[0,0] / R[0,0]"


def test_a_node_of_the_mode_is_nan_not_an_error():
    node = np.asarray([1.0, 0.0, 0.5, 0.0])  # R = [[1, 0.5], [0.5, 0]]
    observable = DiagonalRayleighObservable(_Model(), mode=1)
    assert np.isnan(observable.value(_Adapter(), node, {}))
    batched = observable.values(_Adapter(), np.asarray([node, STATE]), {})
    assert np.isnan(batched[0])
    assert np.isfinite(batched[1])
    # A tiny (but non-zero) amplitude is below the default floor as well.
    tiny = np.asarray([1.0, 1.0e-15, 0.0, 0.0])
    assert np.isnan(observable.value(_Adapter(), tiny, {}))


def test_the_floor_is_configurable_and_zero_keeps_the_plain_division():
    node = np.asarray([1.0, 0.0, 0.5, 0.0])  # numerator 0.25*0.5, denominator 0
    floored = DiagonalRayleighObservable(_Model(), mode=1)
    assert np.isnan(floored.value(_Adapter(), node, {}))
    unfloored = DiagonalRayleighObservable(_Model(), mode=1, denominator_floor=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = unfloored.value(_Adapter(), node, {})
    assert not np.isnan(value)
    assert np.isinf(value) and value > 0.0
    # A small but non-zero amplitude divides normally when no floor is asked for.
    tiny = np.asarray([1.0, 1.0e-15, 0.0, 0.0])
    assert unfloored.value(_Adapter(), tiny, {}) == pytest.approx(-0.5, rel=1e-12)
    assert np.isnan(floored.value(_Adapter(), tiny, {}))


def test_the_gradient_is_the_central_difference_of_the_readout():
    observable = DiagonalRayleighObservable(_Model(), mode=1, relative_step=1.0e-6)
    record = observable.gradient(_Adapter(), STATE, {})
    assert record.method == "central_difference"
    assert len(record.gradient) == STATE.size
    assert record.metadata["relative_step"] == pytest.approx(1.0e-6)
    # Readout: (H10 R01 + H11 R11) / R11 with R01 the real coordinate 2.
    matrix = np.asarray(vector_to_matrix(STATE, 2), dtype=complex)
    r11, r01 = float(np.real(matrix[1, 1])), float(np.real(matrix[0, 1]))
    expected = np.zeros(STATE.size)
    expected[1] = -HAMILTONIAN[1, 0].real * r01 / r11**2
    expected[2] = HAMILTONIAN[1, 0].real / r11
    np.testing.assert_allclose(record.gradient, expected, rtol=1e-7, atol=1e-9)


def test_the_configuration_is_validated():
    with pytest.raises(ValueError, match="non-negative"):
        DiagonalRayleighObservable(_Model(), mode=-1)
    for floor in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="denominator_floor"):
            DiagonalRayleighObservable(_Model(), mode=0, denominator_floor=floor)
    with pytest.raises(ValueError, match="needs a model"):
        DiagonalRayleighObservable(None, mode=0)
    with pytest.raises(TypeError, match="cam_hamiltonian"):
        DiagonalRayleighObservable(object(), mode=0)


def test_the_shapes_are_validated():
    with pytest.raises(ValueError, match="Hamiltonian shape"):
        DiagonalRayleighObservable(_BrokenModel(), mode=0).value(_Adapter(), STATE, {})
    with pytest.raises(ValueError, match="out of range"):
        DiagonalRayleighObservable(_Model(), mode=5).value(_Adapter(), STATE, {})
    with pytest.raises(ValueError, match="out of range"):
        DiagonalRayleighObservable(_Model(), mode=5).values(
            _Adapter(), np.asarray([STATE, STATE]), {}
        )
    observable = DiagonalRayleighObservable(_Model(), mode=0)
    assert observable.value(_Adapter(), STATE, {}) == _expected(STATE, 0)
