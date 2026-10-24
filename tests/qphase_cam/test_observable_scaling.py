"""Phase 10F2b: readout-resolved scaling analysis and its annotation.

Two layers are checked separately:

* the numeric service, against the frozen values of the reference script's own
  ``_prefix_upper`` / ``_fixed_exponent_fit_upper`` / ``_free_power_fit`` on the
  deterministic input ``x = geomspace(1e-4, 1e-1, 9)``,
  ``y = 2.5 x^(1/3) - 0.8 x^(2/3)``;
* the postprocessor, end to end on a typed ``state_scan`` and on a candidate scan,
  including the shift handling, the outer extrapolation and an artifact restore.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.data import load_bundle
from qphase_cam.data import build_candidate_product, build_state_product, cam_bundle
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
    RayleighObservable,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.observable_scaling import ObservableScaling
from qphase_cam.scaling_analysis import (
    fixed_exponent_tolerance_upper,
    local_log_exponents,
    log_linear_scaling_fit,
    predict_puiseux,
    puiseux_outer_extrapolation,
    series_identity,
    tolerance_prefix_upper,
    window_log_scaling_fits,
)
from qphase_cam.solver.scaling import free_exponent_fit

X = np.geomspace(1.0e-4, 1.0e-1, 9)
Y = 2.5 * X ** (1.0 / 3.0) - 0.8 * X ** (2.0 / 3.0)
EPSILON = X
READOUT_IDENTITY = ObservableIdentity(
    kind=ObservableKind.CARRIER,
    sector=ChargeSector.CHARGE_ZERO,
    name="nu",
    measurement_mapping="1.234 + 2.5 x^(1/3) - 0.8 x^(2/3)",
)
CRITICAL = 1.234


# --- numeric service: reference parity -------------------------------------


def test_tolerance_prefix_upper_matches_the_reference():
    accepted = np.asarray([True] * 7 + [False, True])
    assert tolerance_prefix_upper(X, accepted) == pytest.approx(
        0.01778279410038923, rel=1e-15
    )
    assert tolerance_prefix_upper(X, np.ones(9, dtype=bool)) == pytest.approx(
        0.1, rel=1e-15
    )
    assert tolerance_prefix_upper(X, np.asarray([False] + [True] * 8)) is None


def test_fixed_exponent_tolerance_upper_matches_the_reference():
    assert fixed_exponent_tolerance_upper(X, Y, 1.0 / 3.0, 0.05) == pytest.approx(
        0.007498942093324558, rel=1e-15
    )
    assert fixed_exponent_tolerance_upper(X, Y, 1.0 / 3.0, 0.02) is None
    assert fixed_exponent_tolerance_upper(X, Y, 1.0 / 3.0, 0.10) == pytest.approx(
        0.1, rel=1e-15
    )
    # The exponent is the caller's: the same series with another exponent does not
    # borrow the script's one-third.
    assert fixed_exponent_tolerance_upper(X, Y, 0.5, 0.05) is None


def test_log_linear_scaling_fit_matches_the_reference():
    wide = log_linear_scaling_fit(X, Y, upper=1.0e-1)
    assert wide.exponent == pytest.approx(0.31350464701565334, rel=1e-12)
    assert wide.log_rmse == pytest.approx(0.014966403281018954, rel=1e-12)
    assert wide.point_count == 9
    narrow = log_linear_scaling_fit(X, Y, upper=1.0e-2)
    assert narrow.exponent == pytest.approx(0.3220172710628815, rel=1e-12)
    assert narrow.log_rmse == pytest.approx(0.0035780305703133728, rel=1e-12)
    assert narrow.point_count == 6
    windows = window_log_scaling_fits(X, Y, (1.0e-2, 1.0e-1))
    assert [fit.point_count for fit in windows] == [6, 9]
    assert windows[0].exponent == pytest.approx(narrow.exponent, rel=1e-12)


def test_local_log_exponents_match_the_reference_stencil():
    exponents = local_log_exponents(X, Y)
    assert exponents.size == X.size
    np.testing.assert_allclose(
        exponents[:3],
        [0.3284916683505831, 0.3264983916429506, 0.32415420968745046],
        rtol=1e-12,
    )
    assert exponents[0] > exponents[-1]


def test_the_series_validation_is_strict():
    with pytest.raises(ValueError, match="strictly increasing"):
        log_linear_scaling_fit(X[::-1], Y[::-1])
    with pytest.raises(ValueError, match="strictly positive"):
        log_linear_scaling_fit(X, -Y)
    with pytest.raises(ValueError, match="finite"):
        log_linear_scaling_fit(np.where(X > 1e-2, np.nan, X), Y)
    with pytest.raises(ValueError, match="at least 3 points"):
        local_log_exponents(X[:2], Y[:2])
    with pytest.raises(ValueError, match="needs two"):
        log_linear_scaling_fit(X, Y, upper=1.0e-6)
    with pytest.raises(ValueError, match="non-negative"):
        fixed_exponent_tolerance_upper(X, Y, 1.0 / 3.0, -0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        fixed_exponent_tolerance_upper(X, Y, 0.0, 0.05)


def test_the_puiseux_prediction_and_outer_extrapolation_reuse_the_10d3_fit():
    fit = free_exponent_fit(X, Y, order=2, bounds=(0.25, 0.45), calculate_loo=False)
    prediction = predict_puiseux(fit, X)
    np.testing.assert_allclose(prediction, Y, rtol=1e-11)
    outer = puiseux_outer_extrapolation(
        X, Y, inner_max=1.0e-2, order=2, bounds=(0.25, 0.45)
    )
    assert outer.inner_point_count == 6
    assert outer.outer_point_count == 3
    assert outer.exponent == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert outer.relative_rmse < 1e-9
    assert outer.max_abs_relative_error < 1e-9
    assert outer.at_exponent_bound is False
    with pytest.raises(ValueError, match="no outer point"):
        puiseux_outer_extrapolation(X, Y, inner_max=0.5, order=2, bounds=(0.25, 0.45))


def test_the_series_identity_is_content_addressed():
    identity = FunctionalObservable(
        identity=READOUT_IDENTITY, value_fn=lambda adapter, state, params: 0.0
    )
    first = series_identity(identity, parameter="epsilon", x=X, y=Y)
    again = series_identity(identity, parameter="epsilon", x=X, y=Y)
    other = series_identity(identity, parameter="epsilon", x=X, y=Y * 1.01)
    renamed = series_identity(identity, parameter="omega", x=X, y=Y)
    assert first == again
    assert first != other
    assert first != renamed
    assert first.startswith("cam:scaling:")
    assert len(first.split(":")[-1]) == 16
    assert "/" not in first and "\\" not in first


# --- postprocessor ---------------------------------------------------------


class _Adapter:
    """One-mode adapter: the state matrix is the diagonal of the state vector."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        values = np.asarray(vector, dtype=float).reshape(-1)
        return np.diag(values.astype(complex))


class _Model:
    """One-mode model whose Hamiltonian is the state matrix itself."""

    name = "scaling_toy"
    n_modes = 1
    params = {"epsilon": 0.0}

    @staticmethod
    def cam_hamiltonian(state, params=None):
        del params
        return np.asarray(state, dtype=complex)


def _patch_adapter(monkeypatch):
    monkeypatch.setattr(
        "qphase_cam.postprocessor.observable_scaling.QPSGenDynamicsAdapter.from_model",
        lambda model: _Adapter(),
    )


def _scan_product(parameters, values, *, success=None):
    states = np.zeros((len(values), 1, 1, 1), dtype=complex)
    states[:, 0, 0, 0] = np.asarray(values, dtype=complex)
    count = len(values)
    return build_state_product(
        states=states,
        residuals=np.zeros((count, 1)),
        success=(
            np.ones((count, 1), dtype=bool)
            if success is None
            else np.asarray(success, dtype=bool).reshape(count, 1)
        ),
        valid_mask=np.ones((count, 1), dtype=bool),
        solution_count=np.ones(count, dtype=np.int64),
        parameters={"epsilon": np.asarray(parameters, dtype=float)},
    )


def _context(product, role="state_scan", metadata=None):
    return CAMAnnotationContext(
        model=_Model(),
        metadata=dict(metadata or {}),
        products={role: product},
        source_role=role,
    )


def _scaling(**kwargs):
    defaults = {
        "parameter": "epsilon",
        "critical_value": CRITICAL,
        "order": 2,
        "leading_exponent": 1.0 / 3.0,
        "window_maxima": (1.0e-2, 1.0e-1),
        "inner_max": 1.0e-2,
    }
    defaults.update(kwargs)
    return ObservableScaling(**defaults)


def _rayleigh_law():
    """Return the carrier readout values ``Re Tr(H R)/Tr(R)`` for ``H = R``."""
    return np.asarray([CRITICAL + value for value in Y])


def test_the_postprocessor_resolves_the_rayleigh_scaling(monkeypatch):
    _patch_adapter(monkeypatch)
    product = _scan_product(EPSILON, _rayleigh_law())
    annotation = _scaling().process(product, _context(product))
    assert annotation.attributes["annotation_kind"] == "observable_scaling"
    assert annotation.attributes["target_kind"] == "case"
    assert list(np.asarray(annotation.handle("status").materialize())) == ["valid"]
    identity = str(np.asarray(annotation.handle("target_identity").materialize())[0])
    assert identity.startswith("cam:scaling:")
    assert np.asarray(annotation.handle("exponent").materialize())[0] == pytest.approx(
        1.0 / 3.0, rel=1e-9
    )
    assert np.asarray(annotation.handle("at_exponent_bound").materialize())[0] == 0
    assert np.asarray(annotation.handle("condition").materialize())[0] > 1.0
    assert np.asarray(annotation.handle("loo_rmse").materialize())[0] > 0.0
    # The leading law is the caller's exponent with the first-point amplitude,
    # which is labelled as such rather than presented as an analytic derivative.
    assert np.asarray(annotation.handle("leading_coefficient").materialize())[
        0
    ] == pytest.approx(Y[0] / X[0] ** (1.0 / 3.0), rel=1e-12)
    assert (
        str(
            np.asarray(annotation.handle("leading_coefficient_source").materialize())[0]
        )
        == "first_point_ratio"
    )
    # Both tolerance prefixes reproduce the reference script on this series.
    assert np.asarray(annotation.handle("fixed_exponent_prefix_upper").materialize())[
        0
    ] == pytest.approx(0.007498942093324558, rel=1e-12)
    assert np.asarray(annotation.handle("leading_prefix_upper").materialize())[
        0
    ] == pytest.approx(0.0031622776601683794, rel=1e-12)
    # The per-point exponents and the windows keep the series' resolution.
    points = np.asarray(annotation.handle("point_x").materialize())
    assert points.shape == (1, 9)
    np.testing.assert_allclose(points[0], X)
    local = np.asarray(annotation.handle("local_log_exponent").materialize())[0]
    assert local[0] > local[-1]
    windows = np.asarray(annotation.handle("window_exponent").materialize())[0]
    np.testing.assert_allclose(windows, [0.3220172710628815, 0.31350464701565334])
    assert np.asarray(
        annotation.handle("window_point_count").materialize()
    ).tolist() == [[6, 9]]
    # The generic Puiseux outer prediction sees the law it fitted.
    assert np.asarray(annotation.handle("inner_point_count").materialize())[0] == 6
    assert np.asarray(annotation.handle("outer_point_count").materialize())[0] == 3
    assert np.asarray(annotation.handle("outer_exponent").materialize())[0] == (
        pytest.approx(1.0 / 3.0, rel=1e-9)
    )
    assert np.asarray(annotation.handle("outer_relative_rmse").materialize())[0] < 1e-9
    metadata = str(np.asarray(annotation.handle("method_metadata").materialize())[0])
    assert "observable_scaling" not in metadata
    assert "free_exponent_fit" in metadata
    assert "rayleigh_frequency" in metadata


def test_a_matrix_element_readout_needs_no_model_special_case(monkeypatch):
    _patch_adapter(monkeypatch)
    # The state carries the coordinate itself, so the element readout is 3x.
    values = 3.0 * EPSILON

    def element(adapter, state, params):
        return float(np.real(adapter.state_matrix(state, params)[0, 0]))

    readout = FunctionalObservable(
        identity=ObservableIdentity(
            kind=ObservableKind.STATE,
            sector=ChargeSector.CHARGE_ZERO,
            name="Re_R_aa",
            measurement_mapping="Re R[0,0]",
        ),
        value_fn=element,
        gradient_fn=lambda adapter, state, params: [1.0],
    )
    product = _scan_product(EPSILON, values)
    annotation = _scaling(
        absolute_shift=False,
        critical_value=None,
        bounds=(0.8, 1.2),
        leading_exponent=1.0,
        observable=readout,
    ).process(product, _context(product))
    assert list(np.asarray(annotation.handle("status").materialize())) == ["valid"]
    assert np.asarray(annotation.handle("exponent").materialize())[0] == pytest.approx(
        1.0, rel=1e-9
    )
    assert np.asarray(annotation.handle("leading_coefficient").materialize())[
        0
    ] == pytest.approx(3.0, rel=1e-9)
    assert str(np.asarray(annotation.handle("charge_sector").materialize())[0]) == (
        "charge_zero"
    )
    assert str(
        np.asarray(annotation.handle("critical_value_source").materialize())[0]
    ) == ("not_used")


def test_the_zero_point_supplies_the_anchor_when_it_exists(monkeypatch):
    _patch_adapter(monkeypatch)
    parameters = np.concatenate(([0.0], EPSILON))
    values = np.concatenate(([CRITICAL], _rayleigh_law()))
    product = _scan_product(parameters, values)
    annotation = _scaling(
        critical_value=None, window_maxima=(), inner_max=None
    ).process(product, _context(product))
    assert list(np.asarray(annotation.handle("status").materialize())) == ["valid"]
    assert str(
        np.asarray(annotation.handle("critical_value_source").materialize())[0]
    ) == ("zero_point")
    assert np.asarray(annotation.handle("critical_value").materialize())[0] == (
        pytest.approx(CRITICAL, rel=1e-15)
    )
    # The x=0 point carries no scaling information and is dropped.
    assert np.asarray(annotation.handle("dropped_point_count").materialize())[0] == 1
    assert np.asarray(annotation.handle("point_count").materialize())[0] == 9


def test_a_missing_anchor_is_a_configuration_error(monkeypatch):
    _patch_adapter(monkeypatch)
    product = _scan_product(EPSILON, _rayleigh_law())
    with pytest.raises(ValueError, match="critical_value is required"):
        _scaling(critical_value=None).process(product, _context(product))


def test_a_candidate_scan_is_resolved_case_by_case(monkeypatch):
    _patch_adapter(monkeypatch)
    values = _rayleigh_law()
    count = len(values)
    product = build_candidate_product(
        states=np.asarray(values, dtype=float).reshape(count, 1, 1).astype(complex),
        state_vectors=np.asarray(values, dtype=float).reshape(count, 1),
        control_values=np.empty((count, 0)),
        control_names=(),
        full_residual_norm=np.zeros(count),
        search_residual_norm=np.zeros(count),
        success=np.ones(count, dtype=bool),
        status=["verified"] * count,
        method=["test"] * count,
        verification_digits=np.full(count, 50, dtype=np.int64),
        verification_status=["verified"] * count,
        candidate_identity=[f"cam:candidate:test{index}" for index in range(count)],
        candidate_offsets=np.arange(count + 1, dtype=np.int64),
        case_params={"epsilon": np.asarray(EPSILON)},
        case_shape=(count,),
        case_metadata=[{"case_identity": f"case:{index}"} for index in range(count)],
    )
    annotation = _scaling().process(product, _context(product, role="candidates"))
    assert list(np.asarray(annotation.handle("status").materialize())) == ["valid"]
    assert np.asarray(annotation.handle("point_count").materialize())[0] == 9
    assert np.asarray(annotation.handle("exponent").materialize())[0] == pytest.approx(
        1.0 / 3.0, rel=1e-9
    )


def test_negative_results_are_statuses_without_a_fabricated_fit(monkeypatch):
    _patch_adapter(monkeypatch)
    values = _rayleigh_law()
    failed = np.ones(9, dtype=bool)
    failed[4] = False
    product = _scan_product(EPSILON, values, success=failed)
    annotation = _scaling().process(product, _context(product))
    assert list(np.asarray(annotation.handle("status").materialize())) == [
        "not_converged"
    ]
    assert np.isnan(np.asarray(annotation.handle("exponent").materialize())[0])
    assert "did not converge" in str(
        np.asarray(annotation.handle("reason").materialize())[0]
    )

    short = _scan_product(EPSILON[:4], values[:4])
    annotation = _scaling(critical_value=CRITICAL).process(short, _context(short))
    assert list(np.asarray(annotation.handle("status").materialize())) == [
        "unidentifiable"
    ]
    assert "usable points" in str(
        np.asarray(annotation.handle("reason").materialize())[0]
    )

    constant = _scan_product(EPSILON, np.full(9, CRITICAL))
    annotation = _scaling(
        critical_value=CRITICAL, window_maxima=(), inner_max=None
    ).process(constant, _context(constant))
    assert list(np.asarray(annotation.handle("status").materialize())) == [
        "unidentifiable"
    ]
    assert "strictly positive" in str(
        np.asarray(annotation.handle("reason").materialize())[0]
    )


def test_structural_problems_raise(monkeypatch):
    _patch_adapter(monkeypatch)
    values = _rayleigh_law()
    product = _scan_product(EPSILON, values)
    with pytest.raises(ValueError, match="no 'omega' coordinate"):
        _scaling(parameter="omega").process(product, _context(product))
    other = _scan_product([1.0, -1.0, 2.0], [CRITICAL + 1, CRITICAL + 2, CRITICAL + 3])
    with pytest.raises(ValueError, match="duplicate"):
        _scaling(critical_value=CRITICAL).process(other, _context(other))


def test_the_annotation_round_trips_through_an_artifact(monkeypatch, tmp_path):
    _patch_adapter(monkeypatch)
    product = _scan_product(EPSILON, _rayleigh_law())
    annotation = _scaling().process(product, _context(product))
    cam_bundle({"observable_scaling": annotation}).save_dataset(
        tmp_path, layout="single", shard_target_bytes=4096
    )
    restored = load_bundle(tmp_path).require("observable_scaling")
    assert list(np.asarray(restored.handle("status").materialize())) == ["valid"]
    assert np.asarray(restored.handle("exponent").materialize())[0] == pytest.approx(
        1.0 / 3.0, rel=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(restored.handle("point_x").materialize())[0], X, rtol=1e-15
    )
    assert restored.attributes["target_kind"] == "case"


def test_the_default_readout_is_the_rayleigh_frequency(monkeypatch):
    _patch_adapter(monkeypatch)
    product = _scan_product(EPSILON, _rayleigh_law())
    annotation = _scaling().process(product, _context(product))
    assert str(np.asarray(annotation.handle("observable_name").materialize())[0]) == (
        "rayleigh_frequency"
    )
    assert isinstance(RayleighObservable(_Model()), object)
    assert CAMOutcomeStatus.VALID.value == "valid"
