"""Phase 10G2d1c2: the public scaling response and series contract.

The fixture is a two-coordinate toy whose "matrix element" is complex, so a response
that takes a modulus or a norm is genuinely a response and not a cast: nothing here
goes through a scalar observable.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.data import load_bundle
from qphase_cam.data import cam_bundle
from qphase_cam.observable import (
    ChargeSector,
    ObservableIdentity,
    ObservableKind,
)
from qphase_cam.postprocessor.observable_scaling import ObservableScalingConfig
from qphase_cam.scaling_series import (
    FunctionalScalingResponse,
    ScalingResponse,
    ScalingSeries,
    evaluate_scaling_series,
    scaling_series,
)


def _identity(name: str, mapping: str) -> ObservableIdentity:
    return ObservableIdentity(
        kind=ObservableKind.STATE,
        sector=ChargeSector.CHARGE_ZERO,
        name=name,
        measurement_mapping=mapping,
    )


class _ToyAdapter:
    """Return ``R = [[1, x], [-x, 2]]``: a matrix element that moves with mu."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[1.0, value], [-value, 2.0]], dtype=complex)


def _element(row: int, column: int):
    def value(adapter, state, params):
        del params
        matrix = adapter.state_matrix(state, None)
        return float(np.real(matrix[row][column]))

    return value


def _modulus(row: int, column: int):
    """Return the complex displacement's magnitude, formed by the response."""

    def value(adapter, state, params):
        del params
        matrix = adapter.state_matrix(state, None)
        return abs(complex(matrix[row][column]) - complex(matrix[column][row]))

    return value


def _frobenius(adapter, state, params):
    del params
    matrix = adapter.state_matrix(state, None)
    return float(np.linalg.norm(matrix - np.tril(matrix).T * 0 - np.eye(2) * matrix))


def _series(points=12, critical=0.5, exponent=0.5):
    x = np.geomspace(1.0e-9, 1.0e-1, points)
    return scaling_series(
        parameter="mu",
        critical_parameter=critical,
        points=[
            {
                "parameter": critical + float(value),
                "x": float(value),
                "response": 1.7 * float(value) ** exponent,
                "converged": True,
            }
            for value in x
        ],
        reference_value=2.5,
        source="unit-test",
        method={"study": "toy"},
    ), x


def test_a_response_is_a_positive_magnitude_with_its_own_identity():
    response = FunctionalScalingResponse(
        identity=_identity("abs_element", "|R_01 - R_10|"),
        response_fn=_modulus(0, 1),
        mapping="|R_01 - R_10|",
    )
    assert isinstance(response, ScalingResponse)
    adapter = _ToyAdapter()
    # |R_01 - R_10| with R_01 = x and R_10 = -x: the response is 2|x|.
    assert response.response(adapter, [0.3], {}) == pytest.approx(0.6)
    assert response.response(adapter, [-0.4], {}) == pytest.approx(0.8)
    method = response.method()
    assert method["mapping"] == "|R_01 - R_10|"
    assert method["identity"]["name"] == "abs_element"
    # A norm is a response too: it is already a real, positive number.
    norm = FunctionalScalingResponse(
        identity=_identity("frobenius", "||R - R(critical)||_F"),
        response_fn=_frobenius,
    )
    assert norm.response(adapter, [0.4], {}) >= 0.0


def test_the_series_keeps_the_exact_x_and_the_physical_parameter():
    series, x = _series()
    assert np.array_equal(np.asarray(series.x, dtype=float), x)
    assert series.parameter_values[0] == pytest.approx(0.5 + x[0])
    # The abscissa is the configured epsilon: the largest point shows that the
    # floating-point difference is a different number.
    assert series.x[-1] == 0.1
    assert series.parameter_values[-1] - 0.5 != series.x[-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        ScalingSeries(
            parameter="mu",
            parameter_values=(0.6, 0.5),
            x=(1.0e-3, 1.0e-4),
            response=(1.0, 2.0),
            converged=(True, True),
            critical_parameter=0.5,
        ).validated()


def test_a_series_refuses_bad_and_incomplete_input():
    good = dict(
        parameter="mu",
        parameter_values=(0.51, 0.52),
        x=(1.0e-2, 2.0e-2),
        response=(1.0, 2.0),
        converged=(True, True),
        critical_parameter=0.5,
    )
    with pytest.raises(ValueError, match="did not converge"):
        ScalingSeries(**{**good, "converged": (True, False)}).validated()
    with pytest.raises(ValueError, match="strictly positive"):
        ScalingSeries(**{**good, "response": (1.0, 0.0)}).validated()
    with pytest.raises(ValueError, match="strictly positive"):
        ScalingSeries(**{**good, "x": (0.0, 2.0e-2)}).validated()
    with pytest.raises(ValueError, match="entries for"):
        ScalingSeries(**{**good, "x": (1.0e-2,)}).validated()
    with pytest.raises(ValueError, match="at least one point"):
        ScalingSeries(
            parameter="mu",
            parameter_values=(),
            x=(),
            response=(),
            converged=(),
            critical_parameter=0.5,
        ).validated()


def test_the_annotation_is_the_shared_builder_and_records_the_series():
    series, x = _series()
    response = FunctionalScalingResponse(
        identity=_identity("abs_element", "|R_01 - R_10|"),
        response_fn=_modulus(0, 1),
        mapping="|R_01 - R_10|",
    )
    annotation = evaluate_scaling_series(response, series)
    assert annotation.attributes["annotation_kind"] == "observable_scaling"
    point_x = np.asarray(annotation.handle("point_x").materialize())[0]
    point_parameter = np.asarray(annotation.handle("point_parameter").materialize())[0]
    point_response = np.asarray(annotation.handle("point_response").materialize())[0]
    np.testing.assert_array_equal(point_x, x)
    np.testing.assert_allclose(point_parameter, np.asarray(series.parameter_values))
    np.testing.assert_allclose(point_response, np.asarray(series.response))
    # The local exponent of a clean power law is the law's exponent.
    exponent = np.asarray(annotation.handle("local_log_exponent").materialize())[0]
    np.testing.assert_allclose(exponent[1:-1], 0.5, rtol=1.0e-9)
    import json

    metadata = json.loads(
        str(np.asarray(annotation.handle("method_metadata").materialize())[0])
    )
    assert metadata["source"] == "unit-test"
    assert metadata["reference_value"] == 2.5
    assert metadata["reference_value_role"].startswith("reported")
    assert metadata["series_identity"].startswith("cam:scaling:")
    assert metadata["response_contract"]["mapping"] == "|R_01 - R_10|"
    assert metadata["critical_parameter"] == 0.5
    assert metadata["leading_error_normalization"] == "prediction"
    # The reported reference value is not subtracted from the response.
    assert not np.allclose(point_response, np.abs(point_response - 2.5))


def test_the_leading_normalisation_is_explicit():
    series, _ = _series()
    response = FunctionalScalingResponse(
        identity=_identity("power", "x**0.5"), response_fn=_modulus(0, 1)
    )
    prediction = evaluate_scaling_series(
        response,
        series,
        config=ObservableScalingConfig(
            parameter="mu",
            critical_parameter=0.5,
            critical_value=2.5,
            absolute_shift=False,
            leading_exponent=0.5,
            leading_coefficient=1.7,
        ),
    )
    response_normalised = evaluate_scaling_series(
        response,
        series,
        config=ObservableScalingConfig(
            parameter="mu",
            critical_parameter=0.5,
            critical_value=2.5,
            absolute_shift=False,
            leading_exponent=0.5,
            leading_coefficient=1.7,
            leading_error_normalization="response",
        ),
    )
    import json

    def metadata(annotation):
        return json.loads(
            str(np.asarray(annotation.handle("method_metadata").materialize())[0])
        )

    assert metadata(prediction)["leading_law"]["error_normalization"] == "prediction"
    assert (
        metadata(response_normalised)["leading_law"]["error_normalization"]
        == "response"
    )
    # The exact law makes both residuals zero; a deliberately wrong coefficient does
    # not, and the two definitions then differ.
    wrong = ObservableScalingConfig(
        parameter="mu",
        critical_parameter=0.5,
        critical_value=2.5,
        absolute_shift=False,
        leading_exponent=0.5,
        leading_coefficient=1.0,
        leading_coefficient_source="leading_state_response",
    )
    strict = evaluate_scaling_series(response, series, config=wrong)
    loose = evaluate_scaling_series(
        response,
        series,
        config=wrong.model_copy(
            update={"leading_error_normalization": "response"}
        ),
    )
    assert float(
        np.asarray(strict.handle("leading_relative_max").materialize())[0]
    ) != pytest.approx(
        float(np.asarray(loose.handle("leading_relative_max").materialize())[0])
    )
    # A supplied coefficient says where it came from.
    assert metadata(strict)["leading_law"]["coefficient_source"] == (
        "leading_state_response"
    )


def test_a_series_refuses_a_config_that_would_shift_it_again():
    series, _ = _series()
    response = FunctionalScalingResponse(
        identity=_identity("power", "x**0.5"), response_fn=_modulus(0, 1)
    )
    with pytest.raises(ValueError, match="absolute_shift=False"):
        evaluate_scaling_series(
            response,
            series,
            config=ObservableScalingConfig(
                parameter="mu", critical_parameter=0.5, absolute_shift=True
            ),
        )
    with pytest.raises(TypeError, match="ObservableIdentity"):
        evaluate_scaling_series(object(), series)


def test_the_series_and_its_product_round_trip_through_an_artifact(tmp_path):
    series, x = _series()
    response = FunctionalScalingResponse(
        identity=_identity("abs_element", "|R_01 - R_10|"),
        response_fn=_modulus(0, 1),
    )
    annotation = evaluate_scaling_series(response, series)
    bundle = cam_bundle({"observable_scaling": annotation})
    bundle.save_dataset(tmp_path, layout="single", shard_target_bytes=1 << 20)
    restored = load_bundle(tmp_path)
    product = restored.require("observable_scaling")
    np.testing.assert_array_equal(
        np.asarray(product.handle("point_x").materialize())[0], x
    )
    assert restored.parents == ()
    assert restored.metadata["artifact_id"]
    # The series identity survives the round trip, so the stored product can be
    # compared with a recomputation.
    import json

    metadata = json.loads(
        str(np.asarray(product.handle("method_metadata").materialize())[0])
    )
    assert metadata["series_identity"] == series.identity(response)
    assert product.attributes["annotation_kind"] == "observable_scaling"


def test_a_point_that_does_not_state_its_convergence_is_refused():
    """A missing convergence flag is an error, never a silent success."""
    x = np.geomspace(1.0e-9, 1.0e-1, 4)
    points = [
        {"parameter": 0.5 + float(value), "x": float(value), "response": float(value)}
        for value in x
    ]
    with pytest.raises(ValueError, match="does not state whether it converged"):
        scaling_series(parameter="mu", critical_parameter=0.5, points=points)
    # Stating it, including a false one, is accepted here and refused by the validator.
    stated = [dict(point, converged=index != 2) for index, point in enumerate(points)]
    with pytest.raises(ValueError, match="did not converge"):
        scaling_series(parameter="mu", critical_parameter=0.5, points=stated)


def test_the_series_control_name_is_the_annotation_identity():
    """A non-default control keeps one identity in the product and in its metadata."""
    series, x = _series()
    series = series.model_copy(update={"parameter": "omega_b"})
    response = FunctionalScalingResponse(
        identity=_identity("abs_element", "|R_01 - R_10|"),
        response_fn=_modulus(0, 1),
    )
    annotation = evaluate_scaling_series(response, series)
    import json

    metadata = json.loads(
        str(np.asarray(annotation.handle("method_metadata").materialize())[0])
    )
    target = str(np.asarray(annotation.handle("target_identity").materialize())[0])
    assert metadata["series_identity"] == target
    assert metadata["parameter"] == "omega_b"
    # The physical parameter values are untouched and the abscissa is still the exact
    # configured epsilon.
    np.testing.assert_array_equal(
        np.asarray(annotation.handle("point_x").materialize())[0], x
    )
    with pytest.raises(ValueError, match="one series has one identity"):
        evaluate_scaling_series(
            response,
            series,
            config=ObservableScalingConfig(
                parameter="omega_c",
                critical_parameter=0.5,
                critical_value=2.5,
                absolute_shift=False,
            ),
        )
