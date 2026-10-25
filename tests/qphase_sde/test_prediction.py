"""Phase 10G2c: the SDE prediction-validation service.

Every test here goes through the real artifact store: the prediction product and the
observation bundle are saved and restored before they are compared, and the result is
persisted and restored again.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.data import (
    ArtifactManifest,
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    StatisticsDataset,
    VariableSchema,
    load_bundle,
)
from qphase_cam.data.evidence import SourceArtifactRef
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.prediction import CAMPrediction, build_prediction_product
from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.prediction import (
    PREDICTION_VALIDATION_SCHEMA,
    PredictionComparison,
    validate_prediction,
)
from qphase_sde.result import SDEDataBundle

PREDICTION_ARTIFACT = "0123456789abcdef0123456789abcdef"
PREDICTION_ANCESTOR = "11111111111111111111111111111111"
OBSERVATION_ARTIFACT = "fedcba9876543210fedcba9876543210"
OBSERVATION_ANCESTOR = "22222222222222222222222222222222"


# --- fixtures --------------------------------------------------------------


def _prediction(
    prediction_id: str = "cam:prediction:one", **overrides
) -> CAMPrediction:
    payload = {
        "prediction_id": prediction_id,
        "target_identity": "cam:state:alpha",
        "observable_name": "rayleigh",
        "observable_kind": "state",
        "charge_sector": "charge_zero",
        "measurement_mapping": "R00",
        "approximation": "nonlinear_manifold",
        "order": 7,
        "status": CAMOutcomeStatus.VALID,
        "completeness": "complete",
        "source_artifacts": (SourceArtifactRef(artifact_id="cam-state-1"),),
        "expected_frequency_response_full": 2.0,
        "allan_intensity_reconstructed": 0.5,
        "full_information_rate": 8.0,
    }
    payload.update(overrides)
    return CAMPrediction(**payload)


def _saved_prediction(
    tmp_path: Path,
    predictions: list[CAMPrediction] | None = None,
    *,
    artifact_id: str = PREDICTION_ARTIFACT,
    parents: tuple[str, ...] = (PREDICTION_ANCESTOR,),
) -> tuple[object, ArtifactManifest]:
    """Save a prediction product as an artifact and restore it."""
    from qphase.data import save_products

    directory = tmp_path / f"prediction-{artifact_id}"
    directory.mkdir(parents=True)
    product = build_prediction_product(
        predictions or [_prediction()], provenance={"artifact_id": artifact_id}
    )
    save_products(
        directory,
        {"prediction": product},
        provenance={"artifact_id": artifact_id, "job_name": "predict"},
        artifact_id=artifact_id,
        parents=list(parents),
        layout="single",
    )
    manifest = ArtifactManifest.read(directory)
    return load_bundle(directory).require("prediction"), manifest


def _spectrum(
    values,
    *,
    dims: tuple[str, ...] = ("scan", "channel"),
    field: str = "response",
) -> StatisticsDataset:
    extents = {dim: 1 for dim in dims}
    array = np.asarray(values, dtype=float)
    for dim, extent in zip(dims, array.shape, strict=True):
        extents[dim] = int(extent)
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            AxisSchema(name=dim, role=AxisRole.INDEX, size=size)
            for dim, size in extents.items()
        ],
        variables=[
            VariableSchema(name=field, dtype="float64", value_domain="real", dims=dims)
        ],
    )
    return StatisticsDataset.from_arrays(schema, {field: array}, owner="engine.sde")


def _saved_observation(
    tmp_path: Path,
    products=None,
    *,
    scan_axes: dict | None = None,
    scan_shape: tuple[int, ...] = (2,),
    artifact_id: str = OBSERVATION_ARTIFACT,
    parents: tuple[str, ...] = (OBSERVATION_ANCESTOR,),
    layout: str = "single",
) -> SDEDataBundle:
    """Save an observation bundle as an artifact and restore it."""
    directory = tmp_path / f"observation-{artifact_id}-{layout}"
    directory.mkdir(parents=True)
    bundle = SDEDataBundle(
        products
        if products is not None
        else {"spectrum": _spectrum([[2.1, 9.0], [3.0, 4.0]])},
        SDEProvenance(dt=1.0),
        scan_axes=scan_axes if scan_axes is not None else {"epsilon": [0.1, 0.2]},
        scan_shape=scan_shape,
        parents=parents,
        artifact_id=artifact_id,
    )
    if layout == "single":
        bundle.save(directory)
    else:
        bundle.save_dataset(directory, layout="sharded", shard_target_bytes=256)
    return load_bundle(directory)


def _comparison(**overrides) -> PredictionComparison:
    payload = {
        "prediction_id": "cam:prediction:one",
        "metric": "expected_frequency_response_full",
        "observed_product": "spectrum",
        "observed_field": "response",
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.05,
        "scan_index": 0,
        "selection": {"channel": 0},
    }
    payload.update(overrides)
    return PredictionComparison(**payload)


def _rows(bundle: SDEDataBundle, name: str = "prediction_validation") -> dict:
    product = bundle.products[name]
    return {
        variable.name: np.asarray(product.handle(variable.name).materialize())
        for variable in product.variables
    }


# --- the validation table ---------------------------------------------------


def test_validate_prediction_compares_and_reports_both_verdicts(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    result = validate_prediction(
        prediction,
        manifest,
        observation,
        [
            _comparison(),
            _comparison(
                metric="allan_intensity_reconstructed",
                scan_index=1,
                relative_tolerance=0.01,
            ),
        ],
    )
    table = result.products["prediction_validation"]
    assert table.kind is DataKind.STATISTICS
    assert table.axis("validation").role is AxisRole.INDEX
    assert table.attributes["schema"] == PREDICTION_VALIDATION_SCHEMA
    assert table.attributes["product"] == "prediction_validation"
    assert table.attributes["validation_count"] == 2
    assert table.attributes["valid_count"] == 1
    assert table.attributes["threshold_rejected_count"] == 1
    assert table.attributes["observation_artifact_id"] == OBSERVATION_ARTIFACT
    assert table.attributes["prediction_artifact_id"] == PREDICTION_ARTIFACT

    assert all(np.dtype(variable.dtype).kind != "O" for variable in table.variables)
    assert all(variable.dims == ("validation",) for variable in table.variables)
    rows = _rows(result)
    assert list(rows["metric"]) == [
        "expected_frequency_response_full",
        "allan_intensity_reconstructed",
    ]
    np.testing.assert_allclose(rows["expected"], [2.0, 0.5])
    np.testing.assert_allclose(rows["observed"], [2.1, 3.0])
    np.testing.assert_allclose(rows["absolute_deviation"], [0.1, 2.5])
    # scale = max(|expected|, |observed|): 2.1 and 3.0
    np.testing.assert_allclose(rows["relative_deviation"], [0.1 / 2.1, 2.5 / 3.0])
    np.testing.assert_allclose(rows["effective_tolerance"], [0.105, 0.03])
    assert list(rows["status"]) == ["valid", "threshold_rejected"]
    assert list(rows["prediction_id"]) == ["cam:prediction:one"] * 2
    assert list(rows["target_identity"]) == ["cam:state:alpha"] * 2
    assert list(rows["prediction_artifact_id"]) == [PREDICTION_ARTIFACT] * 2
    assert list(rows["observation_artifact_id"]) == [OBSERVATION_ARTIFACT] * 2
    assert list(rows["observed_product"]) == ["spectrum"] * 2
    assert list(rows["observed_field"]) == ["response"] * 2
    selection = [json.loads(str(value)) for value in rows["selection"]]
    assert selection[0] == {
        "scan_index": [0],
        "scan_coordinates": {"epsilon": 0.1},
        "selection": {"channel": 0},
    }
    assert selection[1]["scan_index"] == [1]
    assert selection[1]["scan_coordinates"] == {"epsilon": 0.2}


def test_multi_prediction_and_multi_comparison_rows(tmp_path):
    prediction, manifest = _saved_prediction(
        tmp_path, [_prediction(), _prediction("cam:prediction:two")]
    )
    observation = _saved_observation(tmp_path)
    result = validate_prediction(
        prediction,
        manifest,
        observation,
        [
            _comparison(),
            _comparison(metric="full_information_rate"),
            _comparison(prediction_id="cam:prediction:two", scan_index=1),
        ],
    )
    rows = _rows(result)
    assert list(rows["prediction_id"]) == [
        "cam:prediction:one",
        "cam:prediction:one",
        "cam:prediction:two",
    ]
    assert list(rows["metric"]) == [
        "expected_frequency_response_full",
        "full_information_rate",
        "expected_frequency_response_full",
    ]
    assert result.products["prediction_validation"].attributes["validation_count"] == 3


def test_scan_index_accepts_an_int_or_a_tuple(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    for index in (0, (0,)):
        result = validate_prediction(
            prediction, manifest, observation, [_comparison(scan_index=index)]
        )
        rows = _rows(result)
        assert float(rows["observed"][0]) == pytest.approx(2.1)
        assert json.loads(str(rows["selection"][0]))["scan_index"] == [0]


def test_a_two_dimensional_scan_needs_the_grid_index(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(
        tmp_path,
        products={
            "spectrum": _spectrum([[1.0, 1.0], [1.0, 1.0], [2.0, 5.0], [7.0, 7.0]])
        },
        scan_axes={"alpha": [1.0, 2.0], "beta": [10.0, 20.0]},
        scan_shape=(2, 2),
    )
    result = validate_prediction(
        prediction,
        manifest,
        observation,
        [_comparison(scan_index=(1, 0), relative_tolerance=0.5)],
    )
    rows = _rows(result)
    assert float(rows["observed"][0]) == pytest.approx(2.0)
    selection = json.loads(str(rows["selection"][0]))
    assert selection["scan_index"] == [1, 0]
    assert selection["scan_coordinates"] == {"alpha": 2.0, "beta": 10.0}
    with pytest.raises(ValueError, match="one index per scan axis"):
        validate_prediction(
            prediction, manifest, observation, [_comparison(scan_index=(1,))]
        )


def test_the_remaining_axes_are_selected_explicitly(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    tau_values = np.asarray(
        [
            [[2.1, 8.0, 3.0], [9.0, 9.0, 9.0]],
            [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        ]
    )
    observation = _saved_observation(
        tmp_path,
        products={"spectrum": _spectrum(tau_values, dims=("scan", "channel", "tau"))},
    )
    result = validate_prediction(
        prediction,
        manifest,
        observation,
        [
            _comparison(
                relative_tolerance=0.5,
                selection={"channel": 0, "tau": 0},
            )
        ],
    )
    assert float(_rows(result)["observed"][0]) == pytest.approx(2.1)
    with pytest.raises(ValueError, match="select it explicitly"):
        validate_prediction(
            prediction,
            manifest,
            observation,
            [_comparison(selection={"channel": 0})],
        )
    with pytest.raises(ValueError, match="axes the observed field does not have"):
        validate_prediction(
            prediction,
            manifest,
            observation,
            [_comparison(selection={"channel": 0, "tau": 0, "mode": 0})],
        )
    with pytest.raises(ValueError, match="out of bounds"):
        validate_prediction(
            prediction,
            manifest,
            observation,
            [_comparison(selection={"channel": 0, "tau": 3})],
        )


# --- input rejection -------------------------------------------------------


def test_the_verdict_compares_the_absolute_deviation_with_the_tolerance(tmp_path):
    """The comparison is in observable units: absolute deviation, not a ratio."""
    prediction, manifest = _saved_prediction(tmp_path)

    def verdict(observed: float, **overrides) -> tuple[str, dict]:
        tag = abs(hash((observed, tuple(sorted(overrides.items()))))) % 10**20
        obs = _saved_observation(
            tmp_path / f"obs-{tag}",
            products={"spectrum": _spectrum([[observed, 9.0], [3.0, 4.0]])},
            artifact_id=f"{tag:032d}",
        )
        result = validate_prediction(
            prediction, manifest, obs, [_comparison(**overrides)]
        )
        rows = _rows(result)
        return str(rows["status"][0]), {
            "expected": float(rows["expected"][0]),
            "observed": float(rows["observed"][0]),
            "absolute_deviation": float(rows["absolute_deviation"][0]),
            "relative_deviation": float(rows["relative_deviation"][0]),
            "effective_tolerance": float(rows["effective_tolerance"][0]),
        }

    # expected 2.0 vs observed 2.1: absolute deviation 0.1 exceeds atol 0.05 even
    # though the dimensionless relative deviation (0.0476) is below 0.05.
    status, numbers = verdict(2.1, absolute_tolerance=0.05, relative_tolerance=0.0)
    assert status == "threshold_rejected"
    assert numbers["absolute_deviation"] == pytest.approx(0.1)
    assert numbers["relative_deviation"] == pytest.approx(0.1 / 2.1)
    assert numbers["effective_tolerance"] == pytest.approx(0.05)

    # A relative-only tolerance scales with max(|expected|, |observed|).
    status, numbers = verdict(2.1, absolute_tolerance=0.0, relative_tolerance=0.05)
    assert status == "valid"
    assert numbers["effective_tolerance"] == pytest.approx(0.105)

    # Exact boundary: equality passes.
    status, _ = verdict(2.05, absolute_tolerance=0.05, relative_tolerance=0.0)
    assert status == "valid"

    # A large-scale pair with a tiny relative deviation still needs absolute room.
    large_prediction, large_manifest = _saved_prediction(
        tmp_path / "large",
        [
            _prediction(
                expected_frequency_response_full=1000.0,
                allan_intensity_reconstructed=0.5,
                full_information_rate=8.0,
            )
        ],
        artifact_id="aa00000000000000000000000000000000",
    )
    large_observation = _saved_observation(
        tmp_path / "large-obs",
        products={"spectrum": _spectrum([[1001.0, 9.0], [3.0, 4.0]])},
        artifact_id="bb00000000000000000000000000000000",
    )
    result = validate_prediction(
        large_prediction,
        large_manifest,
        large_observation,
        [
            _comparison(
                absolute_tolerance=0.5,
                relative_tolerance=0.0001,
                selection={"channel": 0},
            )
        ],
    )
    rows = _rows(result)
    assert float(rows["absolute_deviation"][0]) == pytest.approx(1.0)
    assert float(rows["effective_tolerance"][0]) == pytest.approx(0.5)
    assert list(rows["status"]) == ["threshold_rejected"]

    # Zero expected and observed is exactly valid, and no NaN enters the table.
    zero_prediction, zero_manifest = _saved_prediction(
        tmp_path / "zero",
        [
            _prediction(
                expected_frequency_response_full=0.0,
                allan_intensity_reconstructed=0.5,
                full_information_rate=8.0,
            )
        ],
        artifact_id="cc00000000000000000000000000000000",
    )
    zero_observation = _saved_observation(
        tmp_path / "zero-obs",
        products={"spectrum": _spectrum([[0.0, 9.0], [3.0, 4.0]])},
        artifact_id="dd00000000000000000000000000000000",
    )
    result = validate_prediction(
        zero_prediction,
        zero_manifest,
        zero_observation,
        [
            _comparison(
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                selection={"channel": 0},
            )
        ],
    )
    rows = _rows(result)
    assert float(rows["relative_deviation"][0]) == 0.0
    assert list(rows["status"]) == ["valid"]


def test_the_manifest_must_actually_publish_the_prediction(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    with pytest.raises(ValueError, match="lists no products"):
        validate_prediction(
            prediction,
            manifest.model_copy(update={"products": []}),
            observation,
            [_comparison()],
        )
    # The observation's manifest is real, but it did not publish this product.
    with pytest.raises(ValueError, match="publishes no product"):
        validate_prediction(
            prediction,
            ArtifactManifest.read(
                tmp_path / f"observation-{OBSERVATION_ARTIFACT}-single"
            ),
            observation,
            [_comparison()],
        )
    # A manifest entry with another schema is not proof either.
    wider, wider_manifest = _saved_prediction(
        tmp_path / "wider",
        [_prediction(), _prediction("cam:prediction:two")],
        artifact_id="ee00000000000000000000000000000000",
    )
    assert wider.axis("prediction").size == 2
    with pytest.raises(ValueError, match="does not match the restored prediction"):
        validate_prediction(wider, manifest, observation, [_comparison()])
    assert wider_manifest.parents == [PREDICTION_ANCESTOR]
    # The other manifest does publish a product named "prediction", but not this one.
    with pytest.raises(ValueError, match="does not match the restored prediction"):
        validate_prediction(prediction, wider_manifest, observation, [_comparison()])


def test_a_prediction_published_under_another_label_is_explicit(tmp_path):
    from qphase.data import save_products

    directory = tmp_path / "labeled"
    directory.mkdir()
    product = build_prediction_product(
        [_prediction()], provenance={"artifact_id": "ff000000000000000000000000000000"}
    )
    save_products(
        directory,
        {"cam_prediction": product},
        provenance={"artifact_id": "ff000000000000000000000000000000"},
        artifact_id="ff000000000000000000000000000000",
        layout="single",
    )
    manifest = ArtifactManifest.read(directory)
    restored = load_bundle(directory).require("cam_prediction")
    observation = _saved_observation(tmp_path)
    with pytest.raises(ValueError, match="publishes no product 'prediction'"):
        validate_prediction(restored, manifest, observation, [_comparison()])
    result = validate_prediction(
        restored,
        manifest,
        observation,
        [_comparison()],
        prediction_label="cam_prediction",
    )
    assert list(_rows(result)["status"]) == ["valid"]


def test_the_observation_must_be_a_real_sde_bundle(tmp_path):
    from qphase.data import GenericDataBundle

    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    generic = GenericDataBundle(
        observation.products,
        observation.bundle_descriptor,
        provenance=observation.provenance.model_dump(mode="json"),
        parents=observation.parents,
    )
    with pytest.raises(TypeError, match="restored SDEDataBundle"):
        validate_prediction(prediction, manifest, generic, [_comparison()])


def test_boolean_or_float_indices_are_rejected():
    with pytest.raises(ValidationError):
        _comparison(scan_index=True)
    with pytest.raises(ValidationError):
        _comparison(selection={"channel": True})
    with pytest.raises(ValidationError):
        _comparison(scan_index=0.0)


def test_a_point_read_does_not_materialize_the_whole_field(tmp_path, monkeypatch):
    """The observed point goes through the lazy point selection, not a full read."""
    from qphase.data.npz import ShardedNpzArrayHandle

    values = np.arange(4 * 256, dtype=float).reshape(4, 256) / 1000.0
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(
        tmp_path,
        products={"spectrum": _spectrum(values)},
        scan_axes={"epsilon": [0.1, 0.2, 0.3, 0.4]},
        scan_shape=(4,),
        artifact_id="77000000000000000000000000000000",
        layout="sharded",
    )
    assert observation.products["spectrum"].axis("scan").size == 4

    calls: list[tuple[str, tuple]] = []
    real_selection = ShardedNpzArrayHandle.materialize_selection
    real_materialize = ShardedNpzArrayHandle.materialize

    def spy_selection(self, indexers):
        calls.append(("selection", tuple(indexers)))
        return real_selection(self, indexers)

    def spy_materialize(self, *args, **kwargs):
        calls.append(("full", ()))
        return real_materialize(self, *args, **kwargs)

    monkeypatch.setattr(ShardedNpzArrayHandle, "materialize_selection", spy_selection)
    monkeypatch.setattr(ShardedNpzArrayHandle, "materialize", spy_materialize)

    result = validate_prediction(
        prediction,
        manifest,
        observation,
        [
            _comparison(
                scan_index=1,
                selection={"channel": 17},
                relative_tolerance=0.5,
            )
        ],
    )
    rows = _rows(result)
    assert float(rows["observed"][0]) == pytest.approx(values[1, 17])
    assert float(rows["expected"][0]) == pytest.approx(2.0)
    # No full read of the field happened; the point selection was used instead.
    assert [kind for kind, _ in calls if kind == "full"] == []
    selections = [indexers for kind, indexers in calls if kind == "selection"]
    assert selections, "the observed point must go through materialize_selection"
    for indexers in selections:
        assert len(indexers) == 2
        assert all(isinstance(index, (int, np.integer)) for index in indexers)


def test_a_prediction_that_cannot_be_validated_is_rejected(tmp_path):
    observation = _saved_observation(tmp_path)
    intermediate, intermediate_manifest = _saved_prediction(
        tmp_path / "intermediate",
        [
            _prediction(
                completeness="intermediate",
                reason="no published source",
                source_artifacts=(),
            )
        ],
        artifact_id="33333333333333333333333333333333",
    )
    with pytest.raises(ValueError, match="only a complete prediction"):
        validate_prediction(
            intermediate, intermediate_manifest, observation, [_comparison()]
        )

    failing, failing_manifest = _saved_prediction(
        tmp_path / "failing",
        [
            _prediction(
                status=CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION,
                reason="not confining",
            )
        ],
        artifact_id="44444444444444444444444444444444",
    )
    with pytest.raises(ValueError, match="only a valid prediction"):
        validate_prediction(failing, failing_manifest, observation, [_comparison()])

    missing, missing_manifest = _saved_prediction(
        tmp_path / "missing",
        [
            _prediction(
                expected_frequency_response_full=None,
                status=CAMOutcomeStatus.VALID,
            )
        ],
        artifact_id="55555555555555555555555555555555",
    )
    with pytest.raises(ValueError, match="no finite"):
        validate_prediction(missing, missing_manifest, observation, [_comparison()])


def test_an_unknown_prediction_id_is_rejected(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    with pytest.raises(ValueError, match="unknown prediction id"):
        validate_prediction(
            prediction,
            manifest,
            observation,
            [_comparison(prediction_id="cam:prediction:absent")],
        )


def test_a_product_that_is_not_a_prediction_is_rejected(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    other = observation.products["spectrum"]
    with pytest.raises(ValueError, match="CAM prediction product"):
        validate_prediction(other, manifest, observation, [_comparison()])
    assert prediction.attributes["product"] == "prediction"


def test_the_prediction_artifact_identity_must_match_the_manifest(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    other_manifest = manifest.model_copy(
        update={"artifact_id": "99999999999999999999999999999999"}
    )
    with pytest.raises(ValueError, match="states artifact id"):
        validate_prediction(prediction, other_manifest, observation, [_comparison()])
    blank = manifest.model_copy(update={"artifact_id": ""})
    with pytest.raises(ValueError, match="no artifact id"):
        validate_prediction(prediction, blank, observation, [_comparison()])


def test_an_observation_without_an_artifact_id_is_rejected(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    loose = SDEDataBundle(
        observation.products,
        observation.provenance,
        scan_axes=observation.axes,
        scan_shape=observation.shape,
    )
    assert loose.declared_artifact_id() is None
    with pytest.raises(ValueError, match="states no artifact id"):
        validate_prediction(prediction, manifest, loose, [_comparison()])


def test_missing_or_non_finite_observations_are_rejected(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    with pytest.raises(ValueError, match="has no product"):
        validate_prediction(
            prediction,
            manifest,
            observation,
            [_comparison(observed_product="missing")],
        )
    with pytest.raises(ValueError, match="has no field"):
        validate_prediction(
            prediction,
            manifest,
            observation,
            [_comparison(observed_field="missing")],
        )
    with pytest.raises(ValueError, match="out of bounds"):
        validate_prediction(
            prediction, manifest, observation, [_comparison(scan_index=2)]
        )
    with pytest.raises(ValueError, match="out of bounds"):
        validate_prediction(
            prediction, manifest, observation, [_comparison(selection={"channel": 5})]
        )

    nan_observation = _saved_observation(
        tmp_path / "nan",
        products={"spectrum": _spectrum([[np.nan, 1.0], [1.0, 1.0]])},
        artifact_id="66666666666666666666666666666666",
    )
    with pytest.raises(ValueError, match="not finite"):
        validate_prediction(prediction, manifest, nan_observation, [_comparison()])

    text_observation = _saved_observation(
        tmp_path / "text",
        products=_text_product(),
        artifact_id="77777777777777777777777777777777",
    )
    with pytest.raises(ValueError, match="not numeric"):
        validate_prediction(
            prediction,
            manifest,
            text_observation,
            [_comparison(observed_product="labels", observed_field="name")],
        )


def _text_product() -> dict:
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="scan", role=AxisRole.INDEX, size=2)],
        variables=[
            VariableSchema(name="name", dtype="U4", value_domain="real", dims=("scan",))
        ],
    )
    return {
        "labels": StatisticsDataset.from_arrays(
            schema, {"name": np.asarray(["a", "b"], dtype="U4")}, owner="engine.sde"
        )
    }


def test_a_single_valued_axis_needs_no_selection(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(
        tmp_path, products={"spectrum": _spectrum([[2.1], [3.0]])}
    )
    result = validate_prediction(
        prediction,
        manifest,
        observation,
        [_comparison(selection={})],
    )
    rows = _rows(result)
    assert float(rows["observed"][0]) == pytest.approx(2.1)
    assert json.loads(str(rows["selection"][0]))["selection"] == {"channel": 0}


def test_a_scanned_field_needs_a_scan_index(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    with pytest.raises(ValueError, match="select one with scan_index"):
        validate_prediction(
            prediction, manifest, observation, [_comparison(scan_index=None)]
        )


def test_a_duplicated_prediction_id_in_the_table_is_rejected(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    doubled = StatisticsDataset.from_arrays(
        prediction.schema.model_copy(
            update={
                "axes": [
                    prediction.axis("prediction").model_copy(update={"size": 2})
                ]
            }
        ),
        {
            variable.name: np.repeat(
                np.asarray(prediction.handle(variable.name).materialize()), 2, axis=0
            )
            for variable in prediction.variables
        },
        owner="engine.cam",
    )
    with pytest.raises(ValueError, match="repeats prediction ids"):
        validate_prediction(doubled, manifest, observation, [_comparison()])


def test_an_empty_comparison_list_is_rejected(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    with pytest.raises(ValueError, match="at least one comparison"):
        validate_prediction(prediction, manifest, observation, [])
    with pytest.raises(TypeError, match="PredictionComparison"):
        validate_prediction(prediction, manifest, observation, [("bad",)])


def test_the_comparison_contract_is_closed():
    with pytest.raises(ValidationError, match="metric"):
        _comparison(metric="allan_intensity_soft")
    with pytest.raises(ValidationError):
        _comparison(absolute_tolerance=-1.0)
    with pytest.raises(ValidationError):
        _comparison(relative_tolerance=float("nan"))
    with pytest.raises(ValidationError):
        _comparison(relative_tolerance=float("inf"))
    with pytest.raises(ValidationError, match="non-negative"):
        _comparison(scan_index=-1)
    with pytest.raises(ValidationError, match="non-negative"):
        _comparison(selection={"channel": -2})
    with pytest.raises(ValidationError):
        _comparison(unknown_option=1)


# --- identities, parents and persistence ------------------------------------


def test_the_result_names_both_inputs_and_states_no_id_of_its_own(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    result = validate_prediction(prediction, manifest, observation, [_comparison()])
    assert result.declared_artifact_id() is None
    assert result.parents == (
        PREDICTION_ANCESTOR,
        PREDICTION_ARTIFACT,
        OBSERVATION_ANCESTOR,
        OBSERVATION_ARTIFACT,
    )
    assert result.provenance.schema_version == "qphase_sde.provenance/1"
    assert result.provenance == observation.provenance
    assert result.metadata["prediction_artifact_id"] == PREDICTION_ARTIFACT
    assert result.metadata["observation_artifact_id"] == OBSERVATION_ARTIFACT
    assert result.metadata["comparison_count"] == 1


def test_shared_ancestors_are_deduplicated_in_order(tmp_path):
    shared = "88888888888888888888888888888888"
    prediction, manifest = _saved_prediction(
        tmp_path, parents=(shared, PREDICTION_ANCESTOR)
    )
    observation = _saved_observation(tmp_path, parents=(shared, OBSERVATION_ANCESTOR))
    result = validate_prediction(prediction, manifest, observation, [_comparison()])
    assert result.parents == (
        shared,
        PREDICTION_ANCESTOR,
        PREDICTION_ARTIFACT,
        OBSERVATION_ANCESTOR,
        OBSERVATION_ARTIFACT,
    )


@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_the_result_persists_with_its_own_id_and_the_chain(tmp_path, layout):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path, layout=layout)
    result = validate_prediction(prediction, manifest, observation, [_comparison()])
    target = tmp_path / f"validation-{layout}"
    target.mkdir()
    if layout == "single":
        result.save(target)
    else:
        result.save_dataset(target, layout="sharded", shard_target_bytes=256)
    saved = ArtifactManifest.read(target)
    assert saved.parents == [
        PREDICTION_ANCESTOR,
        PREDICTION_ARTIFACT,
        OBSERVATION_ANCESTOR,
        OBSERVATION_ARTIFACT,
    ]
    assert saved.artifact_id not in {
        PREDICTION_ARTIFACT,
        OBSERVATION_ARTIFACT,
    }
    restored = load_bundle(target)
    assert isinstance(restored, SDEDataBundle)
    assert restored.declared_artifact_id() == saved.artifact_id
    assert restored.parents == tuple(saved.parents)
    rows = _rows(restored)
    assert list(rows["status"]) == ["valid"]
    assert float(rows["observed"][0]) == pytest.approx(2.1)
    assert json.loads(str(rows["selection"][0]))["selection"] == {"channel": 0}


def test_the_prediction_is_never_modified(tmp_path):
    prediction, manifest = _saved_prediction(tmp_path)
    observation = _saved_observation(tmp_path)
    before = {
        variable.name: np.asarray(prediction.handle(variable.name).materialize())
        for variable in prediction.variables
    }
    before_schema = prediction.schema.model_dump(mode="json")
    result = validate_prediction(prediction, manifest, observation, [_comparison()])
    after = {
        variable.name: np.asarray(prediction.handle(variable.name).materialize())
        for variable in prediction.variables
    }
    assert prediction.schema.model_dump(mode="json") == before_schema
    for name, values in before.items():
        np.testing.assert_array_equal(after[name], values, err_msg=name)
    # Saving the validation result does not touch the prediction artifact either.
    target = tmp_path / "validation"
    target.mkdir()
    result.save(target)
    assert ArtifactManifest.read(tmp_path / f"prediction-{PREDICTION_ARTIFACT}") == (
        manifest
    )
    assert prediction.provenance["artifact_id"] == manifest.artifact_id


# --- dependency direction ---------------------------------------------------


def test_the_service_does_not_import_the_cam_package():
    source = (
        Path(__file__).resolve().parents[2]
        / "packages/qphase_sde/qphase_sde/prediction.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported if "qphase_cam" in name]
    assert "qphase.cam" not in imported
