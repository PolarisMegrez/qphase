"""Continuation and annotation products: schema, builders, import, storage.

Like the 10C1 tests these build small arrays directly: the products are about
layout, identity and mapping, and Phase 10A already freezes the numerics.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
import qphase_cam.data as cam_data
from qphase.data import (
    GenericDataBundle,
    ProductSchema,
    load_bundle,
    load_products,
    save_products,
)
from qphase_cam.data import (
    CAM_ANNOTATION_PRODUCT,
    CAM_BUNDLE_TYPE_ID,
    CAM_CONTINUATION_PRODUCT,
    CAMOutcomeStatus,
    CAMTargetKind,
    build_annotation_product,
    build_continuation_product,
    cam_bundle,
    coerce_outcome_status,
    coerce_target_kind,
    compare_continuation_export,
    import_continuation_result,
    import_legacy_artifact,
)
from qphase_cam.result import CAMResult

#: Imported path identity: ``legacy:<16 hex digits>:<path>``.
LEGACY_PATH_IDENTITY = re.compile(
    r"^legacy:(?P<fingerprint>[0-9a-f]{16}):(?P<path>\d+)$"
)

#: Every outcome status the vocabulary must offer (negative results included).
ALL_OUTCOMES = (
    "valid",
    "not_found",
    "not_searched",
    "not_converged",
    "nonphysical",
    "unstable",
    "stochastic_invalid",
    "unidentifiable",
    "uncontrolled_approximation",
    "threshold_rejected",
    "unchecked",
)


# --- fixtures --------------------------------------------------------------


def _path_payload(*, points=3, rows=2, cols=2):
    """Return builder keyword arguments for one valid continuation path."""
    return {
        "continuation_coordinate": np.linspace(0.0, 0.2, points),
        "states": np.asarray(
            [np.eye(rows) * (index + 1) for index in range(points)], dtype=complex
        ).reshape(points, rows, cols),
        "residuals": np.full(points, 1e-12),
        "success": np.ones(points, dtype=bool),
    }


def _legacy_continuation(*, points=4, rounds=3):
    """Return a legacy continuation-shaped ``CAMResult``.

    The shape mirrors what ``ContinuationSolver`` + ``Engine._pack`` produce: a
    leading point axis, a fixed-capacity solution axis with only slot 0 filled,
    one continuation coordinate axis and ``meta['continuation'] = True``.
    """
    capacity = 4
    coordinate = np.linspace(0.5, 0.5001, points)
    states = np.full((points, capacity, 2, 2), np.nan + 1j * np.nan)
    residuals = np.full((points, capacity), np.nan)
    success = np.zeros((points, capacity), dtype=bool)
    valid = np.zeros((points, capacity), dtype=bool)
    for index in range(points):
        states[index, 0] = np.eye(2) * (index + 1) + 0.5j
        residuals[index, 0] = 1e-12 * (index + 1)
        success[index, 0] = index < rounds
        valid[index, 0] = True
    return CAMResult(
        states=states,
        residuals=residuals,
        success=success,
        valid_mask=valid,
        solution_count=np.ones(points, dtype=np.int64),
        params={"gamma_b": coordinate[0], "Gamma": 1.0, "operator": np.eye(2)},
        axes={"gamma_b": coordinate},
        postprocess={"response": np.asarray([1.0] * points)},
        meta={"continuation": True, "label": "cont-run", "solver": "continuation"},
    )


def _no_object_dtypes(product):
    """Assert no variable of a product can carry a Python object."""
    for variable in product.variables:
        assert not np.dtype(variable.dtype).hasobject, variable.name
        array = np.asarray(product.handle(variable.name).materialize())
        assert not array.dtype.hasobject, variable.name


def _assert_roundtrip(bundle, *, tmp_path, layout):
    root = tmp_path / layout
    report = bundle.save_dataset(root, layout=layout, shard_target_bytes=256)
    assert report.layout == layout
    assert report.files
    assert set(load_products(root)) == set(bundle.products)
    restored = load_bundle(root)
    assert isinstance(restored, GenericDataBundle)
    assert restored.bundle_descriptor.type_id == CAM_BUNDLE_TYPE_ID
    assert restored.provenance
    return restored


# --- templates and vocabulary ---------------------------------------------


def test_continuation_and_annotation_templates_keep_their_axes_open():
    for template in (CAM_CONTINUATION_PRODUCT, CAM_ANNOTATION_PRODUCT):
        assert isinstance(template, ProductSchema)
        assert template.is_closed is False
        assert all(axis.size is None for axis in template.axes)
    assert [axis.name for axis in CAM_CONTINUATION_PRODUCT.axes] == [
        "point",
        "path",
        "path_offset",
        "state_row",
        "state_col",
    ]
    assert CAM_CONTINUATION_PRODUCT.axis("point").role.value == "parameter"
    assert [axis.name for axis in CAM_ANNOTATION_PRODUCT.axes] == ["annotation"]
    assert CAM_CONTINUATION_PRODUCT.attributes["product"] == "continuation"
    assert CAM_ANNOTATION_PRODUCT.attributes["product"] == "annotations"
    # A path identity is not a branch identity; the schema says so itself.
    assert (
        "not claims about one global branch"
        in (CAM_CONTINUATION_PRODUCT.attributes["identity_semantics"])
    )


def test_outcome_vocabulary_covers_the_negative_results():
    values = [member.value for member in CAMOutcomeStatus]
    assert values == list(ALL_OUTCOMES)
    assert CAMOutcomeStatus.VALID == "valid"
    assert CAMOutcomeStatus("not_searched") is CAMOutcomeStatus.NOT_SEARCHED
    assert coerce_outcome_status("not_found") is CAMOutcomeStatus.NOT_FOUND
    with pytest.raises(ValueError, match="unknown CAM outcome status"):
        coerce_outcome_status("rejected")
    assert [member.value for member in CAMTargetKind] == [
        "state",
        "candidate",
        "path",
        "case",
    ]
    assert coerce_target_kind("path") is CAMTargetKind.PATH
    with pytest.raises(ValueError, match="unknown CAM target kind"):
        coerce_target_kind("branch")


# --- continuation builder --------------------------------------------------


def test_build_continuation_product_normalizes_one_path():
    product = build_continuation_product(
        path_identity=["solver:path:0"],
        **_path_payload(points=3),
        parameters={"gamma": 0.25},
    )
    assert product.attributes["path_count"] == 1
    assert product.attributes["point_count"] == 3
    assert product.attributes["state_shape"] == [2, 2]
    np.testing.assert_array_equal(
        np.asarray(product.handle("path_offsets").materialize()), [0, 3]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("path_identity").materialize()), ["solver:path:0"]
    )
    assert product.axis("point").size == 3
    assert product.axis("path").size == 1
    assert product.axis("path_offset").size == 2
    assert product.shape["states"] == (3, 2, 2)
    _no_object_dtypes(product)


def test_build_continuation_product_keeps_several_paths():
    payload = _path_payload(points=4)
    product = build_continuation_product(
        path_identity=["solver:path:0", "solver:path:1"],
        path_offsets=np.asarray([0, 1, 4]),
        **payload,
    )
    assert product.attributes["path_count"] == 2
    assert product.attributes["point_count"] == 4
    np.testing.assert_array_equal(
        np.asarray(product.handle("path_offsets").materialize()), [0, 1, 4]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("path_identity").materialize()),
        ["solver:path:0", "solver:path:1"],
    )
    assert product.axis("path").size == 2
    assert product.axis("path_offset").size == 3
    # The point block stays flat: the offsets, not a slot, say which path a
    # point belongs to.
    np.testing.assert_allclose(
        np.asarray(product.handle("continuation_coordinate").materialize()),
        payload["continuation_coordinate"],
    )


def test_build_continuation_product_keeps_a_path_without_points():
    payload = _path_payload(points=2)
    product = build_continuation_product(
        path_identity=["solver:path:0", "solver:path:1", "solver:path:2"],
        path_offsets=np.asarray([0, 2, 2, 2]),
        **payload,
    )
    assert product.attributes["path_count"] == 3
    assert product.attributes["point_count"] == 2
    np.testing.assert_array_equal(
        np.asarray(product.handle("path_offsets").materialize()), [0, 2, 2, 2]
    )
    assert product.axis("path").size == 3


def test_build_continuation_product_allows_a_product_without_any_path():
    product = build_continuation_product(
        path_identity=[],
        path_offsets=np.asarray([0]),
        continuation_coordinate=np.asarray([], dtype=np.float64),
        states=np.zeros((0, 2, 2), dtype=complex),
        residuals=np.asarray([], dtype=np.float64),
        success=np.asarray([], dtype=bool),
    )
    assert product.attributes["path_count"] == 0
    assert product.attributes["point_count"] == 0
    assert product.axis("path").size == 0
    assert product.axis("point").size == 0
    _no_object_dtypes(product)


@pytest.mark.parametrize(
    "identity, match",
    [
        (["a", ""], "is empty"),
        (["a", "a"], "must be unique"),
    ],
)
def test_continuation_path_identity_must_be_explicit_and_unique(identity, match):
    payload = _path_payload(points=2)
    with pytest.raises(ValueError, match=match):
        build_continuation_product(
            path_identity=identity,
            path_offsets=np.asarray([0, 1, 2]),
            **payload,
        )


@pytest.mark.parametrize(
    "offsets, match",
    [
        ([1, 3], "must start at 0"),
        ([0, 3, 1], "monotonic"),
        ([0, 2], "must end at the point count"),
        ([[0, 3]], "one-dimensional with n_path"),
        ([0], "must end at the point count"),
    ],
)
def test_continuation_path_offsets_are_validated(offsets, match):
    payload = _path_payload(points=3)
    with pytest.raises(ValueError, match=match):
        build_continuation_product(
            path_identity=["p0", "p1"],
            path_offsets=np.asarray(offsets),
            **payload,
        )


def test_continuation_offsets_must_match_the_path_identity_count():
    payload = _path_payload(points=2)
    with pytest.raises(ValueError, match="describes 2 paths but path_identity"):
        build_continuation_product(
            path_identity=["only-one"],
            path_offsets=np.asarray([0, 1, 2]),
            **payload,
        )


def test_continuation_states_must_cover_the_point_axis():
    payload = _path_payload(points=3)
    payload["states"] = payload["states"][:2]
    with pytest.raises(ValueError, match="rows for 3 points"):
        build_continuation_product(path_identity=["p0"], **payload)


def test_continuation_coordinate_must_be_one_dimensional():
    payload = _path_payload(points=3)
    payload["continuation_coordinate"] = np.zeros((3, 1))
    with pytest.raises(ValueError, match="one-dimensional"):
        build_continuation_product(path_identity=["p0"], **payload)


def test_continuation_carries_the_swept_parameter_once():
    payload = _path_payload(points=3)
    coordinate = payload["continuation_coordinate"]
    product = build_continuation_product(
        path_identity=["p0"],
        continuation_coordinate_name="gamma",
        parameters={"gamma": coordinate, "Gamma": 1.0},
        axes={"gamma": coordinate},
        **payload,
    )
    names = {variable.name for variable in product.variables}
    assert "parameter.gamma" not in names
    assert "axis.gamma" not in names
    assert product.attributes["continuation_parameter_from_parameters"] == "duplicate"
    assert product.attributes["continuation_parameter_from_axes"] == "duplicate"
    # The continuation coordinate is registered last, so it is the parameter
    # coordinate a bundle reports for the point axis.
    coordinates = {item.name: item for item in product.coordinates()}
    assert list(coordinates)[-1] == "gamma"
    assert coordinates["gamma"].variable == "continuation_coordinate"
    assert coordinates["gamma"].role == "parameter"
    assert coordinates["Gamma"].variable == "parameter.Gamma"
    np.testing.assert_allclose(product.coordinate("gamma"), coordinate)


def test_continuation_rejects_a_contradictory_swept_parameter():
    payload = _path_payload(points=3)
    with pytest.raises(ValueError, match="disagrees with continuation_coordinate"):
        build_continuation_product(
            path_identity=["p0"],
            continuation_coordinate_name="gamma",
            parameters={"gamma": np.zeros(3)},
            **payload,
        )


def test_continuation_reports_a_swept_parameter_that_is_not_per_point():
    payload = _path_payload(points=3)
    product = build_continuation_product(
        path_identity=["p0"],
        continuation_coordinate_name="gamma",
        parameters={"gamma": 0.0},
        **payload,
    )
    assert product.attributes["continuation_parameter_from_parameters"] == (
        "not_per_point"
    )
    assert product.provenance["unmapped_parameter_names"] == ["parameter.gamma"]


def test_continuation_events_are_canonical_json_text():
    payload = _path_payload(points=2)
    product = build_continuation_product(
        path_identity=["p0"],
        events=[{"event": "fold", "at": 0.1}, {}],
        **payload,
    )
    events = [
        json.loads(str(value))
        for value in np.asarray(product.handle("events").materialize())
    ]
    assert events == [{"at": 0.1, "event": "fold"}, {}]


# --- continuation import ---------------------------------------------------


def test_import_continuation_result_maps_the_path():
    legacy = _legacy_continuation()
    bundle = import_continuation_result(legacy)
    product = bundle.require("continuation")
    assert list(bundle.products) == ["continuation"]
    identity = [
        str(value)
        for value in np.asarray(product.handle("path_identity").materialize())
    ]
    match = LEGACY_PATH_IDENTITY.match(identity[0])
    assert match is not None, identity
    assert match["path"] == "0"
    assert bundle.provenance["legacy_source_fingerprint"] == match["fingerprint"]
    np.testing.assert_array_equal(
        np.asarray(product.handle("path_offsets").materialize()), [0, 4]
    )
    np.testing.assert_allclose(
        np.asarray(product.handle("continuation_coordinate").materialize()),
        legacy.axes["gamma_b"],
    )
    # The filled slot of every point becomes the point's state.
    np.testing.assert_allclose(
        np.asarray(product.handle("states").materialize()),
        np.asarray(legacy.states)[:, 0],
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("success").materialize()),
        [True, True, True, False],
    )
    assert product.attributes["continuation_coordinate_name"] == "gamma_b"
    # The legacy scalar start value cannot become a point column; the coordinate
    # carries the swept values and the name is reported, not dropped silently.
    assert product.attributes["continuation_parameter_from_parameters"] == (
        "not_per_point"
    )
    assert bundle.provenance["unmapped_legacy_parameters"] == [
        "operator",
        "parameter.gamma_b",
    ]
    assert bundle.provenance["unmapped_legacy_postprocess"] == ["response"]
    assert bundle.provenance["unmapped_legacy_axes"] == []
    assert bundle.provenance["legacy_point_count"] == 4
    assert bundle.provenance["legacy_meta"]["label"] == "cont-run"
    report = compare_continuation_export(product, legacy)
    assert report["passed"], report["mismatches"]


def test_import_continuation_result_rejects_a_non_continuation_result():
    legacy = _legacy_continuation()
    legacy.meta["continuation"] = False
    with pytest.raises(ValueError, match="not a continuation result"):
        import_continuation_result(legacy)


def test_import_continuation_result_requires_exactly_one_axis():
    legacy = _legacy_continuation()
    legacy.axes["gamma_a"] = np.zeros(4)
    with pytest.raises(ValueError, match="exactly one continuation coordinate"):
        import_continuation_result(legacy)


def test_import_continuation_result_respects_the_source_id():
    legacy = _legacy_continuation()
    bundle = import_continuation_result(legacy, source_id="artifact-3")
    identity = [
        str(value)
        for value in np.asarray(
            bundle.require("continuation").handle("path_identity").materialize()
        )
    ]
    assert identity == ["legacy:artifact-3:0"]
    assert bundle.provenance["legacy_source_token"] == "source_id"
    # Re-importing the same snapshot keeps the content fingerprint.
    other = import_continuation_result(legacy)
    assert other.provenance["legacy_source_fingerprint"] != "artifact-3"


def test_import_legacy_artifact_dispatches_a_continuation_payload(tmp_path):
    legacy = _legacy_continuation()
    target = tmp_path / "legacy_continuation"
    legacy.save(target)
    bundle = import_legacy_artifact(target)
    assert list(bundle.products) == ["continuation"]
    assert bundle.require("continuation").attributes["point_count"] == 4


# --- annotation builder ----------------------------------------------------


def test_build_annotation_product_declares_typed_fields():
    product = build_annotation_product(
        annotation_kind="stochastic_validity",
        target_kind="candidate",
        target_identity=["candidate:0", "candidate:1"],
        status=["valid", "stochastic_invalid"],
        fields={
            "margin": np.asarray([0.5, -0.1]),
            "frequencies": np.asarray([[1.0, 2.0], [1.5, 2.5]]),
            "covariance": np.asarray([np.eye(2), np.eye(2) * 2]),
        },
        field_axes={
            "frequencies": ("mode",),
            "covariance": ("mode_row", "mode_col"),
        },
        axes={"mode": 2, "mode_row": 2, "mode_col": 2},
        method_metadata=[
            {"method": "additive"},
            {"method": "additive", "note": "edge"},
        ],
    )
    assert product.attributes["annotation_kind"] == "stochastic_validity"
    assert product.attributes["target_kind"] == "candidate"
    assert product.attributes["annotation_count"] == 2
    assert product.attributes["field_variables"] == [
        {"name": "margin", "dtype": "<f8", "dims": ["annotation"]},
        {"name": "frequencies", "dtype": "<f8", "dims": ["annotation", "mode"]},
        {
            "name": "covariance",
            "dtype": "<f8",
            "dims": ["annotation", "mode_row", "mode_col"],
        },
    ]
    assert [axis.name for axis in product.axes] == [
        "annotation",
        "mode",
        "mode_row",
        "mode_col",
    ]
    # Scalar, vector and matrix fields are real variables with real dims.
    assert product.variable("margin").dims == ("annotation",)
    assert product.variable("frequencies").dims == ("annotation", "mode")
    assert product.variable("covariance").dims == (
        "annotation",
        "mode_row",
        "mode_col",
    )
    assert product.shape["covariance"] == (2, 2, 2)
    np.testing.assert_allclose(
        np.asarray(product.handle("margin").materialize()), [0.5, -0.1]
    )
    assert list(np.asarray(product.handle("status").materialize())) == [
        "valid",
        "stochastic_invalid",
    ]
    metadata = [
        json.loads(str(value))
        for value in np.asarray(product.handle("method_metadata").materialize())
    ]
    assert metadata == [{"method": "additive"}, {"method": "additive", "note": "edge"}]
    # Numbers are not smuggled through JSON: the fields are numeric columns.
    assert product.variable("margin").value_domain == "real"
    _no_object_dtypes(product)


def test_annotations_name_their_trailing_axes_by_default():
    product = build_annotation_product(
        annotation_kind="petermann",
        target_kind="state",
        target_identity=["state:0"],
        status="valid",
        fields={"excess_noise": np.asarray([[1.0, 1.1]])},
    )
    assert product.variable("excess_noise").dims == ("annotation", "excess_noise_dim0")
    assert product.axis("excess_noise_dim0").size == 2


def test_annotation_carries_every_outcome_status():
    product = build_annotation_product(
        annotation_kind="acceptance",
        target_kind="case",
        target_identity=[f"case:{index}" for index in range(len(ALL_OUTCOMES))],
        status=list(ALL_OUTCOMES),
        fields={"score": np.arange(len(ALL_OUTCOMES), dtype=float)},
    )
    assert list(np.asarray(product.handle("status").materialize())) == list(
        ALL_OUTCOMES
    )
    assert product.attributes["status_vocabulary"] == list(ALL_OUTCOMES)
    # A negative result is a distinct status, never one "rejected" flag.
    assert "not_found" in ALL_OUTCOMES
    assert "not_searched" in ALL_OUTCOMES
    _no_object_dtypes(product)


def test_annotation_status_accepts_an_enum_for_the_whole_product():
    product = build_annotation_product(
        annotation_kind="physicality",
        target_kind=CAMTargetKind.STATE,
        target_identity=["state:0", "state:1"],
        status=CAMOutcomeStatus.NOT_SEARCHED,
    )
    assert list(np.asarray(product.handle("status").materialize())) == [
        "not_searched",
        "not_searched",
    ]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"target_identity": ["a", ""]}, "is empty"),
        ({"target_identity": ["a", "a"]}, "must be unique"),
        ({"status": ["valid", "rejected"]}, "unknown CAM outcome status"),
        ({"target_kind": "branch"}, "unknown CAM target kind"),
    ],
)
def test_annotation_validates_targets_status_and_kind(overrides, match):
    arguments = {
        "annotation_kind": "physicality",
        "target_kind": "state",
        "target_identity": ["a", "b"],
        "status": ["valid", "valid"],
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=match):
        build_annotation_product(**arguments)


def test_annotation_rejects_a_json_field():
    with pytest.raises(ValueError, match="object array"):
        build_annotation_product(
            annotation_kind="physicality",
            target_kind="state",
            target_identity=["state:0"],
            status="valid",
            fields={"payload": np.asarray([{"value": 1.0}], dtype=object)},
        )


def test_annotation_field_shapes_must_be_consistent():
    with pytest.raises(ValueError, match="already declared with size 2"):
        build_annotation_product(
            annotation_kind="petermann",
            target_kind="state",
            target_identity=["state:0"],
            status="valid",
            fields={
                "a": np.zeros((1, 2)),
                "b": np.zeros((1, 3)),
            },
            field_axes={"a": ("mode",), "b": ("mode",)},
        )
    with pytest.raises(ValueError, match="rows for 1 annotations"):
        build_annotation_product(
            annotation_kind="petermann",
            target_kind="state",
            target_identity=["state:0"],
            status="valid",
            fields={"a": np.zeros(2)},
        )
    with pytest.raises(ValueError, match="repeats a trailing axis name"):
        build_annotation_product(
            annotation_kind="petermann",
            target_kind="state",
            target_identity=["state:0"],
            status="valid",
            fields={"a": np.zeros((1, 2, 2))},
            field_axes={"a": ("mode", "mode")},
        )


def test_annotation_rejects_reserved_and_unknown_field_names():
    common = {
        "annotation_kind": "petermann",
        "target_kind": "state",
        "target_identity": ["state:0"],
        "status": "valid",
    }
    with pytest.raises(ValueError, match="is reserved"):
        build_annotation_product(**common, fields={"status": np.zeros(1)})
    with pytest.raises(ValueError, match="must be a valid identifier"):
        build_annotation_product(**common, fields={"not a name": np.zeros(1)})
    with pytest.raises(ValueError, match="unknown fields"):
        build_annotation_product(
            **common, fields={"a": np.zeros(1)}, field_axes={"b": ("mode",)}
        )
    with pytest.raises(ValueError, match="not used by any field"):
        build_annotation_product(**common, fields={"a": np.zeros(1)}, axes={"mode": 2})


def test_annotation_empty_product_is_representable():
    product = build_annotation_product(
        annotation_kind="stochastic_validity",
        target_kind="candidate",
        target_identity=[],
        status=[],
    )
    assert product.attributes["annotation_count"] == 0
    assert product.axis("annotation").size == 0
    _no_object_dtypes(product)


# --- storage ---------------------------------------------------------------


def test_continuation_bundle_roundtrips_in_both_layouts(tmp_path):
    payload = _path_payload(points=3)
    product = cam_bundle(
        {
            "continuation": build_continuation_product(
                path_identity=["solver:path:0"],
                continuation_coordinate_name="gamma",
                parameters={"gamma": payload["continuation_coordinate"]},
                events=[{}, {"event": "fold"}, {}],
                **payload,
            )
        },
        provenance={"cam_task": "continue"},
    )
    for layout in ("single", "sharded"):
        restored = _assert_roundtrip(product, tmp_path=tmp_path, layout=layout)
        assert restored.bundle_descriptor.product_roles == {
            "continuation": "continuation"
        }
        reloaded = restored.require("continuation").materialize()
        assert reloaded.attributes["path_count"] == 1
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("path_identity").materialize()),
            ["solver:path:0"],
        )
        np.testing.assert_allclose(restored.axes["point"], np.linspace(0.0, 0.2, 3))
        assert [
            json.loads(str(value))
            for value in np.asarray(reloaded.handle("events").materialize())
        ] == [{}, {"event": "fold"}, {}]
        _no_object_dtypes(reloaded)


def test_annotation_bundle_roundtrips_in_both_layouts(tmp_path):
    bundle = cam_bundle(
        {
            "annotations": build_annotation_product(
                annotation_kind="stochastic_validity",
                target_kind="candidate",
                target_identity=["candidate:0", "candidate:1"],
                status=["valid", "stochastic_invalid"],
                fields={"frequencies": np.asarray([[1.0, 2.0], [1.5, 2.5]])},
                field_axes={"frequencies": ("mode",)},
                axes={"mode": 2},
            )
        },
        provenance={"cam_task": "analyze"},
    )
    for layout in ("single", "sharded"):
        restored = _assert_roundtrip(bundle, tmp_path=tmp_path, layout=layout)
        assert restored.bundle_descriptor.product_roles == {
            "annotations": "annotations"
        }
        reloaded = restored.require("annotations").materialize()
        np.testing.assert_allclose(
            np.asarray(reloaded.handle("frequencies").materialize()),
            [[1.0, 2.0], [1.5, 2.5]],
        )
        assert reloaded.variable("frequencies").dims == ("annotation", "mode")
        assert reloaded.attributes["annotation_kind"] == "stochastic_validity"
        _no_object_dtypes(reloaded)


def test_imported_continuation_bundle_persists_provenance(tmp_path):
    bundle = import_continuation_result(_legacy_continuation())
    root = tmp_path / "artifact"
    save_products(
        root,
        bundle.products,
        provenance=bundle.provenance,
        bundle=bundle.bundle_descriptor,
        layout="single",
    )
    restored = load_bundle(root)
    assert restored.provenance["legacy_continuation_parameter"] == "gamma_b"
    assert restored.provenance["unmapped_legacy_postprocess"] == ["response"]


# --- point views -----------------------------------------------------------


def test_point_view_reads_continuation_strings_and_coordinates():
    payload = _path_payload(points=2)
    product = build_continuation_product(
        path_identity=["solver:path:0"],
        continuation_coordinate_name="gamma",
        events=[{"event": "fold"}, {}],
        **payload,
    )
    view = product.point_view(point=0)
    assert list(np.asarray(view.handle("path_identity").materialize())) == [
        "solver:path:0"
    ]
    assert json.loads(str(np.asarray(view.handle("events").materialize()))) == {
        "event": "fold"
    }
    np.testing.assert_allclose(
        np.asarray(view.handle("continuation_coordinate").materialize()), 0.0
    )
    assert view.shape["states"] == (2, 2)


def test_point_view_reads_annotation_strings_and_fields():
    product = build_annotation_product(
        annotation_kind="petermann",
        target_kind="state",
        target_identity=["state:0", "state:1"],
        status=["valid", "nonphysical"],
        fields={"excess_noise": np.asarray([1.0, 0.5])},
        method_metadata=[{"method": "bdg"}, {"method": "bdg", "note": "clipped"}],
    )
    view = product.point_view(annotation=1)
    assert str(np.asarray(view.handle("target_identity").materialize())) == "state:1"
    assert str(np.asarray(view.handle("status").materialize())) == "nonphysical"
    assert json.loads(
        str(np.asarray(view.handle("method_metadata").materialize()))
    ) == {
        "method": "bdg",
        "note": "clipped",
    }
    np.testing.assert_allclose(
        np.asarray(view.handle("excess_noise").materialize()), 0.5
    )


# --- package layout --------------------------------------------------------


def test_data_package_grew_by_the_two_new_modules():
    from pathlib import Path

    package = Path(cam_data.__file__).parent
    for module in ("annotations", "continuation"):
        assert (package / f"{module}.py").exists(), module
    # The outcome vocabulary is shared with the solver contract, so it lives at
    # the package root and the data package re-exports it.
    assert (package.parent / "outcomes.py").exists()
    assert cam_data.CAMOutcomeStatus.VALID == "valid"
