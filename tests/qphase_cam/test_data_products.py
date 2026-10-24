"""Typed CAM products: schemas, builders, bundles and the legacy import.

The tests build small matrices directly instead of running a solver: the
product layer is about layout, identity and mapping, and Phase 10A already
freezes the physics end to end.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from qphase.data import (
    GenericDataBundle,
    ProductSchema,
    load_bundle,
    load_products,
    save_products,
)
from qphase_cam.bifurcation_result import (
    CAMBifurcationResult,
    CAMBifurcationScanResult,
)
from qphase_cam.data import (
    CAM_BUNDLE_TYPE_ID,
    CAM_CANDIDATE_PRODUCT,
    CAM_STATE_PRODUCT,
    CAM_STATE_SCAN_PRODUCT,
    build_candidate_product,
    build_state_product,
    cam_bundle,
    compare_candidates_export,
    compare_state_export,
    import_bifurcation_result,
    import_bifurcation_scan,
    import_cam_result,
    import_legacy_artifact,
)
from qphase_cam.engine import Engine
from qphase_cam.result import CAMResult
from qphase_cam.state import CAMBifurcationCandidate, CAMBifurcationOutput

# --- fixtures --------------------------------------------------------------


class _FoldModel:
    """Minimal model for packing candidate output without a dynamics adapter."""

    name = "fold_stub"
    n_modes = 2
    steady_state_capacity = 2
    params = {"gamma": -1.0, "Gamma": 1.0}


def _state_payload(counts, *, capacity=2, n_modes=2):
    """Return one fixed-capacity state payload for the given solution counts."""
    count_array = np.asarray(counts)
    shape = count_array.shape
    states = np.full(shape + (capacity, n_modes, n_modes), np.nan + 1j * np.nan)
    residuals = np.full(shape + (capacity,), np.nan)
    success = np.zeros(shape + (capacity,), dtype=bool)
    valid = np.zeros(shape + (capacity,), dtype=bool)
    for index in np.ndindex(shape):
        for slot in range(int(count_array[index])):
            offset = sum(index) + slot
            states[index + (slot,)] = np.eye(n_modes) * (offset + 1) + 0.5j
            residuals[index + (slot,)] = 1e-12 * (slot + 1)
            success[index + (slot,)] = True
            valid[index + (slot,)] = True
    return {
        "states": states,
        "residuals": residuals,
        "success": success,
        "valid_mask": valid,
        "solution_count": count_array.astype(np.int64),
    }


def _candidate_payload(*, count=2, n_modes=2, n_state=4):
    """Return one legacy candidate payload with ``count`` candidates."""
    vectors = np.arange(count * n_state, dtype=float).reshape(count, n_state) / 10.0
    return {
        "states": np.asarray(
            [np.eye(n_modes) * (position + 1) for position in range(count)],
            dtype=complex,
        ).reshape(count, n_modes, n_modes),
        "state_vectors": vectors,
        "control_values": np.full((count, 1), -0.5),
        "control_names": ("gamma",),
        "full_residual_norm": np.full(count, 1e-15),
        "search_residual_norm": np.full(count, 2e-15),
        "success": np.ones(count, dtype=bool),
        "status": ["verified"] * count,
        "method": ["fraction_free"] * count,
        "verification_digits": np.full(count, 50, dtype=np.int64),
        "verification_status": ["verified"] * count,
        "diagnostics": {
            "classification_accepted": np.ones(count, dtype=bool),
            "classification_status": np.asarray(["classified"] * count),
            "regularity_determinant": np.full(count, -0.25),
            "scaling_signatures": np.asarray([{"exponent": 0.5}] * count, dtype=object),
        },
        "meta": {"target": "equilibrium_multiplicity", "state_matrix_shape": (2, 2)},
    }


def _legacy_candidates(**overrides):
    payload = _candidate_payload(**overrides)
    return CAMBifurcationResult(
        states=payload["states"],
        state_vectors=payload["state_vectors"],
        control_values=payload["control_values"],
        control_names=payload["control_names"],
        full_residual_norm=payload["full_residual_norm"],
        search_residual_norm=payload["search_residual_norm"],
        success=payload["success"],
        status=np.asarray(payload["status"]),
        method=np.asarray(payload["method"]),
        verification_digits=payload["verification_digits"],
        verification_status=np.asarray(payload["verification_status"]),
        diagnostics=payload["diagnostics"],
        postprocess={"local_response_validation": {"candidate_index": np.asarray([0])}},
        meta=payload["meta"],
    )


def _identity_payload(count, *, offsets=None, control_names=("gamma",)):
    """Return builder keyword arguments for an all-valid candidate block."""
    return {
        "states": np.asarray(
            [np.eye(2) * (position + 1) for position in range(count)], dtype=complex
        ).reshape(count, 2, 2),
        "state_vectors": np.zeros((count, 4)),
        "control_values": np.full((count, len(control_names)), -0.5),
        "control_names": control_names,
        "full_residual_norm": np.full(count, 1e-15),
        "search_residual_norm": np.full(count, 2e-15),
        "success": np.ones(count, dtype=bool),
        "status": ["verified"] * count,
        "method": ["fraction_free"] * count,
        "verification_digits": np.full(count, 50, dtype=np.int64),
        "verification_status": ["verified"] * count,
        "candidate_identity": [f"new:candidate:{index}" for index in range(count)],
    }


def _no_object_dtypes(product):
    """Assert no variable of a product can carry a Python object."""
    for variable in product.variables:
        assert not np.dtype(variable.dtype).hasobject, variable.name
        array = np.asarray(product.handle(variable.name).materialize())
        assert not array.dtype.hasobject, variable.name


# --- templates -------------------------------------------------------------


def test_schema_templates_keep_their_axes_open():
    """A template is a plan: the builder, not the constant, closes the axes."""
    for template in (
        CAM_STATE_PRODUCT,
        CAM_STATE_SCAN_PRODUCT,
        CAM_CANDIDATE_PRODUCT,
    ):
        assert isinstance(template, ProductSchema)
        assert template.is_closed is False
        assert all(axis.size is None for axis in template.axes)
    assert [axis.name for axis in CAM_STATE_PRODUCT.axes] == [
        "scan",
        "solution",
        "state_row",
        "state_col",
    ]
    assert [axis.name for axis in CAM_CANDIDATE_PRODUCT.axes] == [
        "scan",
        "group",
        "group_offset",
        "candidate",
        "state_row",
        "state_col",
        "state_vector",
        "control",
    ]
    assert CAM_STATE_PRODUCT.attributes["product"] == "state"
    assert CAM_STATE_SCAN_PRODUCT.attributes["product"] == "state_scan"
    assert CAM_CANDIDATE_PRODUCT.attributes["product"] == "candidates"


def test_state_template_states_the_solution_axis_is_a_slot():
    """The artifact itself must not be readable as cross-parameter branches."""
    semantics = CAM_STATE_PRODUCT.attributes["solution_semantics"]
    assert "not a branch identity" in semantics
    assert "continuity" in semantics
    assert CAM_STATE_PRODUCT.attributes["ragged"] is False
    assert CAM_CANDIDATE_PRODUCT.attributes["ragged"] is True


# --- state product ---------------------------------------------------------


def test_build_state_product_closes_the_fixed_capacity_layout():
    payload = _state_payload(2)
    product = build_state_product(**payload, parameters={"gamma": 0.25})
    assert isinstance(product.attributes, dict)
    assert product.attributes["product"] == "state"
    assert product.attributes["grid_shape"] == []
    assert product.attributes["scan_size"] == 1
    assert product.attributes["solution_capacity"] == 2
    assert product.schema.is_closed
    assert product.axis("scan").size == 1
    assert product.axis("solution").size == 2
    assert product.axis("state_row").size == 2
    assert product.shape["states"] == (1, 2, 2, 2)
    assert product.shape["solution_count"] == (1,)
    _no_object_dtypes(product)


def test_build_state_product_flattens_a_multi_dimensional_grid():
    payload = _state_payload([[2, 1], [0, 2]])
    product = build_state_product(**payload)
    assert product.attributes["product"] == "state_scan"
    assert product.attributes["grid_shape"] == [2, 2]
    assert product.attributes["scan_size"] == 4
    assert product.axis("scan").size == 4
    assert product.shape["states"] == (4, 2, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(product.handle("solution_count").materialize()),
        [2, 1, 0, 2],
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("valid_mask").materialize()).sum(axis=-1),
        [2, 1, 0, 2],
    )


def test_build_state_product_rejects_a_count_mask_mismatch():
    payload = _state_payload(2)
    payload["valid_mask"] = np.zeros_like(payload["valid_mask"])
    with pytest.raises(ValueError, match="exactly solution_count slots"):
        build_state_product(**payload)


def test_build_state_product_rejects_an_out_of_range_count():
    payload = _state_payload(1)
    payload["solution_count"] = np.asarray(3, dtype=np.int64)
    payload["valid_mask"] = np.ones_like(payload["valid_mask"])
    with pytest.raises(ValueError, match=r"solution_count must lie in \[0, 2\]"):
        build_state_product(**payload)


def test_build_state_product_rejects_a_mismatched_solution_axis():
    payload = _state_payload(2)
    payload["residuals"] = payload["residuals"][..., :1]
    with pytest.raises(ValueError, match="states shape"):
        build_state_product(**payload)


def test_state_product_carries_explicit_parameter_coordinates():
    payload = _state_payload([1, 1])
    product = build_state_product(
        **payload,
        parameters={"gamma": np.asarray([0.0, 0.5]), "fixed": 2.0},
        axes={"detuning": np.asarray([1.0, 2.0])},
    )
    coordinates = {item.name: item for item in product.coordinates()}
    assert set(coordinates) == {"gamma", "fixed", "detuning"}
    assert coordinates["gamma"].role == "parameter"
    assert coordinates["gamma"].variable == "parameter.gamma"
    assert coordinates["fixed"].variable == "parameter.fixed"
    assert coordinates["detuning"].variable == "axis.detuning"
    np.testing.assert_allclose(product.coordinate("gamma"), [0.0, 0.5])
    np.testing.assert_allclose(product.coordinate("fixed"), [2.0, 2.0])
    np.testing.assert_allclose(product.coordinate("detuning"), [1.0, 2.0])


def test_state_product_reports_a_parameter_it_cannot_reshape():
    payload = _state_payload([1, 1])
    product = build_state_product(
        **payload, parameters={"matrix": np.eye(2), "gamma": np.asarray([0.0, 0.5])}
    )
    assert product.provenance["unmapped_parameter_names"] == ["matrix"]
    assert "parameter.matrix" not in {variable.name for variable in product.variables}


# --- candidate product -----------------------------------------------------


def test_build_candidate_product_normalizes_one_case_to_a_scan_of_one():
    payload = _identity_payload(2)
    product = build_candidate_product(**payload)
    assert product.attributes["case_count"] == 1
    assert product.attributes["case_shape"] == [1]
    assert product.attributes["candidate_count"] == 2
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_offsets").materialize()), [0, 2]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("group_scan_index").materialize()), [0]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_identity").materialize()),
        ["new:candidate:0", "new:candidate:1"],
    )
    assert product.axis("candidate").size == 2
    assert product.axis("group_offset").size == 2
    _no_object_dtypes(product)


def test_build_candidate_product_keeps_empty_cases():
    payload = _identity_payload(2)
    payload["candidate_offsets"] = np.asarray([0, 1, 1, 2])
    product = build_candidate_product(
        **payload,
        case_shape=(3,),
        case_metadata=({"case": 0}, {"case": 1}, {"case": 2}),
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_offsets").materialize()), [0, 1, 1, 2]
    )
    assert product.attributes["case_count"] == 3
    assert product.attributes["case_shape"] == [3]
    metadata = np.asarray(product.handle("case_metadata").materialize())
    assert [json.loads(str(value)) for value in metadata] == [
        {"case": 0},
        {"case": 1},
        {"case": 2},
    ]


def test_build_candidate_product_allows_a_scan_without_any_candidate():
    product = build_candidate_product(
        **_identity_payload(0),
        candidate_offsets=np.asarray([0, 0, 0]),
        case_shape=(2,),
    )
    assert product.attributes["candidate_count"] == 0
    assert product.attributes["case_count"] == 2
    assert product.axis("candidate").size == 0
    assert product.shape["states"] == (0, 2, 2)
    _no_object_dtypes(product)


@pytest.mark.parametrize(
    "identity, match",
    [
        (["a", ""], "is empty"),
        (["a", "a"], "must be unique"),
    ],
)
def test_candidate_identity_must_be_explicit_and_unique(identity, match):
    payload = _identity_payload(2)
    payload["candidate_identity"] = identity
    with pytest.raises(ValueError, match=match):
        build_candidate_product(**payload)


@pytest.mark.parametrize(
    "offsets, match",
    [
        ([1, 2], "must start at 0"),
        ([0, 2, 1, 2], "monotonic"),
        ([0, 1], "must end at the candidate count"),
        ([0], "one-dimensional with n_case"),
    ],
)
def test_candidate_offsets_are_validated(offsets, match):
    payload = _identity_payload(2)
    payload["candidate_offsets"] = np.asarray(offsets)
    payload["case_shape"] = (len(offsets) - 1,)
    with pytest.raises(ValueError, match=match):
        build_candidate_product(**payload)


def test_candidate_offsets_must_cover_the_declared_case_shape():
    payload = _identity_payload(2)
    payload["candidate_offsets"] = np.asarray([0, 2, 2])
    with pytest.raises(ValueError, match="does not cover the 2 scan cases"):
        build_candidate_product(**payload, case_shape=(3,))


def test_candidate_json_columns_are_canonical_json_text():
    payload = _identity_payload(2)
    product = build_candidate_product(
        **payload,
        classification=[{"accepted": True}, {"accepted": False}],
        base_audit=[{"reason": "ok", "value": float("nan")}, {"reason": "low"}],
    )
    classification = np.asarray(product.handle("classification").materialize())
    assert list(classification) == ['{"accepted":true}', '{"accepted":false}']
    audit = np.asarray(product.handle("base_audit").materialize())
    assert json.loads(str(audit[0])) == {"reason": "ok", "value": None}
    # Defaults are an empty object, never an empty string.
    assert list(np.asarray(product.handle("case_metadata").materialize())) == ["{}"]


def test_candidate_columns_must_cover_the_candidate_axis():
    payload = _identity_payload(2)
    payload["status"] = ["verified"]
    with pytest.raises(ValueError, match="status has 1 entries for 2 candidates"):
        build_candidate_product(**payload)


# --- bundles and persistence ----------------------------------------------


def _assert_roundtrip(bundle, *, tmp_path, layout):
    root = tmp_path / layout
    report = bundle.save_dataset(root, layout=layout, shard_target_bytes=256)
    assert report.layout == layout
    assert report.files
    products = load_products(root)
    assert set(products) == set(bundle.products)
    restored = load_bundle(root)
    assert isinstance(restored, GenericDataBundle)
    assert restored.bundle_descriptor.type_id == CAM_BUNDLE_TYPE_ID
    assert restored.provenance
    return restored


def test_single_point_state_bundle_roundtrips(tmp_path):
    product = cam_bundle(
        {"state": build_state_product(**_state_payload(2), parameters={"gamma": 0.25})},
        provenance={"cam_task": "solve"},
    )
    restored = _assert_roundtrip(product, tmp_path=tmp_path, layout="single")
    assert restored.bundle_descriptor.product_roles == {"state": "state"}
    assert restored.shape == (1,)
    reloaded = restored.require("state").materialize()
    assert reloaded.attributes["product"] == "state"
    assert reloaded.attributes["grid_shape"] == []
    np.testing.assert_array_equal(
        np.asarray(reloaded.handle("solution_count").materialize()), [2]
    )
    np.testing.assert_allclose(reloaded.coordinate("gamma"), [0.25])


def test_state_bundle_roundtrips_in_both_layouts(tmp_path):
    payload = _state_payload([2, 0, 1])
    product = cam_bundle(
        {"state_scan": build_state_product(**payload, parameters={"gamma": 0.5})},
        provenance={"cam_task": "scan"},
    )
    for layout in ("single", "sharded"):
        restored = _assert_roundtrip(product, tmp_path=tmp_path, layout=layout)
        assert restored.bundle_descriptor.product_roles == {"state": "state_scan"}
        reloaded = restored.require("state_scan").materialize()
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("solution_count").materialize()), [2, 0, 1]
        )
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("valid_mask").materialize()).sum(axis=-1),
            [2, 0, 1],
        )
        assert reloaded.attributes["grid_shape"] == [3]
        # Parameter coordinates survive as typed columns, not as attributes.
        np.testing.assert_allclose(reloaded.coordinate("gamma"), [0.5, 0.5, 0.5])
        _no_object_dtypes(reloaded)


def test_state_bundle_exposes_scan_axes_and_point_views(tmp_path):
    payload = _state_payload([1, 1])
    product = cam_bundle(
        {
            "state_scan": build_state_product(
                **payload, parameters={"gamma": np.asarray([0.0, 0.5])}
            )
        },
        provenance={"cam_task": "scan"},
    )
    restored = _assert_roundtrip(product, tmp_path=tmp_path, layout="single")
    assert restored.shape == (2,)
    np.testing.assert_allclose(restored.axes["scan"], [0.0, 0.5])
    view = restored.point_view((1,))
    assert view.shape == ()
    np.testing.assert_array_equal(
        np.asarray(view.require("state_scan").handle("solution_count").materialize()),
        [1],
    )


def test_candidate_bundle_roundtrips_in_both_layouts(tmp_path):
    payload = _identity_payload(2)
    payload["candidate_offsets"] = np.asarray([0, 1, 2])
    bundle = cam_bundle(
        {
            "candidates": build_candidate_product(
                **payload,
                case_shape=(2,),
                case_axes={"gamma": np.asarray([0.0, 0.5])},
                case_params={"gamma": np.asarray([0.0, 0.5]), "g": 0.5},
                classification=[{"accepted": True}, {"accepted": True}],
            )
        },
        provenance={"cam_task": "search"},
    )
    for layout in ("single", "sharded"):
        restored = _assert_roundtrip(bundle, tmp_path=tmp_path, layout=layout)
        assert restored.bundle_descriptor.product_roles == {"candidates": "candidates"}
        reloaded = restored.require("candidates").materialize()
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("candidate_offsets").materialize()), [0, 1, 2]
        )
        np.testing.assert_allclose(restored.axes["scan"], [0.0, 0.5])
        np.testing.assert_allclose(reloaded.coordinate("gamma"), [0.0, 0.5])
        np.testing.assert_allclose(
            np.asarray(reloaded.handle("case_parameter.g").materialize()), [0.5, 0.5]
        )
        assert json.loads(
            str(np.asarray(reloaded.handle("classification").materialize())[0])
        ) == {"accepted": True}
        _no_object_dtypes(reloaded)


# --- legacy import ---------------------------------------------------------


def _legacy_state():
    payload = _state_payload([2, 1])
    return CAMResult(
        **payload,
        params={
            "gamma": np.asarray([0.0, 0.5]),
            "Gamma": 1.0,
            "operator": np.eye(2),
        },
        axes={"detuning": np.asarray([1.0, 2.0])},
        postprocess={"jacobian_source": np.asarray(["analytic", "analytic"])},
        meta={"label": "run-7", "engine": "cam"},
    )


def test_import_cam_result_maps_every_field_and_records_unmapped_keys():
    legacy = _legacy_state()
    bundle = import_cam_result(legacy)
    assert list(bundle.products) == ["state_scan"]
    provenance = bundle.provenance
    assert provenance["legacy_source"] == "qphase_cam.result:CAMResult"
    assert provenance["legacy_result_kind"] == "fixed_points"
    # Nothing is dropped silently: the postprocess table and the operator
    # parameter are named, not mapped into a typed column.
    assert provenance["unmapped_legacy_postprocess"] == ["jacobian_source"]
    assert provenance["unmapped_legacy_parameters"] == ["operator"]
    assert provenance["unmapped_legacy_axes"] == []
    assert provenance["legacy_meta"]["label"] == "run-7"
    report = compare_state_export(bundle.require("state_scan"), legacy)
    assert report["passed"], report["mismatches"]
    assert report["checks"]["states"]["max_abs_error"] == 0.0


def test_import_cam_result_reads_a_trusted_npz(tmp_path):
    legacy = _legacy_state()
    target = tmp_path / "legacy_state"
    legacy.save(target)
    assert target.with_suffix(".npz").exists()
    bundle = import_legacy_artifact(target)
    assert list(bundle.products) == ["state_scan"]
    report = compare_state_export(bundle.require("state_scan"), legacy)
    assert report["passed"], report["mismatches"]


def test_import_bifurcation_result_normalizes_one_case_to_a_scan():
    legacy = _legacy_candidates()
    bundle = import_bifurcation_result(legacy)
    product = bundle.require("candidates")
    assert product.attributes["case_count"] == 1
    assert product.attributes["case_shape"] == [1]
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_identity").materialize()),
        ["legacy:bifurcation_candidates:0:0", "legacy:bifurcation_candidates:0:1"],
    )
    assert (
        bundle.provenance["legacy_source"]
        == "qphase_cam.bifurcation_result:CAMBifurcationResult"
    )
    # Object-array diagnostics and the postprocess table are reported by name.
    assert bundle.provenance["unmapped_legacy_diagnostics"] == ["scaling_signatures"]
    assert bundle.provenance["unmapped_legacy_postprocess"] == [
        "local_response_validation"
    ]
    classification = np.asarray(product.handle("classification").materialize())
    assert json.loads(str(classification[0])) == {
        "classification_accepted": True,
        "classification_status": "classified",
    }
    audit = json.loads(str(np.asarray(product.handle("base_audit").materialize())[1]))
    assert audit == {"regularity_determinant": -0.25}
    report = compare_candidates_export(product, legacy)
    assert report["passed"], report["mismatches"]


def test_import_bifurcation_scan_maps_offsets_coordinates_and_identity():
    candidates = CAMBifurcationResult.concatenate(
        [_legacy_candidates(), _legacy_candidates(count=1)]
    )
    scan = CAMBifurcationScanResult(
        case_axes={"gamma": np.asarray([0.0, 0.1, 0.2])},
        case_shape=(3,),
        case_params={
            "gamma": np.asarray([0.0, 0.1, 0.2]),
            "g": np.asarray([0.5, 0.5, 0.5]),
        },
        candidate_offsets=np.asarray([0, 2, 2, 3]),
        candidates=candidates,
        case_metadata=({"case": 0}, {"case": 1}, {"case": 2}),
        postprocess={"stochastic_validity": {"candidate_index": np.asarray([0])}},
        meta={"scan_shape": (3,), "label": "scan-1"},
    )
    bundle = import_bifurcation_scan(scan)
    product = bundle.require("candidates")
    assert product.attributes["case_count"] == 3
    assert product.attributes["case_shape"] == [3]
    # The middle case accepted nothing; its identity block is empty, not zero.
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_identity").materialize()),
        ["legacy:scan-1:0:0", "legacy:scan-1:0:1", "legacy:scan-1:2:0"],
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_offsets").materialize()), [0, 2, 2, 3]
    )
    np.testing.assert_allclose(
        np.asarray(product.handle("parameter.gamma").materialize()), [0.0, 0.1, 0.2]
    )
    np.testing.assert_allclose(
        np.asarray(product.handle("case_parameter.g").materialize()), [0.5, 0.5, 0.5]
    )
    metadata = np.asarray(product.handle("case_metadata").materialize())
    assert [json.loads(str(value)) for value in metadata] == [
        {"case": 0},
        {"case": 1},
        {"case": 2},
    ]
    assert bundle.provenance["unmapped_legacy_postprocess"] == ["stochastic_validity"]
    report = compare_candidates_export(product, scan)
    assert report["passed"], report["mismatches"]


def test_imported_candidate_identities_are_unique_across_cases():
    candidates = CAMBifurcationResult.concatenate(
        [_legacy_candidates(count=1), _legacy_candidates(count=1)]
    )
    scan = CAMBifurcationScanResult(
        case_axes={"gamma": np.asarray([0.0, 0.5])},
        case_shape=(2,),
        case_params={"gamma": np.asarray([0.0, 0.5])},
        candidate_offsets=np.asarray([0, 1, 2]),
        candidates=candidates,
        case_metadata=({}, {}),
    )
    product = import_bifurcation_scan(scan).require("candidates")
    identity = np.asarray(product.handle("candidate_identity").materialize())
    assert list(identity) == [
        "legacy:bifurcation_scan:0:0",
        "legacy:bifurcation_scan:1:0",
    ]
    assert len(set(identity)) == len(identity)


def test_import_legacy_artifact_dispatches_on_the_payload(tmp_path):
    candidates = _legacy_candidates()
    target = tmp_path / "legacy_candidates"
    candidates.save(target)
    bundle = import_legacy_artifact(target)
    assert list(bundle.products) == ["candidates"]
    assert bundle.require("candidates").attributes["case_count"] == 1
    unknown = tmp_path / "unknown"
    np.savez_compressed(unknown, foo=np.asarray([1.0]))
    with pytest.raises(ValueError, match="not a recognizable pre-2.0 CAM artifact"):
        import_legacy_artifact(unknown)


def test_imported_bundle_persists_the_legacy_provenance(tmp_path):
    bundle = import_cam_result(_legacy_state())
    root = tmp_path / "artifact"
    save_products(
        root,
        bundle.products,
        provenance=bundle.provenance,
        bundle=bundle.bundle_descriptor,
        layout="single",
    )
    restored = load_bundle(root)
    assert restored.provenance["legacy_source"] == "qphase_cam.result:CAMResult"
    assert restored.provenance["unmapped_legacy_postprocess"] == ["jacobian_source"]


# --- comparison helper -----------------------------------------------------


def test_comparison_reports_a_changed_value():
    """The helper is not vacuous: a corrupted column must be reported."""
    legacy = _legacy_state()
    product = import_cam_result(legacy).require("state_scan")
    arrays = {
        variable.name: np.asarray(product.handle(variable.name).materialize()).copy()
        for variable in product.variables
    }
    arrays["solution_count"] = np.asarray([2, 2])
    broken = type(product).from_arrays(
        product.schema,
        arrays,
        owner="test",
        provenance=product.provenance,
    )
    report = compare_state_export(broken, legacy)
    assert report["passed"] is False
    assert "solution_count" in report["mismatches"]
    assert "valid_mask_matches_solution_count" in report["mismatches"]


def test_candidate_comparison_reports_a_changed_classification():
    legacy = _legacy_candidates()
    product = import_bifurcation_result(legacy).require("candidates")
    arrays = {
        variable.name: np.asarray(product.handle(variable.name).materialize()).copy()
        for variable in product.variables
    }
    arrays["classification"] = np.asarray(['{"accepted":true}'] * 2, dtype="U32")
    broken = type(product).from_arrays(
        product.schema.model_copy(
            update={
                "variables": [
                    variable.model_copy(update={"dtype": "<U32"})
                    if variable.name == "classification"
                    else variable
                    for variable in product.schema.variables
                ]
            }
        ),
        arrays,
        owner="test",
    )
    report = compare_candidates_export(broken, legacy)
    assert report["passed"] is False
    assert "classification" in report["mismatches"]


# --- end-to-end from the engine -------------------------------------------


def test_engine_candidate_output_imports_and_round_trips(tmp_path):
    """A real packed result keeps its diagnostics through import and storage."""
    output = CAMBifurcationOutput(
        candidates=[
            CAMBifurcationCandidate(
                state_vector=np.asarray([1.0, 0.5, 0.0, 0.0]),
                controls={"gamma": -0.5},
                full_residual_norm=1e-15,
                search_residual_norm=1e-15,
                success=True,
                status="verified",
                method="test",
                metadata={
                    "classification_accepted": True,
                    "classification_status": "classified",
                    "regularity_determinant": -0.25,
                },
            ),
            CAMBifurcationCandidate(
                state_vector=np.asarray([2.0, 0.5, 0.0, 0.0]),
                controls={"gamma": -0.25},
                full_residual_norm=1e-14,
                search_residual_norm=1e-14,
                success=True,
                status="verified",
                method="test",
                metadata={
                    "classification_accepted": False,
                    "classification_status": "rejected",
                    "regularity_determinant": -0.5,
                },
            ),
        ],
        target="equilibrium_multiplicity",
        order=2,
        metadata={"control_names": ("gamma",)},
    )
    result = Engine._pack_bifurcation(output, _FoldModel())
    bundle = import_bifurcation_result(result)
    product = bundle.require("candidates")
    statuses = [
        json.loads(str(value))["classification_status"]
        for value in np.asarray(product.handle("classification").materialize())
    ]
    assert statuses == ["classified", "rejected"]
    restored = _assert_roundtrip(bundle, tmp_path=tmp_path, layout="sharded")
    reloaded = restored.require("candidates").materialize()
    assert restored.provenance["unmapped_legacy_branches"] == []
    reported = compare_candidates_export(reloaded, result)
    assert reported["passed"], reported["mismatches"]
