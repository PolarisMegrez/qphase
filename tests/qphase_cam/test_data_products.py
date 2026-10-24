"""Typed CAM products: schemas, builders, bundles and the legacy import.

The tests build small matrices directly instead of running a solver: the
product layer is about layout, identity and mapping, and Phase 10A already
freezes the physics end to end.
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

#: Imported candidate identity: ``legacy:<16 hex digits>:<case>:<candidate>``.
LEGACY_IDENTITY = re.compile(
    r"^legacy:(?P<fingerprint>[0-9a-f]{16}):(?P<case>\d+):(?P<slot>\d+)$"
)

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


def _scaling_signatures(count, *, control=-0.5):
    """Return one legacy ``scaling_signatures`` object array per candidate.

    This is the shape the solver really packs: an object array whose elements
    are tuples of dicts mixing ints, floats, bools and nested tuples. It is the
    classification evidence, so it must reach the product.
    """
    return np.asarray(
        [
            (
                {
                    "state_order": 3 + index,
                    "perturbation_order": 1,
                    "coupling_state_order": 0,
                    "exponent_numerator": 1,
                    "exponent_denominator": 3,
                    "exponent": 1.0 / 3.0,
                    "sublinear": True,
                    "coefficient": complex(0.5, control),
                    "branches": (),
                },
            )
            for index in range(count)
        ],
        dtype=object,
    )


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
            "scaling_signatures": _scaling_signatures(count),
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


def _identities(product):
    """Return the imported candidate identities of a candidate product."""
    return [str(value) for value in product.handle("candidate_identity").materialize()]


def _identity_tuples(product):
    """Return ``(fingerprint, case, slot)`` for every imported identity."""
    parsed = []
    for value in _identities(product):
        match = LEGACY_IDENTITY.match(value)
        assert match is not None, value
        parsed.append((match["fingerprint"], int(match["case"]), int(match["slot"])))
    return parsed


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
    parsed = _identity_tuples(product)
    assert [(case, slot) for _, case, slot in parsed] == [(0, 0), (0, 1)]
    assert len({fingerprint for fingerprint, _, _ in parsed}) == 1
    assert (
        bundle.provenance["legacy_source"]
        == "qphase_cam.bifurcation_result:CAMBifurcationResult"
    )
    assert bundle.provenance["legacy_source_fingerprint"] == parsed[0][0]
    # The postprocess table is reported by name; the object-array
    # classification evidence is mapped, not refused.
    assert bundle.provenance["unmapped_legacy_diagnostics"] == []
    assert bundle.provenance["unmapped_legacy_postprocess"] == [
        "local_response_validation"
    ]
    classification = json.loads(
        str(np.asarray(product.handle("classification").materialize())[0])
    )
    assert classification["classification_accepted"] is True
    assert classification["classification_status"] == "classified"
    assert classification["scaling_signatures"] == [
        {
            "state_order": 3,
            "perturbation_order": 1,
            "coupling_state_order": 0,
            "exponent_numerator": 1,
            "exponent_denominator": 3,
            "exponent": 1.0 / 3.0,
            "sublinear": True,
            "coefficient": {"real": 0.5, "imag": -0.5},
            "branches": [],
        }
    ]
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
    parsed = _identity_tuples(product)
    assert [(case, slot) for _, case, slot in parsed] == [(0, 0), (0, 1), (2, 0)]
    assert len({fingerprint for fingerprint, _, _ in parsed}) == 1
    assert bundle.provenance["legacy_source_fingerprint"] == parsed[0][0]
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
    parsed = _identity_tuples(product)
    assert [(case, slot) for _, case, slot in parsed] == [(0, 0), (1, 0)]
    identity = _identities(product)
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


# --- imported identity is a content fingerprint ---------------------------


def test_identity_fingerprint_is_stable_for_the_same_source():
    """Re-importing one snapshot twice must not invent new identities."""
    legacy = _legacy_candidates()
    first = _identities(import_bifurcation_result(legacy).require("candidates"))
    second = _identities(import_bifurcation_result(legacy).require("candidates"))
    assert first == second
    assert (
        len(
            {
                fingerprint
                for fingerprint, _, _ in _identity_tuples(
                    import_bifurcation_result(legacy).require("candidates")
                )
            }
        )
        == 1
    )


def test_identity_fingerprint_matches_between_object_and_npz(tmp_path):
    """The same content imported from memory and from disk agrees."""
    legacy = _legacy_candidates()
    target = tmp_path / "legacy"
    legacy.save(target)
    from_object = _identities(import_bifurcation_result(legacy).require("candidates"))
    from_npz = _identities(import_bifurcation_result(target).require("candidates"))
    assert from_object == from_npz


def test_two_different_sources_do_not_collide():
    """Two results that differ only in their control values stay distinct."""
    left = _legacy_candidates()
    right_payload = _candidate_payload()
    right_payload["control_values"] = np.full((2, 1), -0.25)
    right = CAMBifurcationResult(
        states=right_payload["states"],
        state_vectors=right_payload["state_vectors"],
        control_values=right_payload["control_values"],
        control_names=right_payload["control_names"],
        full_residual_norm=right_payload["full_residual_norm"],
        search_residual_norm=right_payload["search_residual_norm"],
        success=right_payload["success"],
        status=np.asarray(right_payload["status"]),
        method=np.asarray(right_payload["method"]),
        verification_digits=right_payload["verification_digits"],
        verification_status=np.asarray(right_payload["verification_status"]),
        diagnostics=right_payload["diagnostics"],
        meta=right_payload["meta"],
    )
    left_products = import_bifurcation_result(left).require("candidates")
    right_products = import_bifurcation_result(right).require("candidates")
    left_identity = _identities(left_products)
    right_identity = _identities(right_products)
    assert set(left_identity).isdisjoint(right_identity)
    left_token = _identity_tuples(left_products)[0][0]
    right_token = _identity_tuples(right_products)[0][0]
    assert left_token != right_token


def test_two_scan_sources_with_identical_cases_do_not_collide():
    """Two scans share case indices; only the fingerprint may separate them."""

    def scan(offset: float) -> CAMBifurcationScanResult:
        return CAMBifurcationScanResult(
            case_axes={"gamma": np.asarray([0.0 + offset, 0.5 + offset])},
            case_shape=(2,),
            case_params={"gamma": np.asarray([0.0 + offset, 0.5 + offset])},
            candidate_offsets=np.asarray([0, 1, 2]),
            candidates=CAMBifurcationResult.concatenate(
                [_legacy_candidates(count=1), _legacy_candidates(count=1)]
            ),
            case_metadata=({}, {}),
        )

    left = import_bifurcation_scan(scan(0.0)).require("candidates")
    right = import_bifurcation_scan(scan(0.1)).require("candidates")
    assert set(_identities(left)).isdisjoint(_identities(right))
    assert [case for _, case, _ in _identity_tuples(left)] == [0, 1]
    assert [case for _, case, _ in _identity_tuples(right)] == [0, 1]


def test_identity_records_how_the_token_was_obtained():
    legacy = _legacy_candidates()
    derived = import_bifurcation_result(legacy)
    assert derived.provenance["legacy_source_token"] == "content_fingerprint"
    named = import_bifurcation_result(legacy, source_id="artifact-7")
    assert named.provenance["legacy_source_token"] == "source_id"
    assert _identities(named.require("candidates")) == [
        "legacy:artifact-7:0:0",
        "legacy:artifact-7:0:1",
    ]
    with pytest.raises(ValueError, match="at least one usable character"):
        import_bifurcation_result(legacy, source_id="   ")


# --- legacy diagnostics keep their evidence -------------------------------


def test_scaling_signatures_are_mapped_not_flagged_unmapped():
    """The object-array classification evidence must survive the import."""
    legacy = _legacy_candidates()
    bundle = import_bifurcation_result(legacy)
    product = bundle.require("candidates")
    assert "scaling_signatures" not in bundle.provenance["unmapped_legacy_diagnostics"]
    rows = [
        json.loads(str(value))
        for value in np.asarray(product.handle("classification").materialize())
    ]
    # Per-candidate slicing is what proves the mapping: each row carries the
    # signature of its own candidate, not candidate 0's repeated.
    assert [row["scaling_signatures"][0]["state_order"] for row in rows] == [3, 4]
    signature = rows[0]["scaling_signatures"][0]
    assert signature["exponent_denominator"] == 3
    assert signature["sublinear"] is True
    assert signature["coefficient"] == {"real": 0.5, "imag": -0.5}
    assert rows[1]["scaling_signatures"][0]["coefficient"] == {
        "real": 0.5,
        "imag": -0.5,
    }


def test_only_values_without_a_json_form_are_reported_unmapped():
    """A scalar diagnostic and a set argument stay reported, by name."""
    payload = _candidate_payload()
    payload["diagnostics"]["summary_scalar"] = 1.5
    payload["diagnostics"]["not_jsonable"] = np.asarray([{1, 2}, {3}], dtype=object)
    legacy = CAMBifurcationResult(
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
        meta=payload["meta"],
    )
    bundle = import_bifurcation_result(legacy)
    assert bundle.provenance["unmapped_legacy_diagnostics"] == [
        "not_jsonable[0]",
        "not_jsonable[1]",
        "summary_scalar",
    ]
    product = bundle.require("candidates")
    assert "scaling_signatures" in json.loads(
        str(np.asarray(product.handle("classification").materialize())[0])
    )


# --- point_view -----------------------------------------------------------


def _candidate_case_metadata(product):
    """Return the per-case JSON metadata of a candidate product.

    A point view drops the ``scan`` axis, so the column may be 0-d.
    """
    values = np.atleast_1d(np.asarray(product.handle("case_metadata").materialize()))
    return [json.loads(str(value)) for value in values]


def test_point_view_reads_identity_json_and_parameters(tmp_path):
    """`point_view(scan=i)` works on a candidate product, runtime and stored."""
    payload = _identity_payload(2)
    payload["candidate_offsets"] = np.asarray([0, 1, 2])
    product = build_candidate_product(
        **payload,
        case_shape=(2,),
        case_axes={"gamma": np.asarray([0.0, 0.5])},
        classification=[{"accepted": True}, {"accepted": False}],
        case_metadata=[{"case": 0}, {"case": 1}],
    )
    runtime = product.point_view(scan=0)
    assert _identities(runtime) == ["new:candidate:0", "new:candidate:1"]
    assert json.loads(
        str(np.asarray(runtime.handle("classification").materialize())[0])
    ) == {"accepted": True}
    np.testing.assert_allclose(
        np.asarray(runtime.handle("parameter.gamma").materialize()), 0.0
    )
    assert (
        str(np.asarray(runtime.handle("case_metadata").materialize())) == '{"case":0}'
    )

    bundle = cam_bundle({"candidates": product}, provenance={"cam_task": "search"})
    for layout in ("single", "sharded"):
        restored = _assert_roundtrip(bundle, tmp_path=tmp_path, layout=layout)
        view = restored.point_view((1,))
        selected = view.require("candidates")
        # The scan axis is dropped, so the case metadata and the parameter
        # coordinate become scalars. The candidate axis is untouched: a point
        # view never re-computes a ragged slice, it reports the same rows.
        assert _identities(selected) == ["new:candidate:0", "new:candidate:1"]
        classification = [
            json.loads(str(value))
            for value in np.asarray(selected.handle("classification").materialize())
        ]
        assert classification == [{"accepted": True}, {"accepted": False}]
        np.testing.assert_allclose(
            np.asarray(selected.handle("parameter.gamma").materialize()), 0.5
        )
        assert _candidate_case_metadata(selected) == [{"case": 1}]


def test_point_view_does_not_filter_candidates_per_case():
    """A point view selects the scan axis only; offsets still scope candidates."""
    payload = _identity_payload(2)
    payload["candidate_offsets"] = np.asarray([0, 2, 2, 2])
    product = build_candidate_product(**payload, case_shape=(3,))
    view = product.point_view(scan=2)
    # Case 2 accepted nothing, but the candidate block is unchanged: a point
    # view must not silently re-compute a ragged slice, so a consumer reads
    # candidate_offsets to scope candidates to a case.
    assert len(_identities(view)) == 2
    np.testing.assert_array_equal(
        np.asarray(view.handle("candidate_offsets").materialize()), [0, 2, 2, 2]
    )
    assert product.attributes["case_count"] == 3


def test_point_view_of_a_shorter_case_metadata_string():
    """Both ends of a variable-width case_metadata column stay readable.

    Per-case JSON texts differ in length, so the ``U`` column is as wide as the
    longest one. Selecting the shorter case used to fail the handle dtype check;
    the declared width is now re-applied, and the selected text, dtype and dims
    are all preserved.
    """
    product = build_candidate_product(
        **_identity_payload(2),
        candidate_offsets=np.asarray([0, 1, 2]),
        case_shape=(2,),
        case_metadata=[{"k": 1}, {"k": 1234567890}],
    )
    assert product.variable("case_metadata").dtype == "<U16"

    short = product.point_view(scan=0)
    assert short.variable("case_metadata").dims == ()
    assert short.variable("case_metadata").dtype == "<U16"
    selected = np.asarray(short.handle("case_metadata").materialize())
    assert selected.dtype == np.dtype("<U16")
    assert json.loads(str(selected)) == {"k": 1}

    longest = product.point_view(scan=1)
    assert longest.variable("case_metadata").dtype == "<U16"
    assert json.loads(
        str(np.asarray(longest.handle("case_metadata").materialize()))
    ) == {"k": 1234567890}
    # The rest of the block is untouched by the selection either side.
    assert _identities(short) == ["new:candidate:0", "new:candidate:1"]


def test_slice_view_keeps_a_short_case_metadata_column():
    """A slice keeps the declared width too, not just a point view."""
    product = build_candidate_product(
        **_identity_payload(2),
        candidate_offsets=np.asarray([0, 1, 2]),
        case_shape=(2,),
        case_metadata=[{"k": 1}, {"k": 1234567890}],
    )
    view = product.slice_view(scan=slice(0, 1))
    column = np.asarray(view.handle("case_metadata").materialize())
    assert column.dtype == np.dtype("<U16")
    assert [json.loads(str(value)) for value in column] == [{"k": 1}]


# --- package layout -------------------------------------------------------


def test_data_is_a_package_with_one_implementation():
    """`qphase_cam.data` is a package; no sibling data.py may shadow it."""
    from pathlib import Path

    package = Path(cam_data.__file__).parent
    assert package.name == "data"
    assert (package / "__init__.py").exists()
    assert not (package.parent / "data.py").exists()
    for module in (
        "_common",
        "bundle",
        "candidates",
        "compare",
        "declarations",
        "legacy",
        "state",
    ):
        assert (package / f"{module}.py").exists(), module


def test_public_api_is_reexported_not_implemented():
    """The package `__init__` declares the surface and defines none of it."""
    from pathlib import Path

    for name in cam_data.__all__:
        assert hasattr(cam_data, name), name
    source = Path(cam_data.__file__).read_text(encoding="utf-8")
    definitions = [
        line for line in source.splitlines() if line.startswith(("def ", "class "))
    ]
    assert definitions == []
