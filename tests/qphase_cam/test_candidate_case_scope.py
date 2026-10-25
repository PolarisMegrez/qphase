"""Per-case candidate scope: the case grid, its metadata and its parameters.

A candidate index and a case index are different index spaces. The historical path
that fed a candidate index to ``CAMBifurcationScanResult.params_at`` read another
case's parameters (or raised, on a single-case scan), and the typed path dropped the
per-case metadata the product already stores — including the perturbation the case
was searched along. These tests pin both: the case-grid addressing contract, the
per-candidate metadata/parameter resolution of :class:`CandidateView`, and the fact
that the two postprocessors and the annotation-target reader all use it.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from qphase_cam.bifurcation_result import (
    CAMBifurcationResult,
    CAMBifurcationScanResult,
)
from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.data import build_candidate_product
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.candidates import (
    PerturbationDesign,
    candidate_parameters,
    candidate_view,
    case_fixed_parameters,
)
from qphase_cam.postprocessor.closure_risk import ClosureRisk
from qphase_cam.postprocessor.local_response import LocalResponseValidation
from qphase_cam.postprocessor.stochastic_validity import StochasticValidity
from qphase_cam.postprocessor.targets import annotation_targets
from qphase_cam.solver.bifurcation import ControlRange

from .test_bifurcation import TwoModeFoldModel, _solver

# --- fixtures --------------------------------------------------------------


class _Model:
    """Minimal parameter container for the maps that never build an adapter."""

    name = "case_scope_stub"
    n_modes = 2
    steady_state_capacity = 2
    params = {"omega": 0.0, "gamma": 1.0, "g": 0.1}


def _candidates(count: int, *, controls: float = -0.5) -> CAMBifurcationResult:
    """Return ``count`` real candidate rows with a full real state tangent."""
    matrices = np.asarray(
        [np.eye(2) * (position + 1) for position in range(count)], dtype=complex
    )
    vectors = np.asarray([matrix_to_vector(item) for item in matrices], dtype=float)
    signature = {
        "state_order": 3,
        "perturbation_order": 1,
        "coupling_state_order": 0,
        "exponent_numerator": 1,
        "exponent_denominator": 3,
        "exponent": 1.0 / 3.0,
        "sublinear": True,
        "branches": (
            {
                "epsilon_side": 1,
                "amplitude": 1.0,
                "leading_state_coefficient": {
                    "real": [[1.0, 0.0], [0.0, 1.0]],
                    "imag": [[0.0, 0.0], [0.0, 0.0]],
                },
            },
        ),
    }
    return CAMBifurcationResult(
        states=matrices,
        state_vectors=vectors,
        control_values=np.full((count, 1), controls),
        control_names=("gamma",),
        full_residual_norm=np.zeros(count),
        search_residual_norm=np.full(count, 1e-14),
        success=np.ones(count, dtype=bool),
        status=np.asarray(["verified"] * count),
        method=np.asarray(["test"] * count),
        verification_digits=np.full(count, 50, dtype=np.int64),
        verification_status=np.asarray(["verified"] * count),
        diagnostics={
            "scaling_signatures": np.asarray([signature] * count, dtype=object)
        },
        meta={"perturbation_parameter": "omega", "perturbation_scale": 1.0},
    )


def _product(
    *,
    offsets: tuple[int, ...] = (0, 2, 3),
    case_metadata: tuple[dict, ...] | None = None,
    case_params: dict | None = None,
) -> object:
    """Return a typed candidate product with the given case layout."""
    count = int(offsets[-1])
    metadata = case_metadata or tuple(
        {
            "case_parameters": {"omega": 0.1 * case, "gamma": 1.0, "g": 0.1},
            "search_fragment": {
                "search_design": {
                    "perturbation_parameter": "omega",
                    "perturbation_scale": 1.0 + case,
                    "perturbation_side": "both",
                }
            },
        }
        for case in range(len(offsets) - 1)
    )
    return build_candidate_product(
        states=np.asarray(
            [np.eye(2) * (index + 1) for index in range(count)], dtype=complex
        ),
        state_vectors=np.zeros((count, 4)),
        control_values=np.full((count, 1), -0.5),
        control_names=("gamma",),
        full_residual_norm=np.zeros(count),
        search_residual_norm=np.full(count, 1e-14),
        success=np.ones(count, dtype=bool),
        status=["verified"] * count,
        method=["test"] * count,
        verification_digits=np.full(count, 50, dtype=np.int64),
        verification_status=["verified"] * count,
        candidate_identity=[f"cam:candidate:{index}" for index in range(count)],
        case_metadata=metadata,
        candidate_offsets=np.asarray(offsets),
        case_shape=(len(offsets) - 1,),
        case_params=(
            {"chi": np.full(len(offsets) - 1, 0.5)}
            if case_params is None
            else case_params
        ),
    )


# --- the case-grid addressing contract -------------------------------------


def test_a_case_is_addressed_by_its_grid_index_not_by_a_candidate_index():
    scan = CAMBifurcationScanResult(
        case_axes={"gamma": np.asarray([0.0, 0.5, 1.0])},
        case_shape=(3,),
        case_params={"gamma": np.asarray([0.0, 0.5, 1.0])},
        candidate_offsets=np.asarray([0, 0, 1, 2]),
        candidates=_candidates(2),
        case_metadata=({"case": 0}, {"case": 1}, {"case": 2}),
        meta={},
    )
    # Candidate 1 belongs to case 2: the offsets say case 0 and case 1 hold no
    # candidate before it.
    assert scan.case_flat_index(1) == 2
    assert scan.case_grid_index(1) == (2,)
    assert float(scan.params_at(scan.case_grid_index(1))["gamma"]) == 1.0
    assert float(scan.params_at((2,))["gamma"]) == 1.0
    view = scan.point_view(scan.case_grid_index(1))
    assert view.meta["case_index"] == 2
    assert view.meta["case_search"] == {"case": 2}
    assert len(view.states) == 1


def test_a_bad_case_index_fails_instead_of_reading_another_case():
    scan = CAMBifurcationScanResult(
        case_axes={"gamma": np.asarray([0.0, 0.5])},
        case_shape=(2,),
        case_params={"gamma": np.asarray([0.0, 0.5])},
        candidate_offsets=np.asarray([0, 1, 2]),
        candidates=_candidates(2),
        case_metadata=({"case": 0}, {"case": 1}),
        meta={},
    )
    with pytest.raises(IndexError, match="rank 2"):
        scan.params_at((0, 0))
    with pytest.raises(IndexError, match="out of bounds"):
        scan.params_at((5,))
    with pytest.raises(IndexError, match="out of bounds"):
        scan.params_at((-1,))
    with pytest.raises(TypeError, match="must be a tuple"):
        scan.params_at(1)
    with pytest.raises(IndexError, match="out of bounds for 2 candidates"):
        scan.case_grid_index(2)


def test_a_single_case_scan_is_addressed_by_the_empty_index():
    scan = CAMBifurcationScanResult(
        case_axes={},
        case_shape=(),
        case_params={"omega": np.asarray([0.25])},
        candidate_offsets=np.asarray([0, 2]),
        candidates=_candidates(2),
        case_metadata=({"case": 0},),
        meta={},
    )
    assert scan.case_count == 1
    assert scan.grid_index(0) == ()
    assert scan.case_grid_index(1) == ()
    assert float(scan.params_at(())["omega"]) == 0.25
    assert len(scan.point_view(()).states) == 2
    with pytest.raises(IndexError, match=r"rank 1 for a case grid of shape \(\)"):
        scan.params_at((0,))
    with pytest.raises(IndexError, match="out of bounds"):
        scan.grid_index(1)


def test_the_legacy_parameter_map_resolves_the_candidate_case():
    scan = CAMBifurcationScanResult(
        case_axes={"gamma": np.asarray([0.0, 0.5])},
        case_shape=(2,),
        case_params={"gamma": np.asarray([0.0, 0.5]), "omega": np.asarray([0.3, 0.7])},
        candidate_offsets=np.asarray([0, 0, 1]),
        candidates=_candidates(1),
        case_metadata=(
            {"fixed_params": {"omega": -1.0, "gamma": 9.0, "g": 0.1}},
            {"fixed_params": {"omega": -1.0, "gamma": 9.0, "g": 0.1}},
        ),
        meta={},
    )
    params = candidate_parameters(scan, 0, _Model())
    # The candidate is in case 1, whose case parameters and controls win over the
    # model-level and case-level fixed values.
    assert params["omega"] == 0.7
    assert params["gamma"] == -0.5
    assert params["g"] == 0.1


# --- the typed view --------------------------------------------------------


def test_every_case_keeps_its_own_base_parameters():
    view = candidate_view(
        _product(
            case_metadata=(
                {"case_parameters": {"omega": 0.1, "gamma": 2.0}},
                {"case_parameters": {"omega": 0.9, "gamma": 3.0}},
            )
        )
    )
    assert [row.case for row in view.rows] == [0, 0, 1]
    assert view.metadata_at(0)["case_parameters"]["omega"] == 0.1
    assert view.metadata_at(2)["case_parameters"]["omega"] == 0.9
    first = view.params_at(0)
    last = view.params_at(2)
    # The case base parameters differ per case and the controls still win.
    assert first["omega"] == pytest.approx(0.1)
    assert last["omega"] == pytest.approx(0.9)
    assert first["gamma"] == pytest.approx(-0.5)
    assert last["gamma"] == pytest.approx(-0.5)


def test_the_case_parameter_columns_win_over_the_case_metadata():
    view = candidate_view(
        _product(
            case_metadata=(
                {"case_parameters": {"omega": 0.1, "g": 0.2}},
                {"case_parameters": {"omega": 0.9, "g": 0.8}},
            ),
            case_params={"omega": np.asarray([0.25, 0.75])},
        )
    )
    assert view.params_at(0)["omega"] == pytest.approx(0.25)
    assert view.params_at(2)["omega"] == pytest.approx(0.75)
    # A parameter only the case metadata records still reaches the map.
    assert view.params_at(0)["g"] == pytest.approx(0.2)
    assert view.params_at(2)["g"] == pytest.approx(0.8)


def test_an_empty_case_shifts_the_case_scope_of_the_following_candidates():
    view = candidate_view(
        _product(
            offsets=(0, 1, 1, 3),
            case_metadata=(
                {"case_parameters": {"omega": 0.1}},
                {"case_parameters": {"omega": 0.2}},
                {"case_parameters": {"omega": 0.3}},
            ),
        )
    )
    assert [row.case for row in view.rows] == [0, 2, 2]
    assert view.params_at(0)["omega"] == pytest.approx(0.1)
    assert view.params_at(1)["omega"] == pytest.approx(0.3)
    assert view.params_at(2)["omega"] == pytest.approx(0.3)


def test_a_product_without_candidates_reports_an_out_of_bounds_candidate():
    view = candidate_view(
        _product(offsets=(0, 0, 0), case_metadata=({"case_parameters": {}},) * 2)
    )
    assert view.rows == []
    with pytest.raises(IndexError, match="out of bounds"):
        view.params_at(0)


def test_the_case_design_is_read_per_candidate():
    view = candidate_view(
        _product(
            case_metadata=(
                {
                    "case_parameters": {"omega": 0.1},
                    "search_fragment": {
                        "search_design": {
                            "perturbation_parameter": "omega",
                            "perturbation_scale": 2.5,
                            "perturbation_side": "positive",
                        }
                    },
                },
                {
                    "fixed_params": {"omega": 0.9},
                    "perturbation_parameter": "g",
                    "perturbation_scale": 0.5,
                },
            )
        )
    )
    first = view.design_at(0)
    last = view.design_at(2)
    assert (first.parameter, first.scale, first.side) == ("omega", 2.5, "positive")
    assert (last.parameter, last.scale) == ("g", 0.5)
    assert last.is_resolved


def test_execution_metadata_overrides_the_stored_case_design():
    view = candidate_view(
        _product(),
        metadata={"perturbation_parameter": "g", "perturbation_scale": 4.0},
    )
    design = view.design_at(0)
    assert (design.parameter, design.scale) == ("g", 4.0)


def test_a_missing_design_is_reported_instead_of_guessed():
    view = candidate_view(_product(case_metadata=({"case_parameters": {}},) * 2))
    assert not view.design_at(0).is_resolved


def test_case_fixed_parameters_reads_either_recorded_form():
    assert case_fixed_parameters({"case_parameters": {"omega": 1}}) == {"omega": 1.0}
    assert case_fixed_parameters({"fixed_params": {"omega": 2}}) == {"omega": 2.0}
    assert case_fixed_parameters({"case": 1}) == {}


def test_the_annotation_targets_use_the_same_parameter_map():
    product = _product(
        case_metadata=(
            {"case_parameters": {"omega": 0.1, "gamma": 2.0}},
            {"case_parameters": {"omega": 0.9, "gamma": 3.0}},
        )
    )
    view = candidate_view(product)
    targets = annotation_targets(
        product,
        CAMAnnotationContext(model=_Model(), metadata={}, source_role="candidates"),
        _Model(),
    )
    assert [target.identity for target in targets] == [
        row.identity for row in view.rows
    ]
    for target, row in zip(targets, view.rows, strict=True):
        expected = candidate_parameters(view, row.index, _Model())
        assert target.params == expected
    assert targets[0].params["omega"] != targets[-1].params["omega"]


# --- the postprocessors ----------------------------------------------------

_FOLD = TwoModeFoldModel()
_FOLD_PARAMS = {"gamma": -1.0, "Gamma": 1.0, "kappa": 1.0}


def _fold_output():
    """Return the real candidates of the shared two-mode fold."""
    from qphase.backend.numpy_backend import NumpyBackend

    return _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        _FOLD, NumpyBackend()
    )


def _two_case_product(*, designs: tuple[dict, ...] | None = None):
    """Return a product whose two cases carry their own perturbation design."""
    from qphase_cam.solver.execution import candidate_product_from_output

    single = candidate_product_from_output(_FOLD, _fold_output()).materialize()
    states = np.asarray(single.handle("states").materialize())
    vectors = np.asarray(single.handle("state_vectors").materialize())
    controls = np.asarray(single.handle("control_values").materialize())
    count = int(states.shape[0])
    resolved = designs or (
        {
            "case_parameters": _FOLD_PARAMS,
            "perturbation_parameter": "gamma",
            "perturbation_scale": 1.0,
        },
        {
            "case_parameters": _FOLD_PARAMS,
            "perturbation_parameter": "kappa",
            "perturbation_scale": 2.0,
        },
    )
    return build_candidate_product(
        states=np.concatenate([states, states]),
        state_vectors=np.concatenate([vectors, vectors]),
        control_values=np.concatenate([controls, controls]),
        control_names=("gamma",),
        full_residual_norm=np.concatenate(
            [np.asarray(single.handle("full_residual_norm").materialize())] * 2
        ),
        search_residual_norm=np.concatenate(
            [np.asarray(single.handle("search_residual_norm").materialize())] * 2
        ),
        success=np.concatenate(
            [np.asarray(single.handle("success").materialize())] * 2
        ),
        status=["verified"] * (2 * count),
        method=["test"] * (2 * count),
        verification_digits=np.zeros(2 * count, dtype=np.int64),
        verification_status=["not_run"] * (2 * count),
        candidate_identity=[f"cam:candidate:{index}" for index in range(2 * count)],
        classification=[
            json.loads(str(value))
            for value in np.asarray(single.handle("classification").materialize())
        ]
        * 2,
        base_audit=[
            json.loads(str(value))
            for value in np.asarray(single.handle("base_audit").materialize())
        ]
        * 2,
        case_metadata=resolved,
        candidate_offsets=np.asarray([0, count, 2 * count]),
        case_shape=(2,),
        case_params={"omega": np.asarray([0.1, 0.2])},
    )


def test_local_response_reads_each_case_perturbation_from_its_metadata():
    product = _two_case_product()
    view = candidate_view(product)
    processor = LocalResponseValidation(
        epsilon_min=1e-10,
        epsilon_max=1e-6,
        epsilon_points=4,
        fit_points=3,
        residual_tolerance=1e-25,
    )
    annotation = processor.process(
        product,
        CAMAnnotationContext(model=_FOLD, metadata={}, source_role="candidates"),
    )
    metadata = [
        json.loads(str(value))
        for value in np.asarray(annotation.handle("method_metadata").materialize())
    ]
    # The two cases were searched along different parameters, and the annotation
    # records each candidate's own design instead of one shared value.
    assert [item["perturbation_parameter"] for item in metadata] == [
        view.design_at(0).parameter,
        view.design_at(len(view.rows) - 1).parameter,
    ]
    assert metadata[0]["perturbation_parameter"] == "gamma"
    assert metadata[-1]["perturbation_parameter"] == "kappa"
    assert metadata[-1]["perturbation_scale"] == 2.0
    assert [
        item["parameter"]
        for item in processor.result_metadata["perturbation_designs"]
    ] == ["gamma", "kappa"]


def test_local_response_refuses_a_case_without_a_perturbation():
    product = _two_case_product(
        designs=({"case_parameters": _FOLD_PARAMS}, {"case_parameters": _FOLD_PARAMS})
    )
    with pytest.raises(ValueError, match="records no perturbation parameter"):
        LocalResponseValidation().process(
            product,
            CAMAnnotationContext(model=_FOLD, metadata={}, source_role="candidates"),
        )


def test_stochastic_validity_reads_the_perturbation_from_the_case_metadata():
    product = _two_case_product()
    annotation = StochasticValidity(probe_epsilon=1.0).process(
        product,
        CAMAnnotationContext(model=_FOLD, metadata={}, source_role="candidates"),
    )
    status = [str(value) for value in annotation.handle("status").materialize()]
    # The perturbation is resolved from the case metadata, so the row is not
    # rejected with a missing-parameter status.
    assert "perturbation_parameter_missing" not in status


# --- case-scope boundaries -------------------------------------------------


def test_a_short_case_metadata_never_rescopes_a_candidate():
    view = candidate_view(
        _product(
            case_metadata=(
                {"case_parameters": {"omega": 0.1}},
                {"case_parameters": {"omega": 0.9}},
            ),
            case_params={"omega": np.asarray([0.25, 0.75])},
        )
    )
    damaged = dataclasses.replace(view, case_metadata=())
    # The case still comes from the offsets; only the metadata is missing.
    assert [damaged.case_flat_index(row.index) for row in damaged.rows] == [0, 0, 1]
    assert damaged.metadata_at(2) == {}
    assert not damaged.design_at(2).is_resolved
    # The case parameter columns are read for the right case, not for case 0.
    assert damaged.params_at(2)["omega"] == pytest.approx(0.75)
    assert damaged.params_at(0)["omega"] == pytest.approx(0.25)


def test_a_case_index_rejects_bools_and_non_integers():
    view = candidate_view(_product())
    with pytest.raises(TypeError, match="must be an integer"):
        view.case_flat_index(True)
    with pytest.raises(TypeError, match="must be an integer"):
        view.case_flat_index(1.9)
    with pytest.raises(TypeError, match="must be an integer"):
        view.params_at(np.float64(1.0))
    # numpy integers are integers.
    assert view.case_flat_index(np.int64(2)) == 1


def test_a_scan_case_index_rejects_bools_and_non_integers():
    scan = CAMBifurcationScanResult(
        case_axes={"gamma": np.asarray([0.0, 0.5])},
        case_shape=(2,),
        case_params={"gamma": np.asarray([0.0, 0.5])},
        candidate_offsets=np.asarray([0, 1, 2]),
        candidates=_candidates(2),
        case_metadata=({"case": 0}, {"case": 1}),
        meta={},
    )
    with pytest.raises(TypeError, match="candidate index must be an integer"):
        scan.case_flat_index(True)
    with pytest.raises(TypeError, match="candidate index must be an integer"):
        scan.case_flat_index(1.9)
    with pytest.raises(TypeError, match="flat case index must be an integer"):
        scan.grid_index(1.0)
    assert scan.case_grid_index(np.int64(1)) == (1,)


def test_conflicting_case_base_parameters_are_refused():
    assert case_fixed_parameters(
        {"case_parameters": {"omega": 1.0}, "fixed_params": {"omega": 1.0}}
    ) == pytest.approx({"omega": 1.0})
    with pytest.raises(ValueError, match="conflicting values for 'omega'"):
        case_fixed_parameters(
            {"case_parameters": {"omega": 1.0}, "fixed_params": {"omega": 2.0}}
        )


def test_an_unrecorded_side_reads_as_both():
    view = candidate_view(
        _product(
            offsets=(0, 2),
            case_metadata=({"case_parameters": {}, "perturbation_parameter": "g"},),
        )
    )
    assert view.design_at(0).side == "both"
    assert PerturbationDesign().side == "both"
    recorded = candidate_view(
        _product(
            offsets=(0, 2),
            case_metadata=({"case_parameters": {}, "perturbation_side": "positive"},),
        )
    )
    assert recorded.design_at(0).side == "positive"


def test_a_single_case_result_keeps_its_legacy_parameter_map():
    result = _candidates(2)
    result.meta = {**result.meta, "fixed_params": {"omega": 5.0, "g": 0.4}}
    params = candidate_parameters(result, 1, _Model())
    # model.params -> the result's fixed parameters -> the candidate's controls.
    assert params["omega"] == pytest.approx(5.0)
    assert params["g"] == pytest.approx(0.4)
    assert params["gamma"] == pytest.approx(-0.5)
    with pytest.raises(IndexError, match="out of bounds"):
        candidate_parameters(result, 2, _Model())
    with pytest.raises(TypeError, match="must be an integer"):
        candidate_parameters(result, 1.5, _Model())


def test_closure_risk_uses_the_shared_case_map():
    product = _product(
        case_metadata=(
            {"case_parameters": {"omega": 0.1, "g": 0.2}},
            {"case_parameters": {"omega": 0.9, "g": 0.8}},
        )
    )
    view = candidate_view(product)
    model = _Model()
    for row in view.rows:
        assert ClosureRisk._candidate_parameters(view, row, model) == (
            candidate_parameters(view, row.index, model)
        )
    # The case's own base parameters reach the assessment, not only the columns.
    assert ClosureRisk._candidate_parameters(view, view.rows[0], model)["g"] == (
        pytest.approx(0.2)
    )
    assert ClosureRisk._candidate_parameters(view, view.rows[-1], model)["g"] == (
        pytest.approx(0.8)
    )


def test_a_json_complex_coefficient_reaches_the_real_chart():
    # The producer stores a coefficient matrix as a nested list of per-element
    # {"real", "imag"} mappings, exactly as this fixture is written.
    coefficient = [
        [{"real": 1.0, "imag": 0.0}, {"real": 0.0, "imag": 0.5}],
        [{"real": 0.0, "imag": -0.5}, {"real": 1.0, "imag": 0.0}],
    ]
    signature = {
        "state_order": 3,
        "perturbation_order": 1,
        "coupling_state_order": 0,
        "exponent_numerator": 1,
        "exponent_denominator": 3,
        "exponent": 1.0 / 3.0,
        "sublinear": True,
        "branches": (
            {
                "epsilon_side": 1,
                "amplitude": 2.0,
                "leading_state_coefficient": coefficient,
            },
        ),
    }
    product = build_candidate_product(
        states=np.asarray([np.eye(2)], dtype=complex),
        state_vectors=np.zeros((1, 4)),
        control_values=np.full((1, 1), -0.5),
        control_names=("gamma",),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.full(1, 1e-14),
        success=np.ones(1, dtype=bool),
        status=["verified"],
        method=["test"],
        verification_digits=np.full(1, 50, dtype=np.int64),
        verification_status=["verified"],
        candidate_identity=["cam:candidate:0"],
        classification=[{"scaling_signatures": [signature]}],
        case_metadata=(
            {
                "case_parameters": _FOLD_PARAMS,
                "perturbation_parameter": "gamma",
                "perturbation_scale": 1.0,
            },
        ),
        candidate_offsets=np.asarray([0, 1]),
        case_shape=(1,),
    )
    view = candidate_view(product)
    # A ragged object column stores one matrix per row; the imaginary part
    # survives the JSON round trip, so the consumer must read a real chart from a
    # normalised complex array.
    stored = view.branches.leading_state_coefficient[0]
    assert np.any(np.asarray(stored, dtype=complex).imag != 0.0)
    np.testing.assert_allclose(
        np.asarray(stored, dtype=complex), np.asarray([[1.0, 0.5j], [-0.5j, 1.0]])
    )
    np.testing.assert_allclose(
        matrix_to_vector(np.asarray(stored, dtype=complex)),
        matrix_to_vector(np.asarray([[1.0, 0.5j], [-0.5j, 1.0]])),
    )
    annotation = StochasticValidity(probe_epsilon=1.0).process(
        product,
        CAMAnnotationContext(model=_FOLD, metadata={}, source_role="candidates"),
    )
    assert len(np.asarray(annotation.handle("status").materialize())) == 1
