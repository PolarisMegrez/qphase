"""Search-audit fragments, the evidence product and the migration input.

The fragment tests are mostly single-field perturbations of one valid fragment:
the point of the contract is that every conservation, unit and provenance rule
fails loudly instead of producing plausible-looking evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import qphase_cam.data as cam_data
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase.data import GenericDataBundle, ProductSchema, load_bundle, load_products
from qphase_cam.data import (
    CAM_BUNDLE_TYPE_ID,
    CAM_EVIDENCE_PRODUCT,
    EVIDENCE_COUNT_FIELDS,
    CAMOutcomeStatus,
    SourceArtifactRef,
    build_evidence_product,
    cam_bundle,
    import_search_evidence_document,
)
from qphase_cam.evidence import (
    AUDIT_CONSERVATION,
    AUDIT_FIELD_UNITS,
    AUDIT_SCHEMA,
    EMPTY_RESULT_NOTE,
    REJECTION_COUNTER_SEMANTICS,
    GenerationPathAudit,
    SearchAuditFragment,
    fragment_from_bifurcation_metadata,
)
from qphase_cam.solver.bifurcation import (
    BifurcationSolver,
    BifurcationSolverConfig,
    ControlRange,
    PerturbationConfig,
)
from qphase_cam.solver.bifurcation_classifier import (
    ScalingSignatureClassifier,
    ScalingSignatureConfig,
)
from qphase_cam.solver.bifurcation_discovery import SeedDiscovery, SeedDiscoveryConfig
from qphase_cam.solver.bifurcation_strategy import FullStrategy, FullStrategyConfig
from qphase_cam.solver.bifurcation_target import (
    EquilibriumMultiplicity,
    EquilibriumMultiplicityConfig,
)

from .test_bifurcation import TwoModeFoldModel

#: One historical document that is present in the repository.
HISTORICAL_DOCUMENT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "models"
    / "vdp_2mode"
    / "vdp_2mode_search_evidence.json"
)


# --- fixtures --------------------------------------------------------------


def _fragment_payload(**overrides):
    """Return one conserving, complete fragment payload."""
    payload = {
        "case_identity": "search:0:case:0",
        "search_identity": "search:0",
        "status": CAMOutcomeStatus.VALID,
        "physical_status": "checked",
        "search_design": {"target": "equilibrium_multiplicity", "order": 2},
        "controls": {"gamma": {"bounds": [-1.0, 0.0], "sampling": "linear"}},
        "coverage": {"coverage": "bordered_local_search", "seed_truncated": False},
        "generated_candidate_count": 7,
        "prefilter_pass_count": 7,
        "prefilter_rejected_count": 0,
        "refinement_start_count": 7,
        "refinement_duplicate_count": 4,
        "accepted_count": 1,
        "rejected_count": 2,
        "prefilter_rejected": {},
        "rejected_by_reason": {"higher_multiplicity": 2, "non_simple_kernel": 2},
        "near_miss_saved": 2,
        "near_miss_dropped": 0,
        "unique_candidate_count": 1,
        "paths": (
            {
                "path": "full",
                "generation_trial_count": 28,
                "generated_candidate_count": 7,
                "seed_skips": {"duplicate_state": 13, "fixed_point_not_converged": 8},
                "seed_truncated": False,
                "workload": {"control_point_count": 7, "fixed_point_guess_count": 28},
            },
        ),
        "generation_trial_count": 28,
        "accepted_candidate_identities": ("legacy:abc:0:0",),
        "near_misses": (
            {
                "path": "full",
                "reduction": None,
                "seed_source": "domain_sampling",
                "controls": {"gamma": -0.25},
                "rejection_reasons": ("higher_multiplicity", "non_simple_kernel"),
                "search_residual": 1e-26,
                "full_residual": 0.0,
                "min_state_eigenvalue": 0.375,
            },
            {
                "path": "full",
                "reduction": None,
                "seed_source": "domain_sampling",
                "controls": {"gamma": -0.75},
                "rejection_reasons": ("higher_multiplicity",),
                "search_residual": 2e-26,
                "full_residual": 0.0,
                "min_state_eigenvalue": None,
            },
        ),
        "solver": "bifurcation",
        "plugins": {"target": "equilibrium_multiplicity", "strategy": "full"},
        "distribution": {"qphase": "2.0.0", "qphase-cam": "2.0.0"},
        "api_revision": AUDIT_SCHEMA,
        "backend": "numpy",
        "dtype": "complex128",
    }
    payload.update(overrides)
    return payload


def _fragment(**overrides) -> SearchAuditFragment:
    return SearchAuditFragment(**_fragment_payload(**overrides))


def _not_found_fragment(**overrides):
    """Return a case that searched and accepted nothing."""
    payload = _fragment_payload(
        case_identity="search:0:case:1",
        status=CAMOutcomeStatus.NOT_FOUND,
        accepted_count=0,
        accepted_candidate_identities=(),
        unique_candidate_count=0,
        generated_candidate_count=2,
        prefilter_pass_count=2,
        refinement_start_count=2,
        refinement_duplicate_count=0,
        rejected_count=2,
        near_miss_saved=2,
        near_miss_dropped=0,
        coverage={
            "coverage": "bordered_local_search",
            "result_note": EMPTY_RESULT_NOTE,
            "seed_truncated": False,
        },
    )
    payload.update(overrides)
    return SearchAuditFragment(**payload)


def _deduplicated_fragment(**overrides):
    """Return a case where two accepted starts deduplicate into one candidate."""
    payload = _fragment_payload(
        accepted_count=2,
        unique_candidate_count=1,
        accepted_start_identities=("cam:start:bbb", "cam:start:aaa"),
        accepted_candidate_identities=("cam:candidate:ccc",),
        generated_candidate_count=8,
        prefilter_pass_count=8,
        refinement_start_count=8,
    )
    payload.update(overrides)
    return SearchAuditFragment(**payload)


def _no_object_dtypes(product):
    for variable in product.variables:
        assert not np.dtype(variable.dtype).hasobject, variable.name
        array = np.asarray(product.handle(variable.name).materialize())
        assert not array.dtype.hasobject, variable.name


class _MetadataView:
    """Solver metadata plus the candidate count the tests refer to.

    The ``states`` attribute is the length of the accepted candidate list, which
    is what the old packed result exposed as its candidate axis.
    """

    def __init__(self, metadata, candidates):
        self.meta = dict(metadata)
        self.states = [None] * len(candidates)

    def __getitem__(self, key):
        return self.meta[key]


def _two_mode_fold_metadata():
    """Run the existing TwoModeFold search once and return its packed metadata."""
    solver = BifurcationSolver(
        BifurcationSolverConfig(
            controls={"gamma": ControlRange(min=-1.0, max=0.0)},
            perturbation=PerturbationConfig(parameter="gamma"),
        ),
        subplugins={
            "target": EquilibriumMultiplicity(EquilibriumMultiplicityConfig(order=2)),
            "strategy": FullStrategy(FullStrategyConfig()),
            "discovery": SeedDiscovery(SeedDiscoveryConfig()),
            "classifier": ScalingSignatureClassifier(ScalingSignatureConfig()),
        },
    )
    model = TwoModeFoldModel()
    output = solver.search(model, NumpyBackend())
    metadata = dict(output.metadata)
    metadata.setdefault("target", output.target)
    metadata.setdefault("order", output.order)
    metadata.setdefault(
        "state_ids",
        list(getattr(model, "state_ids", ())) or [],
    )
    return _MetadataView(metadata, output.candidates)


# --- fragment vocabulary ---------------------------------------------------


def test_fragment_reuses_the_solver_audit_vocabulary():
    """One counting system: the fragment ships the solver's own constants."""
    assert AUDIT_SCHEMA == "cam_bifurcation_audit/2"
    assert "generation_trial_count = generated_candidate_count + sum(seed_skips" in (
        " ".join(AUDIT_CONSERVATION)
    )
    assert "multi-label" in REJECTION_COUNTER_SEMANTICS
    assert EMPTY_RESULT_NOTE.endswith("stated_numerical_coverage")
    fragment = _fragment()
    assert fragment.fragment_schema == "qphase_cam.search_audit_fragment/1"
    assert set(fragment.paths[0].workload) <= set(AUDIT_FIELD_UNITS)


def test_fragment_defaults_to_the_documented_schema():
    assert _fragment().to_status() is CAMOutcomeStatus.VALID


# --- conservation ----------------------------------------------------------


def test_fragment_accepts_a_conserving_case():
    fragment = _fragment()
    assert fragment.generated_candidate_count == 7
    assert fragment.refinement_start_count == 7
    assert fragment.refinement_start_count == fragment.prefilter_pass_count
    assert fragment.generation_trial_count == 28
    assert len(fragment.accepted_candidate_identities) == fragment.accepted_count
    assert len(fragment.near_misses) == fragment.near_miss_saved


@pytest.mark.parametrize(
    "overrides, match",
    [
        (
            {"generated_candidate_count": 6},
            "generated_candidate_count must equal",
        ),
        (
            {"refinement_start_count": 6},
            "refinement_start_count must equal",
        ),
        (
            {"prefilter_pass_count": 6, "prefilter_rejected_count": 1},
            "must equal prefilter_pass_count",
        ),
        (
            {"prefilter_rejected_count": 1, "generated_candidate_count": 8},
            "prefilter_rejected must sum to",
        ),
        (
            {"rejected_by_reason": {"higher_multiplicity": 1}},
            "multi-label counter and must sum to at least",
        ),
        (
            {"near_miss_dropped": 1},
            "near_miss_dropped must equal",
        ),
        (
            {"generated_candidate_count": -1},
            "greater than or equal to 0",
        ),
    ],
)
def test_fragment_rejects_a_non_conserving_case(overrides, match):
    with pytest.raises(ValidationError, match=match):
        _fragment(**overrides)


def test_fragment_accepts_multi_label_over_counting():
    """Multi-label reasons may sum to more than the rejected count."""
    fragment = _fragment(
        rejected_by_reason={"higher_multiplicity": 2, "non_simple_kernel": 2}
    )
    assert sum(fragment.rejected_by_reason.values()) == 4
    assert fragment.rejected_count == 2
    # Equality would reject this case; the contract only requires >=.
    with pytest.raises(ValidationError, match="at least"):
        _fragment(rejected_by_reason={"higher_multiplicity": 1})


# --- per-path units --------------------------------------------------------


def test_fragment_never_pools_generation_trials_across_paths():
    with pytest.raises(ValidationError, match="must not be pooled"):
        _fragment(
            paths=(
                {
                    "path": "reduced_fraction_free",
                    "generation_trial_count": 14,
                    "generated_candidate_count": 10,
                    "seed_skips": {"complex_root": 4},
                    "workload": {"polynomial_root_count": 14},
                },
                {
                    "path": "full",
                    "generation_trial_count": 28,
                    "generated_candidate_count": 7,
                    "seed_skips": {
                        "duplicate_state": 13,
                        "fixed_point_not_converged": 8,
                    },
                    "workload": {"fixed_point_guess_count": 28},
                },
            ),
            generation_trial_count=42,
        )


def test_fragment_accepts_a_multi_path_case_without_a_trial_total():
    fragment = _fragment(
        paths=(
            {
                "path": "reduced_fraction_free",
                "generation_trial_count": 14,
                "generated_candidate_count": 10,
                "seed_skips": {"complex_root": 4},
                "workload": {"polynomial_root_count": 14},
            },
            {
                "path": "full",
                "generation_trial_count": 28,
                "generated_candidate_count": 7,
                "seed_skips": {"duplicate_state": 21},
                "workload": {"fixed_point_guess_count": 28},
            },
        ),
        generation_trial_count=None,
    )
    assert fragment.generation_trial_count is None
    assert [entry.path for entry in fragment.paths] == [
        "reduced_fraction_free",
        "full",
    ]


def test_fragment_validates_the_per_path_trial_identity():
    with pytest.raises(ValidationError, match="generation_trial_count must equal"):
        GenerationPathAudit(
            path="full",
            generation_trial_count=10,
            generated_candidate_count=7,
            seed_skips={"duplicate_state": 1},
        )


def test_fragment_rejects_a_workload_unit_without_a_declared_unit():
    with pytest.raises(ValidationError, match="workload keys without a declared unit"):
        GenerationPathAudit(
            path="full",
            generation_trial_count=1,
            generated_candidate_count=1,
            workload={"made_up_unit": 1},
        )


# --- status semantics ------------------------------------------------------


def test_negative_results_are_distinct_statuses():
    not_found = _not_found_fragment()
    assert not_found.status is CAMOutcomeStatus.NOT_FOUND
    assert not not_found.accepted

    not_searched = _fragment(
        case_identity="search:0:case:2",
        status=CAMOutcomeStatus.NOT_SEARCHED,
        generated_candidate_count=0,
        prefilter_pass_count=0,
        refinement_start_count=0,
        refinement_duplicate_count=0,
        accepted_count=0,
        rejected_count=0,
        rejected_by_reason={},
        near_miss_saved=0,
        near_miss_dropped=0,
        unique_candidate_count=0,
        accepted_candidate_identities=(),
        near_misses=(),
        paths=(),
        generation_trial_count=None,
        coverage={"coverage": "not_covered", "seed_truncated": False},
    )
    assert not_searched.status is CAMOutcomeStatus.NOT_SEARCHED
    assert not_found.status is not not_searched.status


def test_not_searched_cannot_carry_searched_counts():
    with pytest.raises(ValidationError, match="cannot carry searched counts"):
        _fragment(status=CAMOutcomeStatus.NOT_SEARCHED)


def test_not_found_requires_an_actual_search():
    with pytest.raises(ValidationError, match="requires that something was actually"):
        _not_found_fragment(
            generated_candidate_count=0,
            prefilter_pass_count=0,
            prefilter_rejected_count=0,
            refinement_start_count=0,
            rejected_count=0,
            near_miss_saved=0,
            rejected_by_reason={},
            near_misses=(),
            paths=(),
            generation_trial_count=None,
        )


def test_not_found_cannot_carry_an_accepted_candidate():
    with pytest.raises(ValidationError, match="cannot carry accepted"):
        _not_found_fragment(
            accepted_count=1,
            accepted_candidate_identities=("legacy:abc:0:0",),
            unique_candidate_count=1,
            generated_candidate_count=3,
            prefilter_pass_count=3,
            refinement_start_count=3,
        )


def test_valid_requires_an_accepted_candidate():
    with pytest.raises(ValidationError, match="requires at least one accepted"):
        _fragment(
            accepted_count=0,
            accepted_candidate_identities=(),
            unique_candidate_count=0,
            generated_candidate_count=6,
            prefilter_pass_count=6,
            refinement_start_count=6,
        )


def test_unchecked_physicality_is_not_a_failure():
    with pytest.raises(
        ValidationError, match="an unchecked physicality is not a failure"
    ):
        _fragment(physical_status="not_checked", status=CAMOutcomeStatus.NONPHYSICAL)
    fragment = _fragment(
        physical_status="not_checked", status=CAMOutcomeStatus.UNCHECKED
    )
    assert fragment.physical_status == "not_checked"
    assert fragment.status is CAMOutcomeStatus.UNCHECKED


# --- identity and evidence payload ----------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        (
            {"accepted_candidate_identities": ()},
            "accepted_count=1 but 0 candidate identities",
        ),
        (
            {"accepted_candidate_identities": ("",)},
            "identities\\[0\\] is empty",
        ),
        (
            {
                "accepted_count": 2,
                "accepted_candidate_identities": ("a", "a"),
                "unique_candidate_count": 2,
                "generated_candidate_count": 8,
                "prefilter_pass_count": 8,
                "refinement_start_count": 8,
            },
            "must be unique",
        ),
        (
            {"unique_candidate_count": 5},
            "unique_candidate_count=5 but 1 candidate identities",
        ),
        (
            {
                "accepted_count": 2,
                "accepted_start_identities": ("start:a", "start:b"),
                "accepted_candidate_identities": (
                    "candidate:a",
                    "candidate:b",
                    "candidate:c",
                ),
                "unique_candidate_count": 3,
                "generated_candidate_count": 8,
                "prefilter_pass_count": 8,
                "refinement_start_count": 8,
            },
            "must not exceed accepted_count",
        ),
        (
            {"near_misses": ()},
            "0 near-miss records for near_miss_saved=2",
        ),
    ],
)
def test_fragment_rejects_inconsistent_evidence(overrides, match):
    with pytest.raises(ValidationError, match=match):
        _fragment(**overrides)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"solver": ""}, "String should have at least 1 character"),
        ({"plugins": {}}, "at least 1 item"),
        ({"distribution": {}}, "at least 1 item"),
        ({"api_revision": ""}, "String should have at least 1 character"),
        ({"backend": ""}, "String should have at least 1 character"),
        ({"dtype": ""}, "String should have at least 1 character"),
        ({"plugins": {"target": " "}}, "must not be blank"),
        ({"search_design": {}}, "at least 1 item"),
        ({"controls": {}}, "at least 1 item"),
        ({"coverage": {}}, "at least 1 item"),
    ],
)
def test_fragment_requires_complete_provenance(overrides, match):
    with pytest.raises(ValidationError, match=match):
        _fragment(**overrides)


# --- from-metadata mapping -------------------------------------------------


def test_fragment_from_two_mode_fold_metadata():
    result = _two_mode_fold_metadata()
    identities = [f"legacy:abc:0:{index}" for index in range(len(result.states))]
    fragment = fragment_from_bifurcation_metadata(
        result.meta,
        case_identity="case:0",
        search_identity="search:0",
        accepted_candidate_identities=identities,
        backend="numpy",
        dtype="complex128",
    )
    assert fragment.status is CAMOutcomeStatus.VALID
    assert fragment.physical_status == "checked"
    assert fragment.accepted_count == len(identities)
    assert [entry.path for entry in fragment.paths] == ["full"]
    assert fragment.generation_trial_count == fragment.paths[0].generation_trial_count
    assert fragment.near_miss_saved == fragment.rejected_count
    assert len(fragment.near_misses) == fragment.near_miss_saved
    assert fragment.near_misses[0].rejection_reasons
    assert fragment.controls["gamma"].bounds == (-1.0, 0.0)
    assert fragment.plugins["target"] == "equilibrium_multiplicity"
    assert set(fragment.distribution) >= {"qphase", "qphase-cam"}
    assert fragment.api_revision == AUDIT_SCHEMA
    assert fragment.search_design["order"] == 2


def test_fragment_from_metadata_reports_what_it_cannot_map():
    result = _two_mode_fold_metadata()
    identities = [f"legacy:abc:0:{index}" for index in range(len(result.states))]
    metadata = dict(result.meta)
    metadata.pop("audit")
    with pytest.raises(ValueError, match="carries no audit payload"):
        fragment_from_bifurcation_metadata(
            metadata,
            case_identity="case:0",
            search_identity="search:0",
            accepted_candidate_identities=identities,
            backend="numpy",
            dtype="complex128",
        )
    # The current auto strategy dedups accepted starts into fewer candidate rows,
    # so one identity per accepted start is not representable yet.
    metadata = dict(result.meta)
    metadata["audit"] = {**metadata["audit"], "schema": "cam_bifurcation_audit/1"}
    with pytest.raises(ValueError, match="unrecognized audit schema"):
        fragment_from_bifurcation_metadata(
            metadata,
            case_identity="case:0",
            search_identity="search:0",
            accepted_candidate_identities=identities,
            backend="numpy",
            dtype="complex128",
        )


def test_fragment_from_metadata_requires_the_producing_environment():
    result = _two_mode_fold_metadata()
    identities = [f"legacy:abc:0:{index}" for index in range(len(result.states))]
    with pytest.raises(ValueError, match="no plugin ids"):
        fragment_from_bifurcation_metadata(
            {
                key: value
                for key, value in result.meta.items()
                if key not in {"target", "strategy"}
            },
            case_identity="case:0",
            search_identity="search:0",
            accepted_candidate_identities=identities,
            backend="numpy",
            dtype="complex128",
            plugins={},
        )


def test_fragment_from_metadata_represents_a_deduplicated_case():
    """The auto strategy accepts 2 starts that dedup into 1 candidate row."""
    from .test_bifurcation import _solver

    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    result = _MetadataView(output.metadata, output.candidates)
    totals = result.meta["audit"]["totals"]
    if totals["unique_candidate_count"] >= totals["accepted_count"]:
        pytest.skip("the auto strategy did not deduplicate this run")
    fragment = fragment_from_bifurcation_metadata(
        result.meta,
        case_identity=result.meta["case_identity"],
        search_identity=result.meta["search_identity"],
        accepted_candidate_identities=result.meta["accepted_candidate_identities"],
        backend="numpy",
        dtype="complex128",
    )
    # accepted_count keeps its scientific meaning: accepted *starts*.
    assert fragment.accepted_count == totals["accepted_count"] == 2
    assert fragment.unique_candidate_count == totals["unique_candidate_count"] == 1
    assert len(fragment.accepted_start_identities) == 2
    assert len(fragment.accepted_candidate_identities) == 1
    assert len(set(fragment.accepted_start_identities)) == 2
    # Two starts point at the same final candidate; neither layer is a slot.
    assert (
        fragment.accepted_start_identities == (result.meta["accepted_start_identities"])
    )
    assert (
        fragment.accepted_candidate_identities
        == (result.meta["accepted_candidate_identities"])
    )
    assert all(
        identity.startswith("cam:start:")
        for identity in fragment.accepted_start_identities
    )
    assert all(
        identity.startswith("cam:candidate:")
        for identity in fragment.accepted_candidate_identities
    )


def test_fragment_from_metadata_requires_start_ids_for_a_deduplicated_case():
    """Without the start layer the case must not be silently collapsed."""
    from .test_bifurcation import _solver

    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    result = _MetadataView(output.metadata, output.candidates)
    totals = result.meta["audit"]["totals"]
    if totals["unique_candidate_count"] >= totals["accepted_count"]:
        pytest.skip("the auto strategy did not deduplicate this run")
    metadata = {
        key: value
        for key, value in result.meta.items()
        if key != "accepted_start_identities"
    }
    with pytest.raises(ValueError, match="must provide accepted_start_identities"):
        fragment_from_bifurcation_metadata(
            metadata,
            case_identity=result.meta["case_identity"],
            search_identity=result.meta["search_identity"],
            accepted_candidate_identities=result.meta["accepted_candidate_identities"],
            backend="numpy",
            dtype="complex128",
        )


# --- evidence product ------------------------------------------------------


def test_evidence_template_keeps_its_axes_open_and_declares_the_units():
    assert isinstance(CAM_EVIDENCE_PRODUCT, ProductSchema)
    assert CAM_EVIDENCE_PRODUCT.is_closed is False
    assert [axis.name for axis in CAM_EVIDENCE_PRODUCT.axes] == [
        "case",
        "case_offset",
        "accepted",
        "accepted_start",
        "near_miss",
        "control",
    ]
    assert CAM_EVIDENCE_PRODUCT.attributes["product"] == "evidence"
    assert CAM_EVIDENCE_PRODUCT.attributes["count_fields"] == list(
        EVIDENCE_COUNT_FIELDS
    )
    semantics = CAM_EVIDENCE_PRODUCT.attributes["status_semantics"]
    assert "not_found" in semantics and "not_searched" in semantics
    assert "none of them is a failure verdict" in semantics
    assert (
        "never summed"
        in (CAM_EVIDENCE_PRODUCT.attributes["generation_trial_semantics"])
    )


def test_build_evidence_product_maps_cases_and_blocks():
    product = build_evidence_product(
        fragments=[_fragment(), _not_found_fragment()],
        source_artifacts=[{"artifact_id": "artifact-1", "job": "search"}],
        search_design={"target": "equilibrium_multiplicity", "order": 2},
    )
    assert product.attributes["case_count"] == 2
    assert product.attributes["accepted_start_count"] == 1
    assert product.attributes["accepted_candidate_count"] == 1
    assert product.attributes["near_miss_count"] == 4
    assert product.attributes["evidence_completeness"] == "complete"
    assert product.attributes["source_artifacts"] == [
        {"artifact_id": "artifact-1", "job": "search"}
    ]
    assert list(np.asarray(product.handle("status").materialize())) == [
        "valid",
        "not_found",
    ]
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_offsets").materialize()), [0, 1, 1]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("near_miss_offsets").materialize()), [0, 2, 4]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_identity").materialize()),
        ["legacy:abc:0:0"],
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_start_offsets").materialize()), [0, 1, 1]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_start_identity").materialize()),
        ["legacy:abc:0:0"],
    )
    # The typed counts are candidate-start counts only.
    for field in EVIDENCE_COUNT_FIELDS:
        assert product.variable(field).dims == ("case",)
    np.testing.assert_array_equal(
        np.asarray(product.handle("rejected_count").materialize()), [2, 2]
    )
    assert list(np.asarray(product.handle("control_names").materialize())) == ["gamma"]
    np.testing.assert_allclose(
        np.asarray(product.handle("near_miss_control_values").materialize()),
        [[-0.25], [-0.75], [-0.25], [-0.75]],
    )
    # Per-path trials stay per path in JSON, never pooled into a column.
    assert "generation_trial_count" not in {v.name for v in product.variables}
    path_audit = [
        json.loads(str(value))
        for value in np.asarray(product.handle("path_audit").materialize())
    ]
    assert path_audit[0][0]["generation_trial_count"] == 28
    assert path_audit[1][0]["generation_trial_count"] == 28
    assert json.loads(
        str(np.asarray(product.handle("rejected_by_reason").materialize())[0])
    ) == {"higher_multiplicity": 2, "non_simple_kernel": 2}
    assert product.attributes["conservation"] == list(AUDIT_CONSERVATION)
    assert product.attributes["rejection_counter_semantics"] == (
        REJECTION_COUNTER_SEMANTICS
    )
    # The producing environment is evidence and is carried per case.
    assert product.attributes["execution_provenance_variable"] == (
        "execution_provenance"
    )
    assert product.attributes["execution_provenance_fields"] == [
        "solver",
        "plugins",
        "distribution",
        "api_revision",
        "backend",
        "dtype",
    ]
    provenance = [
        json.loads(str(value))
        for value in np.asarray(product.handle("execution_provenance").materialize())
    ]
    assert provenance == [
        {
            "solver": "bifurcation",
            "plugins": {"strategy": "full", "target": "equilibrium_multiplicity"},
            "distribution": {"qphase": "2.0.0", "qphase-cam": "2.0.0"},
            "api_revision": AUDIT_SCHEMA,
            "backend": "numpy",
            "dtype": "complex128",
        },
        {
            "solver": "bifurcation",
            "plugins": {"strategy": "full", "target": "equilibrium_multiplicity"},
            "distribution": {"qphase": "2.0.0", "qphase-cam": "2.0.0"},
            "api_revision": AUDIT_SCHEMA,
            "backend": "numpy",
            "dtype": "complex128",
        },
    ]
    _no_object_dtypes(product)


def test_evidence_product_carries_an_empty_near_miss_block():
    product = build_evidence_product(
        fragments=[
            _fragment(
                rejected_count=0,
                near_miss_saved=0,
                near_misses=(),
                rejected_by_reason={},
                generated_candidate_count=5,
                prefilter_pass_count=5,
                refinement_start_count=5,
                refinement_duplicate_count=4,
            )
        ],
        source_artifacts=[{"artifact_id": "artifact-1"}],
        search_design={"target": "equilibrium_multiplicity"},
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("near_miss_offsets").materialize()), [0, 0]
    )
    assert product.axis("near_miss").size == 0
    _no_object_dtypes(product)


def test_evidence_product_rejects_a_dangling_accepted_reference():
    from qphase_cam.data import build_candidate_product

    candidate_product = build_candidate_product(
        states=np.eye(2)[None].astype(complex),
        state_vectors=np.zeros((1, 4)),
        control_values=np.full((1, 1), -0.5),
        control_names=("gamma",),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=["verified"],
        method=["test"],
        verification_digits=np.zeros(1, dtype=np.int64),
        verification_status=["verified"],
        candidate_identity=["candidate:0"],
    )
    with pytest.raises(ValueError, match="not present in the candidate product"):
        build_evidence_product(
            fragments=[_fragment()],
            source_artifacts=[{"artifact_id": "artifact-1"}],
            search_design={"target": "equilibrium_multiplicity"},
            candidate_product=candidate_product,
        )
    # A reference that exists passes.
    product = build_evidence_product(
        fragments=[_fragment(accepted_candidate_identities=("candidate:0",))],
        source_artifacts=[{"artifact_id": "artifact-1"}],
        search_design={"target": "equilibrium_multiplicity"},
        candidate_product=candidate_product,
    )
    assert product.attributes["accepted_candidate_count"] == 1


def test_evidence_product_requires_complete_provenance():
    with pytest.raises(ValueError, match="needs at least one source artifact"):
        build_evidence_product(
            fragments=[_fragment()],
            source_artifacts=[],
            search_design={"target": "equilibrium_multiplicity"},
        )
    with pytest.raises(ValueError, match="requires an intermediate_reason"):
        build_evidence_product(
            fragments=[_fragment()],
            source_artifacts=[],
            search_design={"target": "equilibrium_multiplicity"},
            complete=False,
        )
    with pytest.raises(ValueError, match="must be unique"):
        build_evidence_product(
            fragments=[_fragment()],
            source_artifacts=[
                {"artifact_id": "artifact-1"},
                {"artifact_id": "artifact-1"},
            ],
            search_design={"target": "equilibrium_multiplicity"},
        )
    with pytest.raises(ValueError, match="must be unique across the product"):
        build_evidence_product(
            fragments=[_fragment(), _fragment()],
            source_artifacts=[{"artifact_id": "artifact-1"}],
            search_design={"target": "equilibrium_multiplicity"},
        )
    with pytest.raises(ValueError, match="one control set"):
        build_evidence_product(
            fragments=[
                _fragment(),
                _not_found_fragment(controls={"Gamma": {"bounds": [0.0, 1.0]}}),
            ],
            source_artifacts=[{"artifact_id": "artifact-1"}],
            search_design={"target": "equilibrium_multiplicity"},
        )


def test_intermediate_evidence_is_marked_and_never_complete():
    product = build_evidence_product(
        fragments=[_fragment()],
        source_artifacts=[],
        search_design={"target": "equilibrium_multiplicity"},
        complete=False,
        intermediate_reason="solver intermediate: search still running",
    )
    assert product.attributes["evidence_completeness"] == "intermediate"
    assert product.attributes["intermediate_reason"] == (
        "solver intermediate: search still running"
    )
    assert product.attributes["source_artifacts"] == []
    assert product.provenance["evidence_completeness"] == "intermediate"


def test_source_artifact_rejects_a_path():
    with pytest.raises(ValueError, match="references artifacts by id"):
        SourceArtifactRef(artifact_id="runs/2026-01-01/artifact")
    with pytest.raises(ValueError, match="must not contain"):
        SourceArtifactRef(artifact_id="run:7")


def test_evidence_product_rejects_a_migration_document():
    document = import_search_evidence_document(HISTORICAL_DOCUMENT)
    with pytest.raises(TypeError, match="never becomes a product"):
        build_evidence_product(
            fragments=[document],
            source_artifacts=[{"artifact_id": "artifact-1"}],
            search_design={"target": "equilibrium_multiplicity"},
        )


def test_evidence_bundle_roundtrips_in_both_layouts(tmp_path):
    bundle = cam_bundle(
        {
            "evidence": build_evidence_product(
                fragments=[_fragment(), _not_found_fragment()],
                source_artifacts=[{"artifact_id": "artifact-1", "job": "search"}],
                search_design={"target": "equilibrium_multiplicity", "order": 2},
            )
        },
        provenance={"cam_task": "search"},
    )
    for layout in ("single", "sharded"):
        root = tmp_path / layout
        report = bundle.save_dataset(root, layout=layout, shard_target_bytes=256)
        assert report.files
        assert set(load_products(root)) == {"evidence"}
        restored = load_bundle(root)
        assert isinstance(restored, GenericDataBundle)
        assert restored.bundle_descriptor.type_id == CAM_BUNDLE_TYPE_ID
        assert restored.bundle_descriptor.product_roles == {"evidence": "evidence"}
        reloaded = restored.require("evidence").materialize()
        assert reloaded.attributes["evidence_completeness"] == "complete"
        assert [
            json.loads(str(value))
            for value in np.asarray(reloaded.handle("search_design").materialize())
        ][0]["order"] == 2
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("near_miss_offsets").materialize()),
            [0, 2, 4],
        )
        provenance = [
            json.loads(str(value))
            for value in np.asarray(
                reloaded.handle("execution_provenance").materialize()
            )
        ]
        assert len(provenance) == 2
        for row in provenance:
            assert row["solver"] == "bifurcation"
            assert row["plugins"] == {
                "strategy": "full",
                "target": "equilibrium_multiplicity",
            }
            assert row["distribution"] == {"qphase": "2.0.0", "qphase-cam": "2.0.0"}
            assert row["api_revision"] == AUDIT_SCHEMA
            assert row["backend"] == "numpy"
            assert row["dtype"] == "complex128"
        _no_object_dtypes(reloaded)


def test_evidence_product_keeps_mixed_execution_provenance(tmp_path):
    """Cases produced by different environments must not be merged."""
    left = _fragment()
    right = _not_found_fragment(
        backend="cupy",
        dtype="complex64",
        solver="bifurcation-v2",
        plugins={"target": "equilibrium_multiplicity", "strategy": "reduced"},
        distribution={"qphase": "2.0.1", "qphase-cam": "2.0.1"},
        api_revision="cam_bifurcation_audit/3",
    )
    bundle = cam_bundle(
        {
            "evidence": build_evidence_product(
                fragments=[left, right],
                source_artifacts=[{"artifact_id": "artifact-1"}],
                search_design={"target": "equilibrium_multiplicity"},
            )
        },
        provenance={"cam_task": "search"},
    )
    expected = [
        {
            "solver": "bifurcation",
            "plugins": {"strategy": "full", "target": "equilibrium_multiplicity"},
            "distribution": {"qphase": "2.0.0", "qphase-cam": "2.0.0"},
            "api_revision": AUDIT_SCHEMA,
            "backend": "numpy",
            "dtype": "complex128",
        },
        {
            "solver": "bifurcation-v2",
            "plugins": {
                "strategy": "reduced",
                "target": "equilibrium_multiplicity",
            },
            "distribution": {"qphase": "2.0.1", "qphase-cam": "2.0.1"},
            "api_revision": "cam_bifurcation_audit/3",
            "backend": "cupy",
            "dtype": "complex64",
        },
    ]
    for layout in ("single", "sharded"):
        root = tmp_path / layout
        bundle.save_dataset(root, layout=layout, shard_target_bytes=256)
        reloaded = load_bundle(root).require("evidence").materialize()
        assert [
            json.loads(str(value))
            for value in np.asarray(
                reloaded.handle("execution_provenance").materialize()
            )
        ] == expected
        assert reloaded.variable("execution_provenance").dims == ("case",)


def test_evidence_product_maps_both_identity_layers():
    product = build_evidence_product(
        fragments=[_deduplicated_fragment()],
        source_artifacts=[{"artifact_id": "artifact-1"}],
        search_design={"target": "equilibrium_multiplicity"},
    )
    assert product.attributes["accepted_start_count"] == 2
    assert product.attributes["accepted_candidate_count"] == 1
    assert product.axis("accepted").size == 1
    assert product.axis("accepted_start").size == 2
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_offsets").materialize()), [0, 1]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_start_offsets").materialize()), [0, 2]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_identity").materialize()),
        ["cam:candidate:ccc"],
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("accepted_start_identity").materialize()),
        ["cam:start:bbb", "cam:start:aaa"],
    )
    assert product.attributes["identity_layers"].startswith("accepted_identity")
    _no_object_dtypes(product)


def test_evidence_product_accepts_start_ids_that_are_not_candidates():
    """The candidate cross-check only applies to the final identity layer."""
    from qphase_cam.data import build_candidate_product

    candidate_product = build_candidate_product(
        states=np.eye(2)[None].astype(complex),
        state_vectors=np.zeros((1, 4)),
        control_values=np.full((1, 1), -0.5),
        control_names=("gamma",),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=["verified"],
        method=["test"],
        verification_digits=np.zeros(1, dtype=np.int64),
        verification_status=["verified"],
        candidate_identity=["cam:candidate:ccc"],
    )
    product = build_evidence_product(
        fragments=[_deduplicated_fragment()],
        source_artifacts=[{"artifact_id": "artifact-1"}],
        search_design={"target": "equilibrium_multiplicity"},
        candidate_product=candidate_product,
    )
    assert product.attributes["accepted_candidate_count"] == 1
    # A final identity that is not a candidate row still fails.
    with pytest.raises(ValueError, match="not present in the candidate product"):
        build_evidence_product(
            fragments=[
                _deduplicated_fragment(
                    accepted_candidate_identities=("cam:candidate:missing",)
                )
            ],
            source_artifacts=[{"artifact_id": "artifact-1"}],
            search_design={"target": "equilibrium_multiplicity"},
            candidate_product=candidate_product,
        )


def test_evidence_bundle_roundtrips_both_identity_layers(tmp_path):
    bundle = cam_bundle(
        {
            "evidence": build_evidence_product(
                fragments=[
                    _deduplicated_fragment(),
                    _fragment(case_identity="search:0:case:1"),
                ],
                source_artifacts=[{"artifact_id": "artifact-1"}],
                search_design={"target": "equilibrium_multiplicity"},
            )
        },
        provenance={"cam_task": "search"},
    )
    for layout in ("single", "sharded"):
        root = tmp_path / layout
        bundle.save_dataset(root, layout=layout, shard_target_bytes=256)
        reloaded = load_bundle(root).require("evidence").materialize()
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("accepted_offsets").materialize()), [0, 1, 2]
        )
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("accepted_start_offsets").materialize()),
            [0, 2, 3],
        )
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("accepted_identity").materialize()),
            ["cam:candidate:ccc", "legacy:abc:0:0"],
        )
        np.testing.assert_array_equal(
            np.asarray(reloaded.handle("accepted_start_identity").materialize()),
            ["cam:start:bbb", "cam:start:aaa", "legacy:abc:0:0"],
        )


def test_evidence_product_point_view_reads_strings():
    product = build_evidence_product(
        fragments=[_fragment(), _not_found_fragment()],
        source_artifacts=[{"artifact_id": "artifact-1"}],
        search_design={"target": "equilibrium_multiplicity"},
    )
    view = product.point_view(case=1)
    assert str(np.asarray(view.handle("status").materialize())) == "not_found"
    assert (
        json.loads(str(np.asarray(view.handle("coverage").materialize())))[
            "result_note"
        ]
        == EMPTY_RESULT_NOTE
    )
    # The ragged blocks are per product; a case view keeps them and the offsets
    # are what scope a case's rows.
    np.testing.assert_array_equal(
        np.asarray(view.handle("near_miss_offsets").materialize()), [0, 2, 4]
    )


# --- migration input -------------------------------------------------------


def _write_document(path: Path, **overrides) -> Path:
    workflows = overrides.pop(
        "workflows",
        [
            {
                "source_path": "configs/workflows/demo/demo_auto.yaml",
                "sha256": "a" * 64,
                "last_git_commit": "b" * 40,
                "normalized_workflow": {
                    "schema": "qphase.workflow/2",
                    "id": "demo_auto",
                    "jobs": [{"name": "demo_sim", "engine": {"sde": {}}}],
                },
            }
        ],
    )
    document = {
        "schema": "qphase.search-evidence/1",
        "model": "demo",
        "status": "retained_bounded_search_conclusion",
        "conclusion_report": "a bounded conclusion",
        "reproducibility_contract": {
            "goal": "reproduce the bounded conclusion",
            "workflow_semantics_embedded": True,
            "result_payloads_embedded": False,
            "legacy_runtime_compatibility_claimed": False,
        },
        "workflow_count": len(workflows),
        "workflows": workflows,
        "workspace_head_at_extraction": "c" * 40,
        "generated_at_gate": "Gate B",
        "machine_evidence_inventory": [],
        "search_scripts": [],
    }
    document.update(overrides)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_import_search_evidence_document_reads_the_history(tmp_path):
    document = import_search_evidence_document(HISTORICAL_DOCUMENT)
    assert document.document_schema == "qphase.search-evidence/1"
    assert document.model == "vdp_2mode"
    assert document.workflow_count == len(document.workflows)
    assert len(document.document_sha256) == 64
    assert document.source_path.endswith("vdp_2mode_search_evidence.json")
    assert document.reproducibility_contract["workflow_semantics_embedded"] is True
    assert document.reproducibility_contract["result_payloads_embedded"] is False
    assert document.is_migration_input is True
    mapping = document.workflow_by_source_path()
    assert len(mapping) == document.workflow_count
    assert all(entry.normalized_workflow for entry in mapping.values())


def test_import_search_evidence_document_reads_a_minimal_document(tmp_path):
    path = _write_document(tmp_path / "evidence.json")
    document = import_search_evidence_document(path)
    assert document.model == "demo"
    assert document.conclusion_report == "a bounded conclusion"
    assert len(document.workflows) == 1
    assert document.workflows[0].normalized_workflow["id"] == "demo_auto"
    # The hash is over the bytes, so it changes when the file changes.
    original = document.document_sha256
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert import_search_evidence_document(path).document_sha256 != original


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"schema": "qphase.search-evidence/2"}, "expected 'qphase.search-evidence/1'"),
        (
            {
                "reproducibility_contract": {
                    "goal": "x",
                    "workflow_semantics_embedded": False,
                    "result_payloads_embedded": False,
                    "legacy_runtime_compatibility_claimed": False,
                }
            },
            "workflow_semantics_embedded must be True",
        ),
        (
            {
                "reproducibility_contract": {
                    "goal": "x",
                    "workflow_semantics_embedded": True,
                    "result_payloads_embedded": True,
                    "legacy_runtime_compatibility_claimed": False,
                }
            },
            "result_payloads_embedded must be False",
        ),
        (
            {
                "reproducibility_contract": {
                    "goal": "x",
                    "workflow_semantics_embedded": True,
                    "result_payloads_embedded": False,
                    "legacy_runtime_compatibility_claimed": True,
                }
            },
            "legacy_runtime_compatibility_claimed must be False",
        ),
        ({"workflow_count": 2}, "workflow_count=2 but 1"),
        (
            {
                "workflows": [
                    {
                        "source_path": "configs/workflows/demo/demo_auto.yaml",
                        "sha256": "a" * 64,
                        "normalized_workflow": {
                            "schema": "qphase.workflow/2",
                            "id": "demo_auto",
                            "jobs": [],
                        },
                    }
                ]
            },
            "carries no jobs",
        ),
        (
            {
                "workflows": [
                    {
                        "source_path": "configs/workflows/demo/demo_auto.yaml",
                        "normalized_workflow": {
                            "schema": "qphase.workflow/2",
                            "id": "demo_auto",
                            "jobs": [{"name": "j"}],
                        },
                    }
                ]
            },
            "sha256",
        ),
    ],
)
def test_import_search_evidence_document_rejects_an_invalid_document(
    tmp_path, overrides, match
):
    path = _write_document(tmp_path / "evidence.json", **overrides)
    with pytest.raises(ValueError, match=match):
        import_search_evidence_document(path)


def _write_workflow_document(tmp_path: Path, workflow: dict) -> Path:
    return _write_document(
        tmp_path / "evidence.json",
        workflows=[
            {
                "source_path": "configs/workflows/demo/demo_auto.yaml",
                "sha256": "a" * 64,
                "last_git_commit": "b" * 40,
                "normalized_workflow": workflow,
            }
        ],
    )


@pytest.mark.parametrize(
    "workflow, match",
    [
        # A re-run input must name the workflow schema; absence is not "unknown".
        (
            {"id": "demo_auto", "jobs": [{"name": "demo_sim"}]},
            "declares schema None; expected exactly 'qphase.workflow/2'",
        ),
        (
            {
                "schema": None,
                "id": "demo_auto",
                "jobs": [{"name": "demo_sim"}],
            },
            "declares schema None; expected exactly 'qphase.workflow/2'",
        ),
        (
            {
                "schema": "qphase.workflow/1",
                "id": "demo_auto",
                "jobs": [{"name": "demo_sim"}],
            },
            "declares schema 'qphase.workflow/1'",
        ),
        (
            {"schema": "qphase.workflow/2", "jobs": [{"name": "demo_sim"}]},
            "carries no id",
        ),
    ],
)
def test_migration_workflow_requires_its_workflow_schema(tmp_path, workflow, match):
    path = _write_workflow_document(tmp_path, workflow)
    with pytest.raises(ValueError, match=match):
        import_search_evidence_document(path)


def test_migration_workflow_accepts_the_exact_schema(tmp_path):
    path = _write_workflow_document(
        tmp_path,
        {
            "schema": "qphase.workflow/2",
            "id": "demo_auto",
            "jobs": [{"name": "demo_sim"}],
        },
    )
    document = import_search_evidence_document(path)
    assert document.workflows[0].normalized_workflow["schema"] == "qphase.workflow/2"


def test_import_search_evidence_document_requires_missing_fields(tmp_path):
    path = _write_document(tmp_path / "evidence.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("model")
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        import_search_evidence_document(path)
    with pytest.raises(ValueError, match="not found"):
        import_search_evidence_document(tmp_path / "absent.json")


def test_migration_input_is_not_evidence():
    """The migration input exposes no counters and no conversion API."""
    document = import_search_evidence_document(HISTORICAL_DOCUMENT)
    for forbidden in (
        "accepted_count",
        "rejected_count",
        "generated_candidate_count",
        "status_counts",
    ):
        assert not hasattr(document, forbidden), forbidden
    assert "Fragment" not in type(document).__name__
    assert "import_search_evidence_document" in cam_data.__all__
    assert "build_evidence_product" in cam_data.__all__
    # No function in the public surface turns the migration input into a product.
    assert not hasattr(cam_data, "evidence_from_migration")


def test_data_package_has_the_three_10c2b_modules():
    package = Path(cam_data.__file__).parent
    for module in ("evidence", "migration"):
        assert (package / f"{module}.py").exists(), module
    root = package.parent
    # The fragment contract, the shared vocabulary and the identity helper.
    assert (root / "evidence.py").exists()
    assert (root / "outcomes.py").exists()
    assert (root / "solver" / "identity.py").exists()
