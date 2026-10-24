"""Phase 10E: explicit CAM tasks, the evidence assembler and the non-solver runtime.

These tests exercise the *active* runtime: the six profiles are enforced by the
compiler (see ``test_manifest.py``), and every task that the compiler accepts has
an implementation here — ``solve``/``scan``/``search``/``continue`` through a
solver, ``analyze`` and ``summarize_search`` without one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.artifacts import ArtifactStore
from qphase.data import ArtifactManifest, GenericDataBundle, load_bundle
from qphase_cam.data import cam_bundle
from qphase_cam.data.evidence import (
    build_evidence_product,
    fragments_from_candidate_product,
    fragments_from_evidence_product,
)
from qphase_cam.engine import Engine, EngineConfig, validate_solver_task
from qphase_cam.evidence import SearchAuditFragment, SearchAuditNearMiss
from qphase_cam.evidence_assembler import EvidenceAssembler, artifact_reference
from qphase_cam.postprocessor.physicality import Physicality
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
from qphase_cam.solver.steady_state import SteadyStateSolver

from .test_bifurcation import TwoModeFoldModel


def _fold_solver(control_max: float = 0.0) -> BifurcationSolver:
    """Return a real two-mode fold search solver."""
    return BifurcationSolver(
        BifurcationSolverConfig(
            controls={"gamma": ControlRange(min=-1.0, max=control_max)},
            perturbation=PerturbationConfig(parameter="gamma"),
        ),
        subplugins={
            "target": EquilibriumMultiplicity(EquilibriumMultiplicityConfig(order=2)),
            "strategy": FullStrategy(FullStrategyConfig()),
            "discovery": SeedDiscovery(SeedDiscoveryConfig()),
            "classifier": ScalingSignatureClassifier(ScalingSignatureConfig()),
        },
    )


class _NoopStage:
    """Context manager stand-in for one progress stage."""

    def __init__(self, stage):
        self.stage = stage

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def update(self, *args, **kwargs):
        del args, kwargs


def _reporting_context(grid=None):
    class Reporter:
        """Minimal progress reporter: the search only calls stage/status."""

        def status(self, *args, **kwargs):
            del args, kwargs

        def update(self, *args, **kwargs):
            del args, kwargs

        def stage(self, stage, **kwargs):
            del kwargs
            return _NoopStage(stage)

        def advance(self, *args, **kwargs):
            del args, kwargs

    return SimpleNamespace(parameter_grid=grid, progress=Reporter())


def _storage_config() -> SimpleNamespace:
    return SimpleNamespace(
        shard_target_mib=1, storage_layout="single", auto_shard_threshold_mib=512
    )


def _comparable(fragment: SearchAuditFragment) -> dict:
    """Return the fragment fields a product round trip must preserve exactly."""
    payload = fragment.model_dump()
    payload.pop("unique_candidate_count")
    return payload


@pytest.fixture(scope="module")
def fold_model():
    return TwoModeFoldModel()


@pytest.fixture(scope="module")
def search_bundle(fold_model):
    """One real search run through the engine (module-scoped: it is not cheap)."""
    return Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": fold_model,
            "cam_solver": _fold_solver(),
        },
    ).run(context=_reporting_context())


# --- explicit task ---------------------------------------------------------


def test_the_task_is_required_and_validated(fold_model):
    with pytest.raises(ValidationError):
        EngineConfig()
    with pytest.raises(ValidationError):
        EngineConfig(task="purge")
    with pytest.raises(TypeError, match="requires an explicit task"):
        Engine(
            plugins={
                "backend": NumpyBackend(),
                "model": fold_model,
                "cam_solver": SteadyStateSolver(method="root"),
            }
        )


def test_a_solver_task_must_match_the_grid(fold_model):
    solver = SteadyStateSolver(method="root")
    grid = object()
    validate_solver_task(solver, "solve", None)
    # A grid-capable solver still must not read a grid as an implicit scan.
    with pytest.raises(ValueError, match="does not accept a grid"):
        validate_solver_task(solver, "solve", grid)
    with pytest.raises(ValueError, match="does not implement the 'search' task"):
        validate_solver_task(solver, "search", None)


# --- search publishes evidence --------------------------------------------


def test_search_publishes_mandatory_evidence_with_a_self_reference(search_bundle):
    assert isinstance(search_bundle, GenericDataBundle)
    assert set(search_bundle.products) == {"candidates", "evidence"}
    assert search_bundle.bundle_descriptor.product_roles == {
        "candidates": "candidates",
        "evidence": "evidence",
    }
    artifact_id = search_bundle.metadata["artifact_id"]
    assert search_bundle.provenance["artifact_id"] == artifact_id
    # The raw audit payload is not repeated: the evidence product carries it.
    assert "audits" not in search_bundle.provenance
    assert search_bundle.provenance["search_audit_count"] == 1

    evidence = search_bundle.require("evidence")
    assert evidence.attributes["evidence_completeness"] == "complete"
    design = evidence.attributes["search_design"]
    assert design["strategy"] == "full"
    assert design["perturbation_parameter"] == "gamma"
    assert evidence.attributes["source_artifacts"] == [
        {"artifact_id": artifact_id, "job": None}
    ]
    cases = fragments_from_evidence_product(evidence)
    assert len(cases) == evidence.attributes["case_count"] == 1
    assert cases[0].search_identity
    assert cases[0].controls["gamma"].sampling
    # The evidence cites the candidate identities the product actually holds.
    stored = [
        str(value)
        for value in np.asarray(
            search_bundle.require("candidates")
            .handle("candidate_identity")
            .materialize()
        )
    ]
    assert list(cases[0].accepted_candidate_identities) == stored


def test_search_evidence_round_trips_through_both_layouts(search_bundle, tmp_path):
    evidence = search_bundle.require("evidence")
    original = fragments_from_evidence_product(evidence)
    bundle = cam_bundle({"evidence": evidence}, provenance={"artifact_id": "deadbeef"})
    for layout in ("single", "sharded"):
        root = tmp_path / layout
        bundle.save_dataset(root, layout=layout, shard_target_bytes=256)
        reloaded = fragments_from_evidence_product(
            load_bundle(root).require("evidence")
        )
        assert [case.model_dump() for case in reloaded] == [
            case.model_dump() for case in original
        ]


def test_search_evidence_reproduces_counts_of_the_solver_audit(
    search_bundle, fold_model
):
    evidence = search_bundle.require("evidence")
    audit = np.asarray(evidence.handle("generated_candidate_count").materialize())
    # The candidates' own case metadata carries the fragment the solver derived,
    # so a candidate-only artifact can be summarized without re-running the search.
    candidates = search_bundle.require("candidates")
    from_candidates = fragments_from_candidate_product(candidates)
    assert len(from_candidates) == len(fragments_from_evidence_product(evidence))
    assert int(audit[0]) == from_candidates[0].generated_candidate_count
    assert (
        from_candidates[0].accepted_count == evidence.attributes["accepted_start_count"]
    )


def test_search_artifact_id_is_published_under_the_declared_id(search_bundle, tmp_path):
    artifact_id = search_bundle.metadata["artifact_id"]
    store = ArtifactStore(tmp_path / "search", _storage_config())
    store.save_result(search_bundle, "search")
    manifest = ArtifactManifest.read(tmp_path / "search")
    assert manifest.artifact_id == artifact_id
    restored = load_bundle(tmp_path / "search")
    assert restored.metadata["artifact_id"] == artifact_id
    evidence = restored.require("evidence").materialize()
    assert evidence.attributes["source_artifacts"] == [
        {"artifact_id": artifact_id, "job": None}
    ]


# --- analyze ---------------------------------------------------------------


def test_analyze_annotates_an_upstream_bundle_without_a_solver(no_jacobian_model):
    state_bundle = Engine(
        EngineConfig(task="solve"),
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": SteadyStateSolver(method="root"),
        },
    ).run()
    input_artifact = state_bundle.metadata["artifact_id"]
    analysed = Engine(
        EngineConfig(task="analyze"),
        plugins={"cam_postprocessor": Physicality()},
    ).run(data=state_bundle)
    # Only the annotations are produced; the upstream products are not copied.
    assert set(analysed.products) == {"physicality"}
    assert analysed.metadata["task"] == "analyze"
    assert analysed.metadata["input_artifact_id"] == input_artifact
    assert analysed.provenance["input_artifact_id"] == input_artifact
    assert analysed.metadata["artifact_id"] == analysed.provenance["artifact_id"]
    annotation = analysed.require("physicality")
    assert annotation.attributes["annotation_kind"] == "physicality"


def _single_state_bundle(artifact_id: str = "00000000000000000000000000000001"):
    """One minimal upstream bundle for the analyze-path error tests."""
    return cam_bundle(
        {"state": _tiny_state()}, provenance={"artifact_id": artifact_id}
    )


def test_analyze_requires_an_upstream_bundle_and_a_postprocessor():
    with pytest.raises(ValueError, match="needs an upstream artifact"):
        Engine(
            EngineConfig(task="analyze"),
            plugins={"cam_postprocessor": Physicality()},
        ).run()
    with pytest.raises(RuntimeError, match="cam_postprocessor"):
        Engine(EngineConfig(task="analyze"), plugins={}).run(
            data=_single_state_bundle()
        )


def test_analyze_reports_a_postprocessor_that_needs_a_missing_model():
    class NeedsModel(Physicality):
        requires_model = True

    with pytest.raises(ValueError, match="requires a model"):
        Engine(
            EngineConfig(task="analyze"),
            plugins={"cam_postprocessor": NeedsModel()},
        ).run(data=_single_state_bundle())


def _tiny_state():
    """Return one minimal state product for the analyze-path tests."""
    from qphase_cam.data import build_state_product

    return build_state_product(
        states=np.zeros((1, 1, 2, 2), dtype=complex),
        residuals=np.zeros((1, 1)),
        success=np.ones((1, 1), dtype=bool),
        valid_mask=np.ones((1, 1), dtype=bool),
        solution_count=np.ones(1, dtype=np.int64),
        parameters={"omega": np.zeros(1)},
    )


# --- summarize_search ------------------------------------------------------


def test_summarize_reads_the_evidence_product_of_one_search(search_bundle):
    summarized = Engine(EngineConfig(task="summarize_search"), plugins={}).run(
        data=search_bundle
    )
    assert set(summarized.products) == {"evidence"}
    evidence = summarized.require("evidence")
    original = fragments_from_evidence_product(search_bundle.require("evidence"))
    rebuilt = fragments_from_evidence_product(evidence)
    # The summary reuses the recorded counts, identities and provenance.
    assert [case.model_dump() for case in rebuilt] == [
        case.model_dump() for case in original
    ]
    artifact_id = summarized.metadata["artifact_id"]
    source_ids = [
        entry["artifact_id"] for entry in evidence.attributes["source_artifacts"]
    ]
    assert source_ids == [artifact_id, search_bundle.metadata["artifact_id"]]
    assert summarized.provenance["summarized_artifact_ids"] == [
        search_bundle.metadata["artifact_id"]
    ]


def test_summarize_reads_a_candidate_only_artifact(fold_model):
    search_bundle = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": fold_model,
            "cam_solver": _fold_solver(),
        },
    ).run(context=_reporting_context())
    candidates_only = cam_bundle(
        {"candidates": search_bundle.require("candidates")},
        provenance={"artifact_id": "0123456789abcdef"},
        metadata={"task": "search"},
    )
    summarized = Engine(EngineConfig(task="summarize_search"), plugins={}).run(
        data=candidates_only
    )
    evidence = summarized.require("evidence")
    assert evidence.attributes["case_count"] == 1
    rebuilt = fragments_from_evidence_product(evidence)
    originals = fragments_from_candidate_product(search_bundle.require("candidates"))
    # ``unique_candidate_count`` is the one derived field: the product stores the
    # final identities, so the reader restates it only when it differs from the
    # accepted-start count.
    assert [_comparable(case) for case in rebuilt] == [
        _comparable(case) for case in originals
    ]
    assert [
        entry["artifact_id"] for entry in evidence.attributes["source_artifacts"]
    ] == [summarized.metadata["artifact_id"], "0123456789abcdef"]


def test_summarize_merges_several_search_artifacts(fold_model):
    from qphase.core.dataset import MappedDatasetResult

    first = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": fold_model,
            "cam_solver": _fold_solver(0.0),
        },
    ).run(context=_reporting_context())
    second = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": fold_model,
            "cam_solver": _fold_solver(-0.25),
        },
    ).run(context=_reporting_context())
    mapped = MappedDatasetResult(
        {"left": first, "right": second}, {}, (2,), {"mode": "map"}
    )
    summarized = Engine(EngineConfig(task="summarize_search"), plugins={}).run(
        data=mapped
    )
    evidence = summarized.require("evidence")
    assert evidence.attributes["case_count"] == 2
    search_identities = {
        str(value)
        for value in np.asarray(evidence.handle("search_identity").materialize())
    }
    assert len(search_identities) == 2
    # Both searches ran the same design, so it is stated once; the two search
    # identities are still recorded per case.
    assert evidence.attributes["search_design"]["strategy"] == "full"
    source_ids = {
        entry["artifact_id"] for entry in evidence.attributes["source_artifacts"]
    }
    assert source_ids == {
        first.metadata["artifact_id"],
        second.metadata["artifact_id"],
        summarized.metadata["artifact_id"],
    }


def test_summarize_keeps_a_negative_result_negative(fold_model):
    with pytest.raises(ValueError, match="needs an upstream artifact"):
        Engine(EngineConfig(task="summarize_search"), plugins={}).run()

    # A fragment that searched and accepted nothing is evidence, not a failure.
    empty = Engine(EngineConfig(task="summarize_search"), plugins={}).run(
        data=_evidence_only_bundle()
    )
    statuses = list(
        np.asarray(empty.require("evidence").handle("status").materialize())
    )
    assert statuses == ["not_found"]


def _not_found_fragment() -> SearchAuditFragment:
    return SearchAuditFragment(
        case_identity="search:9:case:0",
        search_identity="search:9",
        status="not_found",
        physical_status="not_checked",
        search_design={"target": "equilibrium_multiplicity", "order": 2},
        controls={"gamma": {"bounds": [-1.0, 0.0], "sampling": "linear"}},
        coverage={"coverage": "bordered_local_search", "seed_truncated": False},
        generated_candidate_count=3,
        prefilter_pass_count=3,
        prefilter_rejected_count=0,
        refinement_start_count=3,
        refinement_duplicate_count=0,
        accepted_count=0,
        rejected_count=3,
        rejected_by_reason={"non_simple_kernel": 3},
        near_miss_saved=1,
        near_miss_dropped=2,
        paths=(
            {
                "path": "full",
                "generation_trial_count": 3,
                "generated_candidate_count": 3,
            },
        ),
        generation_trial_count=3,
        accepted_candidate_identities=(),
        near_misses=(
            SearchAuditNearMiss(
                path="full",
                seed_source="domain_sampling",
                controls={"gamma": -0.5},
                rejection_reasons=("non_simple_kernel",),
                search_residual=1e-8,
                full_residual=2e-8,
            ),
        ),
        solver="bifurcation",
        plugins={"target": "equilibrium_multiplicity", "strategy": "full"},
        distribution={"qphase": "test"},
        api_revision="test",
        backend="numpy",
        dtype="complex128",
        notes=("negative search",),
    )


def _evidence_only_bundle() -> GenericDataBundle:
    evidence = build_evidence_product(
        fragments=[_not_found_fragment()],
        source_artifacts=[{"artifact_id": "artifact-negative"}],
        search_design={"target": "equilibrium_multiplicity", "order": 2},
    )
    return cam_bundle(
        {"evidence": evidence}, provenance={"artifact_id": "negative-search"}
    )


def test_summarize_cites_published_ids_only():
    class Anonymous:
        products = cam_bundle({"evidence": _evidence_placeholder()}).products
        metadata: dict = {}

    with pytest.raises(ValueError, match="carries no artifact_id"):
        artifact_reference(Anonymous())


def _evidence_placeholder():
    return build_evidence_product(
        fragments=[_not_found_fragment()],
        source_artifacts=[{"artifact_id": "artifact-placeholder"}],
        search_design={"target": "equilibrium_multiplicity", "order": 2},
    )


def test_summarize_requires_search_products():
    with pytest.raises(ValueError, match="needs an"):
        EvidenceAssembler(allow_multiple_searches=True).fragments_from_products(
            {"state": _tiny_state()}
        )


# --- assembler contract ----------------------------------------------------


def test_the_assembler_rejects_mixed_searches_and_duplicate_cases():
    first = _not_found_fragment()
    second = _not_found_fragment().model_copy(
        update={"case_identity": "search:9:case:1", "search_identity": "search:10"}
    )
    with pytest.raises(ValueError, match="must not mix search identities"):
        EvidenceAssembler().from_fragments([first, second], artifact_id="artifact-a")
    # A summary states the mixing explicitly, and then accepts it.
    evidence = EvidenceAssembler(allow_multiple_searches=True).from_fragments(
        [first, second], artifact_id="artifact-a"
    )
    assert evidence.attributes["case_count"] == 2
    duplicate = first.model_copy(update={"search_identity": "search:11"})
    with pytest.raises(ValueError, match="must be unique"):
        EvidenceAssembler(allow_multiple_searches=True).from_fragments(
            [first, duplicate], artifact_id="artifact-a"
        )


def test_the_assembler_rejects_a_non_fragment_and_an_empty_search():
    with pytest.raises(TypeError, match="SearchAuditFragment objects"):
        EvidenceAssembler().from_fragments(
            [{"case_identity": "x"}], artifact_id="artifact-a"
        )
    with pytest.raises(ValueError, match="must publish evidence"):
        EvidenceAssembler().from_fragments([], artifact_id="artifact-a")


def test_the_assembler_keeps_one_design_for_one_search():
    evidence = EvidenceAssembler().from_fragments(
        [_not_found_fragment()], artifact_id="artifact-a"
    )
    assert evidence.attributes["search_design"] == {
        "target": "equilibrium_multiplicity",
        "order": 2,
    }
    assert evidence.attributes["source_artifacts"] == [
        {"artifact_id": "artifact-a", "job": None}
    ]
    # The per-case control contract and the notes survive the round trip.
    cases = fragments_from_evidence_product(evidence)
    assert cases[0].controls["gamma"].bounds == (-1.0, 0.0)
    assert cases[0].controls["gamma"].sampling == "linear"
    assert cases[0].notes == ("negative search",)
    assert cases[0].near_misses[0].controls == {"gamma": -0.5}


def test_a_search_without_an_audit_fragment_fails_instead_of_publishing_nothing(
    fold_model,
):
    class SilentSolver(BifurcationSolver):
        def solve(self, request):  # pragma: no cover - wrapped for the test
            fragment = super().solve(request)
            return type(fragment)(
                products=fragment.products, audits=(), metadata=fragment.metadata
            )

    engine = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": fold_model,
            "cam_solver": SilentSolver(
                BifurcationSolverConfig(
                    controls={"gamma": ControlRange(min=-1.0, max=0.0)},
                    perturbation=PerturbationConfig(parameter="gamma"),
                ),
                subplugins={
                    "target": EquilibriumMultiplicity(
                        EquilibriumMultiplicityConfig(order=2)
                    ),
                    "strategy": FullStrategy(FullStrategyConfig()),
                    "discovery": SeedDiscovery(SeedDiscoveryConfig()),
                    "classifier": ScalingSignatureClassifier(ScalingSignatureConfig()),
                },
            ),
        },
    )
    with pytest.raises(RuntimeError, match="publishes evidence"):
        engine.run(context=_reporting_context())


# --- no legacy output ------------------------------------------------------


def test_no_task_returns_a_legacy_result(search_bundle):
    """The runtime produces typed bundles only: no result object, no dict."""
    assert not hasattr(search_bundle, "postprocess")
    assert not hasattr(search_bundle, "states")
    assert not hasattr(search_bundle, "candidates")
    assert search_bundle.metadata["task"] == "search"
    assert json.loads(json.dumps(dict(search_bundle.provenance)))["task"] == "search"
