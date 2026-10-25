import json
from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.scan import ScanSpec
from qphase.data import load_bundle
from qphase_cam.data import build_annotation_product, cam_bundle
from qphase_cam.engine import Engine, EngineConfig
from qphase_cam.evidence import (
    AUDIT_SCHEMA,
    EMPTY_RESULT_NOTE,
    REJECTION_COUNTER_SEMANTICS,
)
from qphase_cam.result import CAMResult
from qphase_cam.solver.base import CAMSolver, CAMSolverConfig, CAMSolveRequest
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
from qphase_cam.solver.bifurcation_strategy import (
    AutoStrategy,
    ReductionStrategyConfig,
)
from qphase_cam.solver.bifurcation_target import (
    EquilibriumMultiplicity,
    EquilibriumMultiplicityConfig,
)
from qphase_cam.solver.continuation import ContinuationSolver
from qphase_cam.state import (
    CAMBifurcationCandidate,
    CAMBifurcationOutput,
    CAMSolution,
    CAMSolverOutput,
)

from models.vdp_2mode import VDP2ModeModel


class BatchSolver(CAMSolver[CAMSolverConfig]):
    """A batch solver double implementing the typed protocol."""

    name = "batch_test"
    description = "batch solver double"
    config_schema = CAMSolverConfig
    supported_tasks = ("solve", "scan")
    output_products = ()
    supports_batch = True
    supports_parameter_grid = True

    def solve(self, request):
        from qphase_cam.solver.execution import state_fragment

        return state_fragment(
            request=request,
            run=lambda model, point: self.solve_batch(model),
            supports_batch=True,
        )

    def solve_batch(self, model, backend=None, point=None):
        del backend, point
        values = np.asarray(model.params["omega_a"])
        return CAMSolverOutput(
            [
                [CAMSolution(np.eye(2) * (index + 1), 0.0, True, "test")]
                for index in range(values.size)
            ]
        )


def _fake_audit(accepted: int, *, control_names=("gamma_a",)) -> dict:
    """Return a conserving audit payload for a search double.

    A case that accepted nothing still reports one generated start that the
    prefilter rejected, so its status is ``not_found`` (searched, nothing there)
    and never ``not_searched``.
    """
    generated = accepted if accepted else 1
    passed = accepted
    prefilter_rejected = generated - passed
    return {
        "schema": AUDIT_SCHEMA,
        "path": "test",
        "physical_status": "checked",
        "reductions": [
            {
                "path": "test",
                "generation_trial_count": generated,
                "generated_candidate_count": generated,
                "prefilter_pass_count": passed,
                "prefilter_rejected_count": prefilter_rejected,
                "prefilter_rejected": {},
                "refinement_start_count": accepted,
                "refinement_duplicate_count": 0,
                "accepted_count": accepted,
                "rejected_count": 0,
                "rejected_by_reason": {},
                "seed_skips": {},
                "seed_truncated": False,
                "workload": {},
            }
        ],
        "totals": {
            "generated_candidate_count": generated,
            "prefilter_pass_count": passed,
            "prefilter_rejected_count": prefilter_rejected,
            "prefilter_rejected": (
                {"prefilter": prefilter_rejected} if prefilter_rejected else {}
            ),
            "refinement_start_count": accepted,
            "refinement_duplicate_count": 0,
            "accepted_count": accepted,
            "rejected_count": 0,
            "rejected_by_reason": {},
            "near_miss_saved": 0,
            "near_miss_dropped": 0,
            "unique_candidate_count": accepted,
            "seed_skips": {},
            "seed_truncated": False,
            "workload": {},
        },
        "rejection_counter_semantics": REJECTION_COUNTER_SEMANTICS,
        "domain_coverage": {
            "control_bounds": [[-1.0, 1.0] for _ in control_names],
            "control_sampling": {name: "linear" for name in control_names},
        },
        "near_misses": [],
        "result_note": "candidates_found" if accepted else EMPTY_RESULT_NOTE,
        "control_names": control_names,
    }


def _fake_domain(control_names) -> dict:
    """Return the search-design coverage a fragment mapping needs."""
    return {
        "control_bounds": [[-1.0, 1.0] for _ in control_names],
        "control_sampling": {name: "linear" for name in control_names},
        "physical_status": "checked",
        "physical_check_stage": "generation",
    }


def _search_double(controls, order, search_impl) -> BifurcationSolver:
    """Return a search solver whose numeric core is replaced by ``search_impl``."""

    class Double(BifurcationSolver):
        def __init__(self) -> None:
            self.target = EquilibriumMultiplicity(
                EquilibriumMultiplicityConfig(order=order)
            )
            self.strategy = AutoStrategy(ReductionStrategyConfig())
            self.discovery = SeedDiscovery(SeedDiscoveryConfig())
            self.classifier = ScalingSignatureClassifier(ScalingSignatureConfig())
            self.config = BifurcationSolverConfig(
                controls=controls,
                perturbation=PerturbationConfig(parameter=next(iter(controls))),
            )
            self._reduction_cache = {}

        def search(
            self, model, backend, *, data=None, context=None, swept_parameters=()
        ):
            del backend, data, context, swept_parameters
            return search_impl(model)

    return Double()


def test_cam_engine_preserves_multidimensional_scan_shape():
    grid = ScanSpec.model_validate(
        {
            "combine": "cartesian",
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.1],
                },
                "gamma_b": {
                    "target": "model.vdp_2mode.gamma_b",
                    "values": [0.2, 0.3, 0.4],
                },
            },
        }
    ).compile()
    context = SimpleNamespace(parameter_grid=grid, progress=None)
    model = _vdp_model()

    bundle = Engine(
        EngineConfig(task="scan"),
        plugins={
            "backend": NumpyBackend(),
            "model": model,
            "cam_solver": BatchSolver(),
        }
    ).run(context=context)

    assert isinstance(bundle, DatasetResultProtocol)
    product = bundle.require("state_scan")
    assert product.attributes["grid_shape"] == [2, 3]
    assert bundle.shape == (6,)
    states = np.asarray(product.handle("states").materialize())
    assert states.shape == (6, 4, 2, 2)
    assert np.asarray(product.handle("solution_count").materialize()).shape == (6,)
    # The flattened scan carry its per-point parameter coordinates.
    np.testing.assert_allclose(
        np.asarray(product.handle("parameter.omega_a").materialize()),
        [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1],
    )
    np.testing.assert_allclose(
        np.asarray(product.handle("parameter.gamma_b").materialize()),
        [0.2, 0.3, 0.4, 0.2, 0.3, 0.4],
    )


def _fold_search_solver() -> BifurcationSolver:
    """Return a real auto-strategy search over one fold control."""
    return BifurcationSolver(
        BifurcationSolverConfig(
            controls={"gamma": ControlRange(min=-1.0, max=0.0)},
            perturbation=PerturbationConfig(parameter="gamma"),
        ),
        subplugins={
            "target": EquilibriumMultiplicity(
                EquilibriumMultiplicityConfig(order=2)
            ),
            "strategy": AutoStrategy(ReductionStrategyConfig()),
            "discovery": SeedDiscovery(SeedDiscoveryConfig()),
            "classifier": ScalingSignatureClassifier(ScalingSignatureConfig()),
        },
    )


def test_a_scan_search_keeps_one_search_identity_and_a_case_per_point():
    """A scan axis is case context: one search task, one case identity per point."""
    from .test_bifurcation import TwoModeFoldModel

    class _Stage:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        @staticmethod
        def update(*args, **kwargs):
            del args, kwargs

    class _Reporter:
        """The progress calls the solver and the scan both make."""

        @staticmethod
        def status(*args, **kwargs):
            del args, kwargs

        @staticmethod
        def update(*args, **kwargs):
            del args, kwargs

        @staticmethod
        def advance(*args, **kwargs):
            del args, kwargs

        @staticmethod
        def stage(*args, **kwargs):
            del args, kwargs
            return _Stage()

    model = TwoModeFoldModel()
    grid = ScanSpec.model_validate(
        {
            "combine": "zipped",
            "axes": {
                "kappa": {
                    "target": "model.two_mode_fold.kappa",
                    "values": [1.0, 1.1],
                }
            },
        }
    ).compile()
    bundle = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": model,
            "cam_solver": _fold_search_solver(),
        },
    ).run(
        context=SimpleNamespace(
            parameter_grid=grid,
            progress=_Reporter(),
            checkpoints=SimpleNamespace(enabled=False),
            cancellation=SimpleNamespace(raise_if_cancelled=lambda: None),
        )
    )

    evidence = bundle.require("evidence")
    search_ids = {
        str(value)
        for value in np.asarray(evidence.handle("search_identity").materialize())
    }
    case_ids = [
        str(value)
        for value in np.asarray(evidence.handle("case_identity").materialize())
    ]
    assert evidence.attributes["case_count"] == 2
    # This is the regression: the evidence assembler used to reject the mixed
    # per-case search identities and produce no product at all.
    assert len(search_ids) == 1
    assert len(set(case_ids)) == 2
    assert all(value.startswith("cam:search:") for value in search_ids)
    assert all(value.startswith("cam:case:") for value in case_ids)

    # The candidate product keeps the per-case scope of every candidate.
    candidates = bundle.require("candidates")
    case_metadata = [
        json.loads(str(value))
        for value in np.asarray(candidates.handle("case_metadata").materialize())
    ]
    assert {entry["case_identity"] for entry in case_metadata} == set(case_ids)
    offsets = np.asarray(candidates.handle("candidate_offsets").materialize())
    assert offsets.shape == (3,)
    assert int(offsets[0]) == 0
    # Per-path names survive: every audit path is the chart or the full search.
    path_audit = [
        json.loads(str(value))
        for value in np.asarray(evidence.handle("path_audit").materialize())
    ]
    for entries in path_audit:
        for entry in entries:
            assert entry["path"] == "full" or "|eq:" in entry["path"]


def test_one_scan_point_reproduces_its_case_identity():
    """The same scan value gives the same case identity on a rerun."""
    from .test_bifurcation import TwoModeFoldModel

    def identity(kappa: float) -> tuple[str, str]:
        grid = ScanSpec.model_validate(
            {
                "combine": "zipped",
                "axes": {
                    "kappa": {
                        "target": "model.two_mode_fold.kappa",
                        "values": [kappa],
                    }
                },
            }
        ).compile()
        fragment = _fold_search_solver().solve(
            CAMSolveRequest(
                task="search",
                model=TwoModeFoldModel(),
                backend=NumpyBackend(),
                parameter_grid=grid,
            )
        )
        return fragment.audits[0].search_identity, fragment.audits[0].case_identity

    left = identity(1.0)
    assert left == identity(1.0)
    assert left[0] == identity(1.1)[0]
    assert left[1] != identity(1.1)[1]


def test_cam_dataset_point_view_and_sharded_storage(tmp_path):
    """The legacy result keeps its one-off persistence path (no engine use)."""
    result = CAMResult(
        states=np.broadcast_to(np.eye(2), (4, 1, 2, 2)).copy(),
        residuals=np.zeros((4, 1)),
        success=np.ones((4, 1), dtype=bool),
        valid_mask=np.ones((4, 1), dtype=bool),
        solution_count=np.ones(4, dtype=int),
        params={"omega": np.arange(4.0)},
        axes={"omega": np.arange(4.0)},
        postprocess={"frequency": np.arange(4.0) + 0.5},
    )

    point = result.point_view((2,))
    report = result.save_dataset(
        tmp_path / "cam", layout="sharded", shard_target_bytes=1
    )

    assert point.params["omega"] == 2.0
    assert report.layout == "sharded"
    loaded = CAMResult.load_dataset(tmp_path / "cam")
    np.testing.assert_allclose(loaded.states, result.states)


def test_continuation_rejects_external_scan():
    model = _vdp_model()
    context = SimpleNamespace(
        parameter_grid=ScanSpec.model_validate(
            {
                "axes": {
                    "omega": {
                        "target": "model.vdp_2mode.omega_a",
                        "values": [0.0],
                    }
                }
            }
        ).compile(),
        progress=None,
    )

    with pytest.raises(ValueError, match="continuation"):
        Engine(
            EngineConfig(task="continue"),
            plugins={
                "backend": NumpyBackend(),
                "model": model,
                "cam_solver": ContinuationSolver(
                    parameter="gamma_b",
                    start=0.5,
                    stop=0.50001,
                    initial_guess=[[20000.0, 20000.0j], [-20000.0j, 20000.0]],
                ),
            }
        ).run(context=context)


def test_bifurcation_scan_preserves_empty_cases_and_round_trips(tmp_path):
    def search(model):
        candidates = []
        if model.params["omega_a"] > 0.0:
            candidates.append(
                CAMBifurcationCandidate(
                    state_vector=np.asarray([1.0, 1.0, 0.0, 0.0]),
                    controls={"gamma_a": 0.5},
                    full_residual_norm=1e-12,
                    search_residual_norm=2e-12,
                    success=True,
                    status="verified",
                    method="test",
                    metadata={
                        "is_physical": True,
                        "is_stable": True,
                        "maximum_jacobian_real_part": -0.5,
                        "candidate_identity": "cam:candidate:test0",
                        "identity_basis": "state_vector_and_controls",
                    },
                )
            )
        return CAMBifurcationOutput(
            candidates=candidates,
            target="equilibrium_multiplicity",
            order=2,
            metadata={
                "control_names": ("gamma_a",),
                "structural_coverage": "test",
                "numerical_coverage": "test",
                "domain_coverage": _fake_domain(("gamma_a",)),
                "audit": _fake_audit(len(candidates)),
                "search_identity": "cam:search:test",
                "case_identity": f"cam:case:test{float(model.params['omega_a'])}",
                "accepted_candidate_identities": tuple(
                    item.metadata["candidate_identity"] for item in candidates
                ),
                "accepted_start_identities": tuple(
                    f"cam:start:test{index}" for index in range(len(candidates))
                ),
            },
        )

    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.1],
                }
            }
        }
    ).compile()
    context = _scan_context(grid)
    bundle = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": _vdp_model(),
            "cam_solver": _search_double(
                {"gamma_a": ControlRange(min=-1.0, max=1.0)}, 2, search
            ),
        }
    ).run(context=context)

    product = bundle.require("candidates")
    assert product.attributes["case_count"] == 2
    assert product.attributes["candidate_count"] == 1
    offsets = np.asarray(product.handle("candidate_offsets").materialize())
    np.testing.assert_array_equal(offsets, [0, 0, 1])
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_identity").materialize()),
        ["cam:candidate:test0"],
    )
    # The empty first case keeps its own metadata row.
    metadata = [
        json.loads(str(value))
        for value in np.asarray(product.handle("case_metadata").materialize())
    ]
    assert [row["case_status"] for row in metadata] == ["complete", "complete"]
    assert metadata[1]["case_parameters"]["omega_a"] == pytest.approx(0.1)
    # The descriptors the old CSV exposed survive as typed audit JSON.
    audit = json.loads(str(np.asarray(product.handle("base_audit").materialize())[0]))
    assert audit["is_physical"] is True
    assert audit["maximum_jacobian_real_part"] == -0.5
    assert bundle.provenance["search_audit_count"] == 2
    # The audit payload lives in the evidence product, not in the provenance.
    assert "audits" not in bundle.provenance
    assert bundle.require("evidence").attributes["case_count"] == 2

    target = tmp_path / "bifurcation_scan"
    bundle.save_dataset(target, layout="sharded", shard_target_bytes=4096)
    reloaded = load_bundle(target)
    np.testing.assert_array_equal(
        np.asarray(
            reloaded.require("candidates").handle("candidate_offsets").materialize()
        ),
        [0, 0, 1],
    )


def test_bifurcation_scan_rejects_control_axis():
    def search(model):
        del model
        raise AssertionError("the control-axis check must fire before the search")

    model = _vdp_model()
    context = SimpleNamespace(
        parameter_grid=ScanSpec.model_validate(
            {
                "axes": {
                    "omega_a": {
                        "target": "model.vdp_2mode.omega_a",
                        "values": [0.0],
                    }
                }
            }
        ).compile(),
        progress=None,
    )
    with pytest.raises(ValueError, match="not solver controls"):
        Engine(
            EngineConfig(task="search"),
            plugins={
                "backend": NumpyBackend(),
                "model": model,
                "cam_solver": _search_double(
                    {"omega_a": ControlRange(min=-1.0, max=1.0)}, 2, search
                ),
            }
        ).run(context=context)


def test_bifurcation_scan_records_case_failure(tmp_path):
    def search(model):
        if model.params["omega_a"] < 0.0:
            raise ValueError("outside numerical chart")
        return CAMBifurcationOutput(
            candidates=[],
            target="equilibrium_multiplicity",
            order=3,
            metadata={
                "control_names": ("gamma_a",),
                "domain_coverage": _fake_domain(("gamma_a",)),
                "audit": _fake_audit(0),
                "search_identity": "cam:search:test",
                "case_identity": "cam:case:test",
                "accepted_candidate_identities": (),
                "accepted_start_identities": (),
            },
        )

    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.1],
                }
            }
        }
    ).compile()
    context = _scan_context(grid)
    bundle = Engine(
        EngineConfig(task="search", case_failure_policy="record"),
        plugins={
            "backend": NumpyBackend(),
            "model": _vdp_model(),
            "cam_solver": _search_double(
                {"gamma_a": ControlRange(min=-1.0, max=1.0)}, 3, search
            ),
        },
    ).run(context=context)

    metadata = [
        json.loads(str(value))
        for value in np.asarray(
            bundle.require("candidates").handle("case_metadata").materialize()
        )
    ]
    assert metadata[0]["case_status"] == "error"
    assert metadata[0]["case_error_type"] == "ValueError"
    assert metadata[0]["case_error_message"] == "outside numerical chart"
    assert metadata[0]["case_flat_index"] == 0
    assert metadata[1]["case_status"] == "complete"

    target = tmp_path / "recorded_failure"
    bundle.save_dataset(target, layout="single", shard_target_bytes=4096)
    reloaded = load_bundle(target)
    rows = [
        json.loads(str(value))
        for value in np.asarray(
            reloaded.require("candidates").handle("case_metadata").materialize()
        )
    ]
    assert rows[0]["case_status"] == "error"
    assert rows[0]["case_error_type"] == "ValueError"


def test_bifurcation_scan_resume_does_not_repeat_completed_case():
    class Checkpoints:
        enabled = True

        def __init__(self):
            self.chunks = {}

        def load_chunk(self, key):
            return self.chunks.get(key)

        def save_chunk(self, key, value):
            self.chunks[key] = value

    calls: list[float] = []
    state = {"fail_once": True}

    def search(model):
        value = float(model.params["omega_a"])
        calls.append(value)
        if value == 0.0 and state["fail_once"]:
            state["fail_once"] = False
            raise RuntimeError("transient failure")
        return CAMBifurcationOutput(
            candidates=[],
            target="equilibrium_multiplicity",
            order=3,
            metadata={
                "control_names": ("gamma_a",),
                "domain_coverage": _fake_domain(("gamma_a",)),
                "audit": _fake_audit(0),
                "search_identity": "cam:search:test",
                "case_identity": f"cam:case:test{value}",
                "accepted_candidate_identities": (),
                "accepted_start_identities": (),
            },
        )

    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.0, 0.1],
                }
            }
        }
    ).compile()
    checkpoints = Checkpoints()
    context = _scan_context(grid, checkpoints=checkpoints)
    engine = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": _vdp_model(),
            "cam_solver": _search_double(
                {"gamma_a": ControlRange(min=-1.0, max=1.0)}, 3, search
            ),
        }
    )

    with pytest.raises(RuntimeError, match="transient failure"):
        engine.run(context=context)
    bundle = engine.run(context=context)

    assert bundle.require("candidates").attributes["case_count"] == 3
    assert calls.count(-0.1) == 1
    assert calls.count(0.0) == 2
    assert calls.count(0.1) == 1


def _vdp_model():
    return VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=0.5,
        Gamma=0.0001,
        g=0.5,
    )


def _scan_context(grid, *, checkpoints=None):
    class Reporter:
        def status(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

    return SimpleNamespace(
        parameter_grid=grid,
        progress=Reporter(),
        checkpoints=checkpoints or SimpleNamespace(enabled=False),
        cancellation=SimpleNamespace(raise_if_cancelled=lambda: None),
    )


def test_cross_case_candidate_identity_stays_annotatable(tmp_path):
    """The same physical candidate in two cases keeps two annotation rows."""

    def search(model):
        value = float(model.params["omega_a"])
        return CAMBifurcationOutput(
            candidates=[
                CAMBifurcationCandidate(
                    state_vector=np.asarray([1.0, 1.0, 0.0, 0.0]),
                    controls={"gamma_a": 0.5},
                    full_residual_norm=1e-12,
                    search_residual_norm=2e-12,
                    success=True,
                    status="verified",
                    method="test",
                    metadata={
                        # One physical candidate: the same content identity in
                        # both cases, which is legal within one product.
                        "candidate_identity": "cam:candidate:shared",
                        "identity_basis": "state_vector_and_controls",
                        "accepted_start_identities": (f"cam:start:test{value}",),
                        "scaling_signatures": (_signature(),),
                    },
                )
            ],
            target="equilibrium_multiplicity",
            order=2,
            metadata={
                "control_names": ("gamma_a",),
                "domain_coverage": _fake_domain(("gamma_a",)),
                "audit": _fake_audit(1),
                "search_identity": "cam:search:test",
                "case_identity": f"cam:case:test{value}",
                "accepted_candidate_identities": ("cam:candidate:shared",),
                "accepted_start_identities": (f"cam:start:test{value}",),
            },
        )

    grid = ScanSpec.model_validate(
        {
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": [-0.1, 0.1],
                }
            }
        }
    ).compile()
    context = _scan_context(grid)
    bundle = Engine(
        EngineConfig(task="search"),
        plugins={
            "backend": NumpyBackend(),
            "model": _vdp_model(),
            "cam_solver": _search_double(
                {"gamma_a": ControlRange(min=-1.0, max=1.0)}, 2, search
            ),
        }
    ).run(context=context)
    product = bundle.require("candidates")
    # The same identity appears once per case...
    identities = list(np.asarray(product.handle("candidate_identity").materialize()))
    assert identities == ["cam:candidate:shared", "cam:candidate:shared"]
    np.testing.assert_array_equal(
        np.asarray(product.handle("candidate_offsets").materialize()), [0, 1, 2]
    )
    # ... and each row carries its own case scope.
    assert not product.attributes.get("case_scoped")

    from qphase_cam.postprocessor.candidates import candidate_view

    view = candidate_view(product)
    assert [row.case for row in view.rows] == [0, 1]
    assert [row.case_identity for row in view.rows] == [
        "cam:case:test-0.1",
        "cam:case:test0.1",
    ]
    # Start identities differ across the cases even for the same candidate.
    audits = [
        json.loads(str(value))
        for value in np.asarray(product.handle("base_audit").materialize())
    ]
    assert [audit["accepted_start_identities"] for audit in audits] == [
        ["cam:start:test-0.1"],
        ["cam:start:test0.1"],
    ]

    annotation = build_annotation_product(
        annotation_kind="local_response_validation",
        target_kind="candidate",
        target_identity=[row.identity for row in view.rows],
        case_identity=[row.case_identity for row in view.rows],
        status=["valid", "valid"],
        fields={"margin": np.asarray([0.5, 0.25])},
    )
    assert annotation.attributes["case_scoped"] is True
    bundle = cam_bundle(
        {"candidates": product, "local_response_validation": annotation}
    )
    target = tmp_path / "cross_case"
    bundle.save_dataset(target, layout="sharded", shard_target_bytes=256)
    restored = load_bundle(target).require("local_response_validation")
    assert list(np.asarray(restored.handle("case_identity").materialize())) == [
        "cam:case:test-0.1",
        "cam:case:test0.1",
    ]
    np.testing.assert_array_equal(
        np.asarray(restored.handle("target_identity").materialize()),
        ["cam:candidate:shared", "cam:candidate:shared"],
    )


def _kappa_grid(values: list[float]):
    """Return a one-axis kappa scan grid for the two-mode fold model."""
    return ScanSpec.model_validate(
        {
            "combine": "zipped",
            "axes": {
                "kappa": {
                    "target": "model.two_mode_fold.kappa",
                    "values": list(values),
                }
            },
        }
    ).compile()


def _kappa_search_request(grid, *, policy: str = "record") -> CAMSolveRequest:
    from .test_bifurcation import TwoModeFoldModel

    return grid, CAMSolveRequest(
        task="search",
        model=TwoModeFoldModel(),
        backend=NumpyBackend(),
        parameter_grid=grid,
        case_failure_policy=policy,
    )


def test_a_scan_search_restores_the_caller_model_parameters():
    """A grid leaves its per-point arrays on no caller model, after every policy."""
    from .test_bifurcation import TwoModeFoldModel

    for policy in ("record", "abort"):
        grid, request = _kappa_search_request(_kappa_grid([1.0, 1.1]), policy=policy)
        model = request.model
        entered = dict(model.params)
        fragment = _fold_search_solver().solve(request)
        # The caller's own model keeps its keys and scalar values.
        assert dict(model.params) == entered
        assert not any(np.ndim(value) for value in model.params.values())
        # A second instance of the same class is not polluted by the scan.
        assert dict(TwoModeFoldModel().params) == entered
        # The artifact still carries the per-case kappa values.
        product = fragment.products["candidates"].materialize()
        np.testing.assert_allclose(
            np.asarray(product.handle("case_parameter.kappa").materialize()),
            [1.0, 1.1],
        )
        assert list(
            np.asarray(product.handle("parameter.kappa").materialize())
        ) == [1.0, 1.1]


def _omega_a_grid(values: list[float]):
    return ScanSpec.model_validate(
        {
            "combine": "zipped",
            "axes": {
                "omega_a": {
                    "target": "model.vdp_2mode.omega_a",
                    "values": list(values),
                }
            },
        }
    ).compile()


def _failing_search_solver(message: str) -> BifurcationSolver:
    """Return a search double that fails on the second case it is given."""

    def search(model):
        if model.params["omega_a"] < 0.0:
            raise ValueError(message)
        return CAMBifurcationOutput(
            candidates=[],
            target="equilibrium_multiplicity",
            order=3,
            metadata={
                "control_names": ("gamma_a",),
                "domain_coverage": _fake_domain(("gamma_a",)),
                "audit": _fake_audit(0),
                "search_identity": "cam:search:test",
                "case_identity": "cam:case:test",
                "accepted_candidate_identities": (),
                "accepted_start_identities": (),
            },
        )

    return _search_double({"gamma_a": ControlRange(min=-1.0, max=1.0)}, 3, search)


def test_a_recorded_failing_scan_case_restores_the_caller_model_parameters():
    """A recorded case failure leaves the caller's model exactly as it was."""
    grid = _omega_a_grid([-0.1, 0.1])
    model = _vdp_model()
    entered = dict(model.params)
    fragment = _failing_search_solver("outside numerical chart").solve(
        CAMSolveRequest(
            task="search",
            model=model,
            backend=NumpyBackend(),
            parameter_grid=grid,
            case_failure_policy="record",
        )
    )
    assert dict(model.params) == entered
    assert not any(np.ndim(value) for value in model.params.values())
    assert dict(_vdp_model().params) == entered
    product = fragment.products["candidates"].materialize()
    np.testing.assert_allclose(
        np.asarray(product.handle("case_parameter.omega_a").materialize()),
        [-0.1, 0.1],
    )
    metadata = [
        json.loads(str(value))
        for value in np.asarray(product.handle("case_metadata").materialize())
    ]
    assert [entry["case_status"] for entry in metadata] == ["error", "complete"]


def test_an_aborted_scan_case_restores_the_caller_model_parameters():
    """An aborting case raises out of the scan but still restores the model."""
    grid = _omega_a_grid([-0.1, 0.1])
    model = _vdp_model()
    entered = dict(model.params)
    with pytest.raises(ValueError, match="synthetic abort"):
        _failing_search_solver("synthetic abort").solve(
            CAMSolveRequest(
                task="search",
                model=model,
                backend=NumpyBackend(),
                parameter_grid=grid,
                case_failure_policy="abort",
            )
        )
    assert dict(model.params) == entered
    assert not any(np.ndim(value) for value in model.params.values())
    assert dict(_vdp_model().params) == entered


class _GridModel:
    """A model double with class-level parameters, like the fold test model."""

    name = "grid_model"
    n_modes = 2
    steady_state_capacity = 2
    params = {"rate": 1.0, "gamma": -1.0}

    @staticmethod
    def cam_solution_sort_key(state, params):
        del params
        return float(np.real(np.asarray(state)[0, 0]))


def test_a_state_scan_restores_the_caller_model_parameters():
    """A pointwise state scan does not leave its grid arrays on the model."""
    from qphase_cam.solver.execution import state_fragment

    grid = ScanSpec.model_validate(
        {
            "combine": "zipped",
            "axes": {
                "rate": {"target": "model.grid_model.rate", "values": [1.0, 1.1]}
            },
        }
    ).compile()
    model = _GridModel()
    entered = dict(model.params)

    def run(current, point):
        del point
        value = float(current.params["rate"])
        return CAMSolverOutput(
            [
                CAMSolution(
                    state=np.eye(2) * value,
                    residual=0.0,
                    success=True,
                    method="test",
                )
            ]
        )

    fragment = state_fragment(
        request=CAMSolveRequest(
            task="scan",
            model=model,
            backend=NumpyBackend(),
            parameter_grid=grid,
        ),
        run=run,
        supports_batch=False,
    )
    assert dict(model.params) == entered
    assert not any(np.ndim(value) for value in model.params.values())
    assert dict(_GridModel().params) == entered
    product = fragment.products["state_scan"].materialize()
    np.testing.assert_allclose(
        np.asarray(product.handle("parameter.rate").materialize()), [1.0, 1.1]
    )
    np.testing.assert_allclose(
        np.asarray(product.handle("states").materialize())[:, 0, 0, 0], [1.0, 1.1]
    )


def _signature() -> dict:
    return {
        "state_order": 3,
        "perturbation_order": 1,
        "coupling_state_order": 0,
        "exponent_numerator": 1,
        "exponent_denominator": 3,
        "exponent": 1.0 / 3.0,
        "sublinear": True,
        "branches": [
            {
                "epsilon_side": 1,
                "amplitude": 1.0,
                "leading_state_coefficient": [[1.0]],
            }
        ],
    }
