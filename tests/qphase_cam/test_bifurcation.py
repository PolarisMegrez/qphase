"""Bifurcation plugin contracts and tagged result tests."""

from __future__ import annotations

import importlib.util
import json
import time
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.protocols import ResultProtocol
from qphase.data import load_bundle
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.core.reduction import (
    CondensedScalarReduction,
    ReductionDiagnostics,
    SeedGenerationStats,
)
from qphase_cam.data import build_candidate_product, cam_bundle
from qphase_cam.engine import Engine
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.candidates import candidate_view
from qphase_cam.postprocessor.local_response import LocalResponseValidation
from qphase_cam.postprocessor.stochastic_validity import StochasticValidity
from qphase_cam.result import CAMResult
from qphase_cam.solver.base import (
    SOLVER_OUTPUT_DECLARATIONS,
    CAMProductFragment,
    CAMSolver,
    CAMSolverConfig,
)
from qphase_cam.solver.bifurcation import (
    BifurcationSolver,
    BifurcationSolverConfig,
    ControlRange,
    PerturbationConfig,
)
from qphase_cam.solver.bifurcation_audit import (
    EMPTY_RESULT_NOTE,
    REJECTION_COUNTER_SEMANTICS,
    NearMissStore,
    near_miss_json,
)
from qphase_cam.solver.bifurcation_classifier import (
    ScalingSignatureClassifier,
    ScalingSignatureConfig,
)
from qphase_cam.solver.bifurcation_discovery import (
    SeedDiscovery,
    SeedDiscoveryConfig,
)
from qphase_cam.solver.bifurcation_strategy import (
    AutoStrategy,
    FullStrategy,
    FullStrategyConfig,
    ReducedStrategy,
    ReductionStrategyConfig,
)
from qphase_cam.solver.bifurcation_target import (
    EquilibriumMultiplicity,
    EquilibriumMultiplicityConfig,
)
from qphase_cam.solver.execution import candidate_product_from_output
from qphase_cam.state import CAMBifurcationCandidate, CAMBifurcationOutput
from qpsgen import (
    LindbladChannel,
    MasterEquation,
    boson_modes,
    derive_kramers_moyal,
)

from models.vdp_2mode import VDP2ModeModel


class TwoModeFoldModel:
    """One nonlinear mode plus a decoupled linear mode with a known fold."""

    name = "two_mode_fold"
    n_modes = 2
    steady_state_capacity = 2
    params = {"gamma": -1.0, "Gamma": 1.0, "kappa": 1.0}

    @staticmethod
    def cam_hamiltonian(state, params):
        del state, params
        return np.zeros((2, 2), dtype=complex)

    @classmethod
    @lru_cache(maxsize=1)
    def cam_qpsgen_dynamics(cls):
        a, b = boson_modes("a", "b")
        gamma, nonlinear_gain, kappa = sp.symbols("gamma Gamma kappa", real=True)
        master = MasterEquation(
            modes=(a, b),
            channels=(
                LindbladChannel(a.dag, gamma),
                LindbladChannel(a**2, nonlinear_gain),
                LindbladChannel(b, kappa),
            ),
        )
        return (
            derive_kramers_moyal(master, "wigner")
            .truncate(2)
            .to_langevin()
            .to_second_moment_dynamics(
                parameters=(gamma, nonlinear_gain, kappa),
                layout="normal",
                closure="factorized_bilinear",
            )
        )


def _solver(order: int, controls: dict[str, ControlRange]) -> BifurcationSolver:
    target = EquilibriumMultiplicity(EquilibriumMultiplicityConfig(order=order))
    return BifurcationSolver(
        BifurcationSolverConfig(
            controls=controls,
            perturbation=PerturbationConfig(parameter=next(iter(controls))),
        ),
        subplugins={
            "target": target,
            "strategy": AutoStrategy(ReductionStrategyConfig()),
            "discovery": SeedDiscovery(SeedDiscoveryConfig()),
            "classifier": ScalingSignatureClassifier(ScalingSignatureConfig()),
        },
    )


def test_bifurcation_solver_validates_target_codimension():
    solver = _solver(2, {"control": ControlRange(min=-1.0, max=1.0)})
    assert solver.output_kind == "bifurcation_candidates"
    assert solver.target.order == 2
    with pytest.raises(ValueError, match="m-1 controls"):
        _solver(3, {"control": ControlRange(min=-1.0, max=1.0)})


def test_nonnegative_model_control_rejects_negative_bounds():
    solver = _solver(2, {"gamma_b": ControlRange(min=-0.1, max=1.0)})
    model = VDP2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        gamma_a=0.5,
        gamma_b=0.8,
        Gamma=0.01,
        g=0.5,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        solver.search(model, NumpyBackend())


def test_old_bifurcation_postprocessor_is_removed():
    assert importlib.util.find_spec("qphase_cam.postprocessor.bifurcation") is None


def test_fraction_free_solver_finds_known_physical_double_root():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    output = solver.search(TwoModeFoldModel(), NumpyBackend())
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    expected_gamma = -6.0 + np.sqrt(28.0)
    expected_q = (expected_gamma + 4.0) / 4.0
    np.testing.assert_allclose(candidate.controls["gamma"], expected_gamma)
    np.testing.assert_allclose(candidate.state_vector, [expected_q, 0.5, 0.0, 0.0])
    assert candidate.full_residual_norm < 1e-10
    assert candidate.status == "verified"
    assert candidate.metadata["verification_digits"] >= 50
    assert candidate.metadata["verification_status"] == "verified"
    assert candidate.metadata["multiplicity_residual_norm"] < 1e-30
    assert candidate.metadata["verified_full_residual_norm"] < 1e-30
    assert len(candidate.metadata["verified_full_state_decimal_values"]) == 4
    assert "verification_residual_norm" not in candidate.metadata
    assert candidate.metadata["is_physical"]
    signature = candidate.metadata["scaling_signatures"][0]
    assert (
        signature["state_order"],
        signature["perturbation_order"],
        signature["coupling_state_order"],
    ) == (2, 1, 0)
    assert signature["exponent"] == pytest.approx(0.5)
    assert output.metadata["reduced_degree"] == 2
    assert output.metadata["closure"]["representation"] == "wigner"
    assert not output.metadata["closure"]["fpe_is_exact"]
    assert not output.metadata["closure"]["moment_closure_is_exact"]
    assert not output.metadata["closure"]["deterministic_cam_is_exact"]
    assert (
        output.metadata["coverage"]
        == "regular_reduction_exhaustive_and_bordered_local_search"
    )
    search = output.metadata["reduction_search"]
    assert search["coverage"] == "exhaustive"
    assert search["regular_branches_only"]
    assert not search["returned_prefix"]
    # auto path merges the reduced and full-fallback audits by unit
    audit = output.metadata["audit"]
    totals = audit["totals"]
    full_audit = output.metadata["full_fallback"]["audit"]
    assert full_audit["seed_source"] == "domain_sampling"
    assert full_audit["physical_check_stage"] == "generation"
    assert full_audit["workload"]["fixed_point_guess_count"] > 0
    assert totals["generated_candidate_count"] == (
        sum(entry["generated_candidate_count"] for entry in audit["reductions"])
        + full_audit["generated_candidate_count"]
    )
    assert totals["refinement_start_count"] == (
        totals["accepted_count"]
        + totals["rejected_count"]
        + totals["refinement_duplicate_count"]
    )
    assert (
        totals["near_miss_saved"] + totals["near_miss_dropped"]
        == totals["rejected_count"]
    )
    assert set(totals["workload"]) >= {
        "control_point_count",
        "fixed_point_guess_count",
    }


def test_reduction_strategy_exposes_qpsgen_work_budgets():
    config = ReductionStrategyConfig(
        partition_limit=12,
        materialization_limit=4,
        max_candidates=3,
    )
    assert config.partition_limit == 12
    assert config.materialization_limit == 4
    assert config.max_candidates == 3


def test_reduction_search_reports_budget_limited_coverage():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    solver.strategy = AutoStrategy(ReductionStrategyConfig(partition_limit=4))
    output = solver.search(TwoModeFoldModel(), NumpyBackend())
    assert (
        output.metadata["coverage"]
        == "regular_reduction_budget_limited_and_bordered_local_search"
    )
    assert not output.metadata["reduction_search"]["complete"]


def test_reduction_structure_is_cached_between_fixed_parameter_cases():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    adapter = QPSGenDynamicsAdapter.from_model(TwoModeFoldModel())
    first_reductions, first = solver._select_reductions(adapter)
    adapter.model.params = {**adapter.model.params, "kappa": 2.0}
    second_reductions, second = solver._select_reductions(adapter)
    assert first["compile_cache_hit"] is False
    assert second["compile_cache_hit"] is True
    assert first_reductions[0] is not second_reductions[0]
    assert first_reductions[0].base_params["kappa"] == 1.0
    assert second_reductions[0].base_params["kappa"] == 2.0
    cached_reductions, _ = next(iter(solver._reduction_cache.values()))
    assert cached_reductions[0] is not first_reductions[0]
    assert cached_reductions[0] is not second_reductions[0]


def test_reduction_start_filter_rejects_singular_reconstruction():
    class SingularReduction:
        @staticmethod
        def reconstruct(start):
            del start
            raise ZeroDivisionError

    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    assert not solver._reduced_start_is_physical(
        SingularReduction(), object(), np.zeros(2)
    )


def test_full_solver_refines_upstream_fixed_point_to_known_fold():
    expected_gamma = -6.0 + np.sqrt(28.0)
    expected_q = (expected_gamma + 4.0) / 4.0
    upstream = CAMResult(
        states=np.asarray([[[expected_q, 0.0], [0.0, 0.5]]]),
        residuals=np.asarray([0.0]),
        success=np.asarray([True]),
        valid_mask=np.asarray([True]),
        solution_count=np.asarray(1),
        params={
            "gamma": expected_gamma,
            "Gamma": 1.0,
            "kappa": 1.0,
        },
    )
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
    output = solver.search(TwoModeFoldModel(), NumpyBackend(), data=upstream)
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    np.testing.assert_allclose(candidate.controls["gamma"], expected_gamma)
    np.testing.assert_allclose(
        candidate.state_vector, [expected_q, 0.5, 0.0, 0.0], atol=1e-14
    )
    assert candidate.method == "bordered_full"
    assert candidate.status == "verified"
    assert candidate.metadata["verification_status"] == "verified"
    assert output.metadata["coverage"] == "upstream_bordered_local_search"


def test_full_seed_discovery_preserves_distinct_control_points(monkeypatch):
    solver = _solver(
        3,
        {
            "gamma_a": ControlRange(min=0.1, max=1.0),
            "gamma_b": ControlRange(min=0.2, max=2.0),
        },
    )
    solver.discovery = SeedDiscovery(
        SeedDiscoveryConfig(samples_per_control=2, max_starts=16)
    )

    class Adapter:
        state_size = 4
        diagonal_state_indices = (0, 1)
        model = SimpleNamespace(params={"k": 1e-3})

        @staticmethod
        def rhs(state, params):
            del state, params
            return np.zeros(4)

        @staticmethod
        def jacobian(state, params):
            del state, params
            return np.eye(4)

    class System:
        @staticmethod
        def seed(state, controls):
            del state
            return np.asarray([controls["gamma_a"], controls["gamma_b"]])

    monkeypatch.setattr(
        "qphase_cam.solver.bifurcation.root",
        lambda *args, **kwargs: SimpleNamespace(success=True, x=np.zeros(4)),
    )
    monkeypatch.setattr(solver, "_state_is_physical", lambda *args: True)

    starts = solver._discover_full_seeds(System(), Adapter())
    assert len(starts) == 4
    assert len({tuple(item) for item in starts}) == 4


def test_pair_hopping_seed_scale_tracks_inverse_nonlinearity():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    adapter = SimpleNamespace(
        state_size=4,
        diagonal_state_indices=(0, 1),
        model=SimpleNamespace(
            params={"nonlinearity": 1e-4},
            cam_bifurcation_scales=lambda params: {
                "state": [1.0 / params["nonlinearity"]] * 4,
                "source": "test",
            },
        ),
    )
    guesses = solver._state_guesses(adapter)
    np.testing.assert_allclose(guesses[-1][[0, 1]], 1e4)


def test_condensed_reduction_verifies_known_fold_at_high_precision():
    model = TwoModeFoldModel()
    adapter = QPSGenDynamicsAdapter.from_model(model)
    plan = adapter.linear_reduction(
        candidate=adapter.search_linear_reductions(retained_dimension=1).candidates[0]
    )
    reduction = CondensedScalarReduction(
        plan,
        order=2,
        control_names=("gamma",),
        base_params=model.params,
    )
    expected_gamma = -6.0 + np.sqrt(28.0)
    expected_q = (expected_gamma + 4.0) / 4.0
    outcome = reduction.verify(
        [expected_q + 1e-5, expected_gamma - 1e-5],
        initial_digits=50,
        max_digits=100,
    )
    assert outcome.success
    assert outcome.digits == 50
    assert outcome.multiplicity_residual_norm < 1e-30
    assert outcome.full_residual_norm < 1e-30
    assert len(outcome.full_state_decimal_values) == 4
    np.testing.assert_allclose(outcome.value, [expected_q, expected_gamma])
    state_vector = reduction.reconstruct(outcome.value)
    classification = ScalingSignatureClassifier(ScalingSignatureConfig()).classify(
        reduction,
        outcome.value,
        state_vector,
        {**model.params, "gamma": expected_gamma},
        adapter,
        perturbation="gamma",
        scale=1.0,
        side="both",
        verification_digits=outcome.digits,
    )
    assert classification["classification_status"] == "classified", classification
    signature = classification["scaling_signatures"][0]
    assert (
        signature["state_order"],
        signature["perturbation_order"],
        signature["coupling_state_order"],
    ) == (2, 1, 0)
    assert signature["exponent"] == pytest.approx(0.5)


def test_engine_packs_tagged_bifurcation_output_and_passes_input(
    no_jacobian_model,
):
    upstream = object()

    class Solver(CAMSolver[CAMSolverConfig]):
        name = "bifurcation"
        description = "search double"
        supported_tasks = ("search",)
        output_products = (SOLVER_OUTPUT_DECLARATIONS["candidates"],)
        search_audit_capability = True
        config_schema = CAMSolverConfig
        target = SimpleNamespace(name="equilibrium_multiplicity", order=2)
        config = SimpleNamespace(controls={"control": object()})

        def solve(self, request):
            from qphase_cam.solver.execution import candidate_product_from_output

            assert request.model is no_jacobian_model
            output = self.search(request.model, request.backend, data=request.data)
            product = candidate_product_from_output(request.model, output)
            return CAMProductFragment(
                products={"candidates": product},
                metadata={"control_names": ("control",)},
            )

        def search(self, model, backend, *, data=None):
            del model, backend
            assert data is upstream
            return CAMBifurcationOutput(
                candidates=[
                    CAMBifurcationCandidate(
                        state_vector=np.array([1.0, 1.0, 0.0, 0.0]),
                        controls={"control": 0.5},
                        full_residual_norm=1e-12,
                        search_residual_norm=2e-12,
                        success=True,
                        status="candidate",
                        method="test",
                        metadata={
                            "regularity_determinant": 2.0,
                            "candidate_identity": "cam:candidate:test",
                            "identity_basis": "state_vector_and_controls",
                        },
                    )
                ],
                target="equilibrium_multiplicity",
                order=2,
                metadata={"control_names": ("control",)},
            )

    bundle = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": Solver(),
        }
    ).run(data=upstream)
    assert isinstance(bundle, ResultProtocol)
    assert isinstance(bundle, DatasetResultProtocol)
    product = bundle.require("candidates")
    assert bundle.bundle_descriptor.product_roles == {"candidates": "candidates"}
    np.testing.assert_allclose(
        np.asarray(product.handle("control_values").materialize()), [[0.5]]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("verification_digits").materialize()), [0]
    )
    np.testing.assert_array_equal(
        np.asarray(product.handle("verification_status").materialize()), ["not_run"]
    )
    audit = json.loads(str(np.asarray(product.handle("base_audit").materialize())[0]))
    assert audit["regularity_determinant"] == 2.0


def test_engine_persists_scaling_branch_table(tmp_path):
    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    product = candidate_product_from_output(model, output)
    view = candidate_view(product)
    assert view.branches.size == 2
    # The scaling branch structure arrives as typed annotation input derived from
    # the candidate product's classification JSON.
    assert view.branches.exponent_denominator[0] == 2
    assert sum(len(row.classification["scaling_signatures"]) for row in view.rows) == 1
    audit = json.loads(str(np.asarray(product.handle("base_audit").materialize())[0]))
    assert "maximum_jacobian_real_part" in audit
    assert len(audit["verified_full_state_decimal_values"]) == 4
    # The typed product is the persisted form.
    target = tmp_path / "classified"
    cam_bundle({"candidates": product}).save_dataset(
        target, layout="single", shard_target_bytes=4096
    )
    loaded = candidate_view(load_bundle(target).require("candidates"))
    np.testing.assert_allclose(
        np.asarray([np.real(item) for item in loaded.branches.amplitude]),
        np.asarray([np.real(item) for item in view.branches.amplitude]),
    )


def test_engine_packs_candidates_with_ragged_scaling_metadata(tmp_path):
    model = TwoModeFoldModel()
    signature = {
        "state_order": 3,
        "perturbation_order": 1,
        "coupling_state_order": 0,
        "exponent_numerator": 1,
        "exponent_denominator": 3,
        "exponent": 1.0 / 3.0,
        "sublinear": True,
        "branches": (),
    }
    output = CAMBifurcationOutput(
        candidates=[
            CAMBifurcationCandidate(
                state_vector=np.asarray([1.0, 0.5, 0.0, 0.0]),
                controls={"gamma": -0.5},
                full_residual_norm=0.0,
                search_residual_norm=0.0,
                success=True,
                status="verified",
                method="test",
                metadata={
                    "scaling_signatures": (signature,),
                    "candidate_identity": "cam:candidate:a",
                },
            ),
            CAMBifurcationCandidate(
                state_vector=np.asarray([2.0, 0.5, 0.0, 0.0]),
                controls={"gamma": -0.25},
                full_residual_norm=0.0,
                search_residual_norm=0.0,
                success=True,
                status="verified",
                method="test",
                metadata={
                    "scaling_signatures": (),
                    "candidate_identity": "cam:candidate:b",
                },
            ),
        ],
        target="equilibrium_multiplicity",
        order=3,
        metadata={"control_names": ("gamma",)},
    )
    product = candidate_product_from_output(model, output)
    classification = [
        json.loads(str(value))
        for value in np.asarray(product.handle("classification").materialize())
    ]
    # The ragged signature tuple becomes canonical JSON per candidate.
    assert [len(row["scaling_signatures"]) for row in classification] == [1, 0]
    target = tmp_path / "ragged"
    cam_bundle({"candidates": product}).save_dataset(
        target, layout="single", shard_target_bytes=4096
    )
    loaded = [
        json.loads(str(value))
        for value in np.asarray(
            load_bundle(target)
            .require("candidates")
            .handle("classification")
            .materialize()
        )
    ]
    assert [len(row["scaling_signatures"]) for row in loaded] == [1, 0]


def test_local_response_validation_solves_and_persists_complete_branches(tmp_path):
    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    product = candidate_product_from_output(model, output)
    processor = LocalResponseValidation(
        epsilon_min=1e-10,
        epsilon_max=1e-6,
        epsilon_points=4,
        fit_points=3,
        residual_tolerance=1e-25,
    )
    annotation = processor.process(
        product,
        CAMAnnotationContext(
            model=model,
            backend=NumpyBackend(),
            metadata={
                "perturbation_parameter": "gamma",
                "perturbation_scale": 1.0,
                "fixed_params": {},
            },
            products={"candidates": product},
            source_role="candidates",
        ),
    )
    converged = np.asarray(annotation.handle("converged").materialize())
    continuous = np.asarray(annotation.handle("continuous").materialize())
    residuals = np.asarray(annotation.handle("full_residual_norm").materialize())
    state_fit = np.asarray(annotation.handle("state_fit_exponent").materialize())
    petermann = np.asarray(annotation.handle("hamiltonian_petermann_max").materialize())
    petermann_fit = np.asarray(
        annotation.handle("hamiltonian_petermann_fit_exponent").materialize()
    )
    # One candidate, two real branches, four epsilon samples each.
    assert converged.shape == (1, 2, 4)
    assert int(converged.sum()) == 8
    assert np.all(converged)
    assert np.all(continuous)
    assert np.max(residuals) < 1e-25
    np.testing.assert_allclose(state_fit, 0.5, atol=1e-3)
    np.testing.assert_allclose(petermann, 1.0)
    np.testing.assert_allclose(petermann_fit, 0.0, atol=1e-12)
    # The per-candidate summary the old table exposed is derived, not lost.
    sample_count = (
        annotation.axis("branch").size * annotation.axis("epsilon_sample").size
    )
    assert [sample_count] == [8]
    np.testing.assert_allclose(float(np.min(petermann)), 1.0)
    np.testing.assert_allclose(float(np.max(petermann)), 1.0)
    assert list(np.asarray(annotation.handle("status").materialize())) == ["valid"]
    target_identity = np.asarray(annotation.handle("target_identity").materialize())
    np.testing.assert_array_equal(
        target_identity,
        np.asarray(product.handle("candidate_identity").materialize()),
    )

    target = tmp_path / "response"
    cam_bundle(
        {"candidates": product, "local_response_validation": annotation}
    ).save_dataset(target, layout="single", shard_target_bytes=4096)
    loaded = load_bundle(target).require("local_response_validation")
    np.testing.assert_allclose(
        np.asarray(loaded.handle("delta_state_norm").materialize()),
        np.asarray(annotation.handle("delta_state_norm").materialize()),
    )
    np.testing.assert_allclose(
        np.asarray(loaded.handle("hamiltonian_petermann_max").materialize()),
        petermann,
    )


def _branch_signature(*, epsilon_side=1, amplitude=1.0) -> dict:
    """Return one classification signature with a single real branch."""
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
                "epsilon_side": epsilon_side,
                "amplitude": amplitude,
                "leading_state_coefficient": [[1.0]],
            }
        ],
    }


def _stochastic_product(*, candidates: int):
    """Return a candidate product standing in for a classified search case."""
    identity = [f"cam:candidate:test{index}" for index in range(candidates)]
    return build_candidate_product(
        states=np.asarray(
            [np.asarray([[1.0 + 0.0j]]) for _ in range(candidates)]
        ).reshape(candidates, 1, 1),
        state_vectors=np.asarray([[1.0]] * candidates).reshape(candidates, 1),
        control_values=np.empty((candidates, 0)),
        control_names=(),
        full_residual_norm=np.zeros(candidates),
        search_residual_norm=np.zeros(candidates),
        success=np.ones(candidates, dtype=bool),
        status=["verified"] * candidates,
        method=["test"] * candidates,
        verification_digits=np.full(candidates, 50, dtype=np.int64),
        verification_status=["verified"] * candidates,
        candidate_identity=identity,
        classification=[
            {"scaling_signatures": [_branch_signature()]} for _ in range(candidates)
        ],
    )


def test_stochastic_validity_estimates_additive_cubic_crossover(monkeypatch, tmp_path):
    class Adapter:
        moment_layout = "normal"

        @staticmethod
        def state_matrix(state, params):
            del params
            return np.asarray([[state[0]]], dtype=complex)

        @staticmethod
        def jacobian(state, params):
            del state, params
            return np.asarray([[0.0]])

        @staticmethod
        def parameter_jacobian(state, params):
            del state, params
            return np.asarray([[1.0]])

        @staticmethod
        def parameter_direction(name, scale=1.0):
            assert name == "mu"
            return np.asarray([scale])

        @staticmethod
        def closure_provenance():
            return {
                "representation": "wigner",
                "fpe_is_exact": False,
                "moment_closure": "factorized_bilinear",
                "moment_closure_is_exact": False,
                "deterministic_cam_is_exact": False,
            }

    class Model:
        params = {"mu": 0.0}

        @staticmethod
        def cam_diffusion(state, params):
            del state, params
            return np.asarray([[1.0]])

    monkeypatch.setattr(
        "qphase_cam.postprocessor.stochastic_validity.QPSGenDynamicsAdapter.from_model",
        lambda model: Adapter(),
    )
    product = _stochastic_product(candidates=1)
    processor = StochasticValidity(probe_epsilon=1.0)
    annotation = processor.process(
        product,
        CAMAnnotationContext(
            model=Model(),
            backend=NumpyBackend(),
            metadata={
                "perturbation_parameter": "mu",
                "perturbation_scale": 1.0,
                "closed_state_capability": True,
            },
            products={"candidates": product},
            source_role="candidates",
        ),
    )
    branch_status = np.asarray(annotation.handle("branch_status").materialize())
    assert branch_status.tolist() == [["complete"]]
    np.testing.assert_allclose(
        np.asarray(annotation.handle("projected_noise_intensity").materialize()),
        [[2.0]],
    )
    np.testing.assert_allclose(
        np.asarray(annotation.handle("normal_form_state_coefficient").materialize()),
        [[-1.0]],
    )
    np.testing.assert_allclose(
        np.asarray(annotation.handle("epsilon_crossover").materialize()),
        [[2.0 * np.sqrt(2.0)]],
    )
    assert np.asarray(annotation.handle("regime").materialize()).tolist() == [
        ["noise_dominated"]
    ]
    # Every field of the old diagnostic table survives as a typed column.
    from qphase_cam.postprocessor import stochastic_validity as validity

    expected = set(
        validity.StochasticValidity._base_row(
            0, branch_index=0, epsilon_side=0, closure={}, status=""
        )
    )
    fields = {item["name"] for item in annotation.attributes["field_variables"]}
    assert {name for name in expected if name != "status"} <= fields
    assert "branch_status" in fields

    target = tmp_path / "stochastic"
    cam_bundle({"candidates": product, "stochastic_validity": annotation}).save_dataset(
        target, layout="single", shard_target_bytes=4096
    )
    loaded = load_bundle(target).require("stochastic_validity")
    np.testing.assert_allclose(
        np.asarray(loaded.handle("projected_noise_intensity").materialize()), [[2.0]]
    )


def test_stochastic_validity_accepts_empty_candidate_table(monkeypatch):
    class Adapter:
        moment_layout = "normal"

        @staticmethod
        def closure_provenance():
            return {
                "representation": "wigner",
                "fpe_is_exact": False,
                "moment_closure": "factorized_bilinear",
                "moment_closure_is_exact": False,
                "deterministic_cam_is_exact": False,
            }

    class Model:
        params = {"mu": 0.0}

    monkeypatch.setattr(
        "qphase_cam.postprocessor.stochastic_validity.QPSGenDynamicsAdapter.from_model",
        lambda model: Adapter(),
    )
    product = _stochastic_product(candidates=0)
    processor = StochasticValidity(probe_epsilon=1.0)
    annotation = processor.process(
        product,
        CAMAnnotationContext(
            model=Model(), backend=NumpyBackend(), products={"candidates": product}
        ),
    )
    assert annotation.axis("annotation").size == 0
    assert annotation.axis("branch").size == 0
    from qphase_cam.postprocessor import stochastic_validity as validity

    expected = validity.StochasticValidity._base_row(
        0, branch_index=0, epsilon_side=0, closure={}, status=""
    )
    fields = {item["name"]: item for item in annotation.attributes["field_variables"]}
    for name in expected:
        resolved = "branch_status" if name == "status" else name
        assert resolved in fields, name
    assert fields["critical_eigenvalue_real"]["dtype"].startswith("<f")
    assert fields["normal_form_confining"]["dtype"].startswith("|b")
    assert fields["branch_status"]["dtype"].startswith("<U")
    assert processor.result_metadata["status"] == "no_supported_branches"
    assert processor.result_metadata["row_count"] == 0


def _fold_condensed_reduction() -> CondensedScalarReduction:
    model = TwoModeFoldModel()
    adapter = QPSGenDynamicsAdapter.from_model(model)
    plan = adapter.linear_reduction(
        candidate=adapter.search_linear_reductions(retained_dimension=1).candidates[0]
    )
    return CondensedScalarReduction(
        plan,
        order=2,
        control_names=("gamma",),
        base_params=model.params,
    )


def test_bifurcation_config_accepts_audit_section():
    base = {
        "controls": {"gamma": ControlRange(min=-1.0, max=0.0)},
        "perturbation": PerturbationConfig(parameter="gamma"),
    }
    config = BifurcationSolverConfig(**base)
    assert config.audit.near_miss_per_reason == 8
    assert config.audit.near_miss_total == 64
    custom = BifurcationSolverConfig(
        **base,
        audit={"near_miss_per_reason": 2, "near_miss_total": 5},
    )
    assert custom.audit.near_miss_per_reason == 2
    assert custom.audit.near_miss_total == 5
    with pytest.raises(ValueError):
        BifurcationSolverConfig(**base, audit={"unknown": 1})


def test_control_range_supports_logarithmic_seed_sampling():
    control = ControlRange(min=1e-6, max=1.0, sampling="log")
    np.testing.assert_allclose(control.sample_values(3), [1e-6, 1e-3, 1.0])
    assert ControlRange(min=-1.0, max=1.0).sampling == "linear"
    with pytest.raises(ValueError, match="must be positive"):
        ControlRange(min=0.0, max=1.0, sampling="log")


def test_control_range_supports_log_absolute_seed_sampling():
    positive = ControlRange(min=1e-6, max=1.0, sampling="log_abs")
    negative = ControlRange(min=-1.0, max=-1e-6, sampling="log_abs")

    np.testing.assert_allclose(positive.sample_values(3), [1e-6, 1e-3, 1.0])
    np.testing.assert_allclose(negative.sample_values(3), [-1.0, -1e-3, -1e-6])
    with pytest.raises(ValueError, match="same sign"):
        ControlRange(min=-1.0, max=1.0, sampling="log_abs")


def test_condensed_initial_starts_accept_explicit_control_axis():
    reduction = _fold_condensed_reduction()
    stats = SeedGenerationStats(source=reduction.seed_source)
    axis = np.asarray([-1.0, -0.1, 0.0])
    starts = reduction.initial_starts(
        ((-1.0, 0.0),),
        samples_per_control=99,
        max_starts=1000,
        order_parameter_bounds=(-2.0, 2.0),
        order_parameter_samples=21,
        stats=stats,
        control_axes=(axis,),
    )
    q_axis = reduction._q_axis((-2.0, 2.0), 21)
    assert len(starts) + stats.skipped_total == (len(q_axis) - 1) * len(axis)


def test_condensed_initial_starts_seed_stats_balance():
    reduction = _fold_condensed_reduction()
    stats = SeedGenerationStats(source=reduction.seed_source)
    starts = reduction.initial_starts(
        ((-1.0, 0.0),),
        samples_per_control=3,
        max_starts=1000,
        order_parameter_bounds=(-2.0, 2.0),
        order_parameter_samples=21,
        stats=stats,
    )
    assert stats.source == "brentq_scan"
    assert not stats.truncated
    q_axis = reduction._q_axis((-2.0, 2.0), 21)
    brackets = (len(q_axis) - 1) * 3
    assert len(starts) + stats.skipped_total == brackets


def test_condensed_initial_starts_marks_max_starts_truncation():
    reduction = _fold_condensed_reduction()
    stats = SeedGenerationStats(source=reduction.seed_source)
    starts = reduction.initial_starts(
        ((-1.0, 0.0),),
        samples_per_control=3,
        max_starts=1,
        order_parameter_bounds=(-2.0, 2.0),
        order_parameter_samples=41,
        stats=stats,
    )
    assert len(starts) == 1
    assert stats.truncated


def test_reduced_start_prefilter_reason_split():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})

    class SingularReduction:
        @staticmethod
        def reconstruct(start):
            del start
            raise ZeroDivisionError

    class HealthyReduction:
        @staticmethod
        def reconstruct(start):
            return np.asarray([start[0]])

    start = np.asarray([0.5, -0.5])
    physical_adapter = SimpleNamespace(
        model=SimpleNamespace(params={}),
        physical_eigenvalues=lambda vector, params: np.asarray([0.5]),
    )
    negative_adapter = SimpleNamespace(
        model=SimpleNamespace(params={}),
        physical_eigenvalues=lambda vector, params: np.asarray([-1.0]),
    )
    singular = SingularReduction()
    healthy = HealthyReduction()
    assert (
        solver._reduced_start_prefilter_reason(singular, physical_adapter, start)
        == "reconstruction_error"
    )
    assert (
        solver._reduced_start_prefilter_reason(healthy, physical_adapter, start) is None
    )
    assert (
        solver._reduced_start_prefilter_reason(healthy, negative_adapter, start)
        == "psd_violation"
    )
    assert not solver._reduced_start_is_physical(singular, physical_adapter, start)
    assert solver._reduced_start_is_physical(healthy, physical_adapter, start)


def test_near_miss_store_enforces_limits_and_prefers_small_residuals():
    store = NearMissStore(per_reason=2, total=64)
    for index in range(5):
        store.add(
            path="full",
            reduction=None,
            seed_source="domain_sampling",
            controls={"g": float(index)},
            rejection_reasons=("full_residual",),
            search_residual=1.0,
            full_residual=10.0 - index,
            min_state_eigenvalue=0.0,
        )
    records = store.finalize()
    assert len(records) == 2
    assert store.dropped == 3
    assert [record["controls"]["g"] for record in records] == [4.0, 3.0]

    store = NearMissStore(per_reason=8, total=3)
    for index in range(10):
        store.add(
            path="reduced",
            reduction="r",
            seed_source="reduction_roots",
            controls={},
            rejection_reasons=(f"reason_{index}",),
            search_residual=0.0,
            full_residual=float(index),
            min_state_eigenvalue=0.0,
        )
    assert len(store.finalize()) == 3
    assert store.dropped == 7

    store = NearMissStore(per_reason=1, total=64)
    for reasons, residual in (
        (("a",), 0.1),
        (("a", "b"), 0.2),
        (("a", "b"), 0.3),
    ):
        store.add(
            path="reduced",
            reduction="r",
            seed_source="reduction_roots",
            controls={},
            rejection_reasons=reasons,
            search_residual=0.0,
            full_residual=residual,
            min_state_eigenvalue=0.0,
        )
    records = store.finalize()
    # strict per-label cap: bucket "a" is saturated by the 0.1 record, so both
    # multi-label records carrying "a" are rejected as a whole and neither
    # "a" nor "b" grows beyond the per-reason limit.
    assert [record["full_residual"] for record in records] == [0.1]
    assert store.dropped == 2
    reason_counts: dict[str, int] = {}
    for record in records:
        for reason in record["rejection_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    assert max(reason_counts.values()) <= 1


def test_near_miss_store_stays_bounded_under_mass_rejection():
    per_reason, total, n_reasons, count = 8, 64, 12, 100_000
    store = NearMissStore(per_reason=per_reason, total=total)
    start = time.perf_counter()
    for index in range(count):
        # descending residual: every record is better than all previous ones,
        # forcing continuous top-k eviction under strict bucket caps
        store.add(
            path="reduced",
            reduction="r",
            seed_source="reduction_roots",
            controls={"g": float(index)},
            rejection_reasons=(f"reason_{index % n_reasons}", "shared"),
            search_residual=0.0,
            full_residual=float(count - index),
            min_state_eigenvalue=0.0,
        )
    elapsed = time.perf_counter() - start
    records = store.finalize()
    assert store._added == count
    assert len(records) == len(store._pool) <= total
    assert all(len(bucket) <= per_reason for bucket in store._buckets.values())
    stored_references = len(store._pool) + sum(
        len(bucket) for bucket in store._buckets.values()
    )
    assert stored_references <= total + (n_reasons + 1) * per_reason
    assert store.dropped == count - len(records)
    # best records win: the kept pool holds the smallest residuals injected
    assert max(record["full_residual"] for record in records) <= float(total)
    assert elapsed < 30.0


def test_near_miss_json_serializes_records():
    store = NearMissStore(per_reason=8, total=64)
    store.add(
        path="full",
        reduction=None,
        seed_source="upstream",
        controls={"g": 1.0},
        rejection_reasons=("non_physical",),
        search_residual=1e-8,
        full_residual=1e-6,
        min_state_eigenvalue=np.nan,
    )
    payloads = near_miss_json(store.finalize())
    assert len(payloads) == 1
    record = json.loads(payloads[0])
    assert record == {
        "path": "full",
        "reduction": None,
        "seed_source": "upstream",
        "controls": {"g": 1.0},
        "rejection_reasons": ["non_physical"],
        "search_residual": 1e-8,
        "full_residual": 1e-6,
        "min_state_eigenvalue": None,
    }


def test_refine_records_multi_label_near_misses():
    class RejectingReduction:
        retained_id = "q"
        method = "reduced_fraction_free"
        degree = 2

        @staticmethod
        def equations(value):
            del value
            return np.zeros(2)

        @staticmethod
        def jacobian(value):
            del value
            return np.eye(2)

        @staticmethod
        def reconstruct(value):
            return np.asarray([value[0]])

        @staticmethod
        def diagnostics(value):
            del value
            return ReductionDiagnostics(
                regularity_determinant=0.0,
                denominator_margin=1.0,
                condition_number=1.0,
                reduced_coefficients=np.asarray([0.0, 0.0, 1.0]),
            )

    class RejectingAdapter:
        model = SimpleNamespace(params={})

        @staticmethod
        def rhs(vector, params):
            del vector, params
            return np.asarray([1.0])

        @staticmethod
        def jacobian(vector, params):
            del vector, params
            return np.asarray([[1.0]])

        @staticmethod
        def physical_eigenvalues(vector, params):
            del vector, params
            return np.asarray([0.5])

    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    store = NearMissStore(per_reason=8, total=64)
    accepted, rejected, rejected_count, dedup_skipped = solver._refine(
        RejectingReduction(),
        RejectingAdapter(),
        [np.asarray([0.5, -0.5])],
        reporter=None,
        near_misses=store,
        seed_source="reduction_roots",
    )
    assert accepted == []
    assert (rejected_count, dedup_skipped) == (1, 0)
    # multi-label: one rejected candidate carries two reason labels
    assert dict(rejected) == {"full_residual": 1, "singular_reduction": 1}
    assert sum(rejected.values()) > rejected_count
    records = store.finalize()
    assert len(records) == 1
    record = records[0]
    assert record["path"] == "reduced"
    assert record["reduction"] == "q"
    assert record["seed_source"] == "reduction_roots"
    assert record["controls"] == {"gamma": pytest.approx(-0.5)}
    assert record["rejection_reasons"] == ("full_residual", "singular_reduction")
    assert record["full_residual"] == pytest.approx(1.0)
    assert record["search_residual"] == pytest.approx(0.0)
    assert record["min_state_eigenvalue"] == pytest.approx(0.5)


def test_reduced_audit_balances_seed_prefilter_and_refinement_counts():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    solver.strategy = ReducedStrategy(ReductionStrategyConfig())
    output = solver.search(TwoModeFoldModel(), NumpyBackend())
    audit = output.metadata["audit"]
    assert audit["schema"] == "cam_bifurcation_audit/2"
    assert audit["rejection_counter_semantics"] == REJECTION_COUNTER_SEMANTICS
    assert audit["result_note"] == "candidates_found"
    assert audit["physical_status"] == "checked"
    assert "generation_trial_count" in audit["field_units"]
    assert any("refinement_start_count" in formula for formula in audit["conservation"])
    for entry in audit["reductions"]:
        assert entry["seed_source"] in {"reduction_roots", "brentq_scan"}
        assert entry["physical_status"] == "checked"
        assert entry["physical_check_stage"] == "prefilter"
        # generation trials = generated candidates + evaluated generation skips
        assert entry["generation_trial_count"] == entry[
            "generated_candidate_count"
        ] + sum(entry["seed_skips"].values())
        # generated candidates split into prefilter pass/reject (same unit)
        assert entry["generated_candidate_count"] == (
            entry["prefilter_pass_count"] + entry["prefilter_rejected_count"]
        )
        assert entry["prefilter_rejected_count"] == sum(
            entry["prefilter_rejected"].values()
        )
        assert set(entry["prefilter_rejected"]) <= {
            "reconstruction_error",
            "psd_violation",
        }
        # refinement starts exhaust into accepted/rejected/duplicate
        assert entry["refinement_start_count"] == entry["prefilter_pass_count"]
        assert entry["refinement_start_count"] == (
            entry["accepted_count"]
            + entry["rejected_count"]
            + entry["refinement_duplicate_count"]
        )
        # path-specific workload units live in their own subfields
        assert entry["workload"]["control_point_count"] > 0
        if entry["seed_source"] == "reduction_roots":
            assert "polynomial_root_count" in entry["workload"]
            assert "brent_interval_count" not in entry["workload"]
        else:
            assert "brent_interval_count" in entry["workload"]
            assert "polynomial_root_count" not in entry["workload"]
    totals = audit["totals"]
    for key in (
        "generated_candidate_count",
        "prefilter_pass_count",
        "prefilter_rejected_count",
        "refinement_start_count",
        "refinement_duplicate_count",
        "accepted_count",
        "rejected_count",
    ):
        assert totals[key] == sum(entry[key] for entry in audit["reductions"])
    # mixed-unit legacy fields must not come back
    assert "raw_start_count" not in totals
    assert "physical_start_count" not in totals
    assert "generation_trial_count" not in totals
    # workload totals merge per unit key only
    for unit, count in totals["workload"].items():
        assert count == sum(
            entry["workload"].get(unit, 0) for entry in audit["reductions"]
        )
    assert totals["near_miss_saved"] == len(audit["near_misses"])
    assert (
        totals["near_miss_saved"] + totals["near_miss_dropped"]
        == totals["rejected_count"]
    )
    assert sum(totals["rejected_by_reason"].values()) >= totals["rejected_count"]
    assert isinstance(output.metadata["near_misses_json"], tuple)


def test_full_path_audit_records_seed_provenance_and_balance():
    expected_gamma = -6.0 + np.sqrt(28.0)
    expected_q = (expected_gamma + 4.0) / 4.0
    upstream = CAMResult(
        states=np.asarray([[[expected_q, 0.0], [0.0, 0.5]]]),
        residuals=np.asarray([0.0]),
        success=np.asarray([True]),
        valid_mask=np.asarray([True]),
        solution_count=np.asarray(1),
        params={"gamma": expected_gamma, "Gamma": 1.0, "kappa": 1.0},
    )
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
    output = solver.search(TwoModeFoldModel(), NumpyBackend(), data=upstream)
    audit = output.metadata["audit"]
    assert audit["path"] == "full"
    assert audit["seed_source"] == "upstream"
    assert audit["result_note"] == "candidates_found"
    assert audit["physical_status"] == "checked"
    assert audit["physical_check_stage"] == "generation"
    assert audit["generation_trial_count"] == audit["generated_candidate_count"] + sum(
        audit["seed_skips"].values()
    )
    assert audit["generated_candidate_count"] == audit["refinement_start_count"]
    assert audit["workload"]["upstream_seed_count"] >= 1
    totals = audit["totals"]
    assert totals["accepted_count"] == 1
    assert audit["refinement_start_count"] == (
        totals["accepted_count"]
        + totals["rejected_count"]
        + totals["refinement_duplicate_count"]
    )
    assert (
        totals["near_miss_saved"] + totals["near_miss_dropped"]
        == totals["rejected_count"]
    )


def test_full_path_checks_each_upstream_seed_physicality():
    expected_gamma = -6.0 + np.sqrt(28.0)
    expected_q = (expected_gamma + 4.0) / 4.0
    upstream = CAMResult(
        states=np.asarray(
            [
                [[expected_q, 0.0], [0.0, 0.5]],
                [[-1.0, 0.0], [0.0, -0.5]],
            ]
        ),
        residuals=np.asarray([0.0, 0.0]),
        success=np.asarray([True, True]),
        valid_mask=np.asarray([True, True]),
        solution_count=np.asarray(2),
        params={"gamma": expected_gamma, "Gamma": 1.0, "kappa": 1.0},
    )
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
    output = solver.search(TwoModeFoldModel(), NumpyBackend(), data=upstream)
    audit = output.metadata["audit"]
    assert audit["seed_source"] == "upstream"
    # the non-PSD upstream state is filtered at generation, never refined
    assert audit["workload"]["upstream_seed_count"] == 2
    assert audit["seed_skips"].get("non_physical_state") == 1
    assert audit["generated_candidate_count"] == 1
    assert audit["refinement_start_count"] == 1
    assert audit["totals"]["accepted_count"] == 1
    assert len(output.candidates) == 1


def test_empty_result_reports_standard_note_in_metadata_and_cases_csv(tmp_path):
    solver = _solver(2, {"gamma": ControlRange(min=-0.1, max=0.0)})
    solver.strategy = ReducedStrategy(ReductionStrategyConfig())
    model = TwoModeFoldModel()
    output = solver.search(model, NumpyBackend())
    assert not output.candidates
    audit = output.metadata["audit"]
    assert audit["result_note"] == EMPTY_RESULT_NOTE
    assert "not_exist" not in audit["result_note"]
    # a zero-candidate case still has a complete, conserved account
    totals = audit["totals"]
    assert totals["accepted_count"] == 0
    assert totals["generated_candidate_count"] > 0
    assert totals["generated_candidate_count"] == (
        totals["prefilter_pass_count"] + totals["prefilter_rejected_count"]
    )
    assert totals["refinement_start_count"] == (
        totals["accepted_count"]
        + totals["rejected_count"]
        + totals["refinement_duplicate_count"]
    )
    assert (
        totals["near_miss_saved"] + totals["near_miss_dropped"]
        == totals["rejected_count"]
    )
    # The zero-candidate case keeps a complete, typed account: the audit fields
    # the old cases CSV exposed are the case metadata and the audit payload.
    product = candidate_product_from_output(model, output)
    scan = build_candidate_product(
        states=np.empty((0, 2, 2), dtype=complex),
        state_vectors=np.empty((0, 4)),
        control_values=np.empty((0, 1)),
        control_names=("gamma",),
        full_residual_norm=np.empty(0),
        search_residual_norm=np.empty(0),
        success=np.empty(0, dtype=bool),
        status=[],
        method=[],
        verification_digits=np.empty(0, dtype=np.int64),
        verification_status=[],
        candidate_identity=[],
        case_axes={"gamma_b": np.asarray([1.0])},
        case_params={"gamma_b": np.asarray([1.0])},
        case_shape=(1,),
        case_metadata=[
            {
                **output.metadata,
                "case_status": "complete",
            }
        ],
    )
    assert product.axis("candidate").size == 0
    metadata = json.loads(
        str(np.asarray(scan.handle("case_metadata").materialize())[0])
    )
    for field in ("singular_coverage", "structural_coverage", "numerical_coverage"):
        assert field in metadata, field
    assert metadata["audit"]["physical_status"] == "checked"
    assert metadata["audit"]["result_note"] == EMPTY_RESULT_NOTE
    for removed in ("raw_start_count", "physical_start_count", "coverage_note"):
        assert removed not in metadata
    assert metadata["audit"]["totals"]["accepted_count"] == 0
    assert (
        int(metadata["audit"]["totals"]["generated_candidate_count"])
        == totals["generated_candidate_count"]
    )
    assert (
        int(metadata["audit"]["totals"]["refinement_start_count"])
        == totals["refinement_start_count"]
    )
    payloads = metadata["near_misses_json"]
    assert isinstance(payloads, list)
    assert len(payloads) == metadata["audit"]["totals"]["near_miss_saved"]
    for payload in payloads:
        record = json.loads(payload)
        assert record["rejection_reasons"]


def test_cases_csv_exposes_truncation_consumer_and_materialization_failures():
    """The reduction-contract failures stay visible in the typed case metadata."""
    solver = _solver(2, {"gamma": ControlRange(min=-0.1, max=0.0)})
    solver.strategy = ReducedStrategy(ReductionStrategyConfig())
    model = TwoModeFoldModel()
    output = solver.search(model, NumpyBackend())
    # inject the qpsgen reduction-contract failure fields (new contract:
    # materialization failures are reported with chart_id/error, outside
    # rejected_reason_counts)
    search = dict(output.metadata["reduction_search"])
    search["truncation_reasons"] = ["partition_limit", "materialization_limit"]
    search["consumer_error_count"] = 2
    search["consumer_errors"] = ("('r_diag_0',): boom",)
    search["materialization_failure_count"] = 2
    search["materialization_failures"] = (
        {"chart_id": "ret:r_diag_0|eq:0", "error": "not polynomial"},
        {"chart_id": "ret:r_diag_1|eq:1", "error": "bad denominator"},
    )
    metadata = {**output.metadata, "reduction_search": search}
    product = build_candidate_product(
        states=np.empty((0, 2, 2), dtype=complex),
        state_vectors=np.empty((0, 4)),
        control_values=np.empty((0, 1)),
        control_names=("gamma",),
        full_residual_norm=np.empty(0),
        search_residual_norm=np.empty(0),
        success=np.empty(0, dtype=bool),
        status=[],
        method=[],
        verification_digits=np.empty(0, dtype=np.int64),
        verification_status=[],
        candidate_identity=[],
        case_axes={"gamma_b": np.asarray([1.0])},
        case_params={"gamma_b": np.asarray([1.0])},
        case_shape=(1,),
        case_metadata=[{**metadata, "case_status": "complete"}],
    )
    row = json.loads(str(np.asarray(product.handle("case_metadata").materialize())[0]))
    search = row["reduction_search"]
    assert "partition_limit" in search["truncation_reasons"]
    assert "materialization_limit" in search["truncation_reasons"]
    assert search["consumer_error_count"] == 2
    assert "boom" in search["consumer_errors"][0]
    assert search["materialization_failure_count"] == 2
    assert search["materialization_failures"][0] == {
        "chart_id": "ret:r_diag_0|eq:0",
        "error": "not polynomial",
    }
    assert search["materialization_failures"][1]["error"] == "bad denominator"
