"""Bifurcation plugin contracts and tagged result tests."""

from __future__ import annotations

import csv
import importlib.util
import json
import time
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
from fpgen import (
    LindbladChannel,
    MasterEquation,
    boson_modes,
    derive_kramers_moyal,
)
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.protocols import ResultProtocol
from qphase_cam.bifurcation_result import (
    CAMBifurcationBranchTable,
    CAMBifurcationResult,
    CAMBifurcationScanResult,
)
from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.core.reduction import (
    CondensedScalarReduction,
    ReductionDiagnostics,
    SeedGenerationStats,
)
from qphase_cam.engine import Engine
from qphase_cam.postprocessor.local_response import LocalResponseValidation
from qphase_cam.postprocessor.stochastic_validity import StochasticValidity
from qphase_cam.result import CAMResult
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
from qphase_cam.state import CAMBifurcationCandidate, CAMBifurcationOutput

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
    def cam_fpgen_dynamics(cls):
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
        solver.solve(model, NumpyBackend())


def test_old_bifurcation_postprocessor_is_removed():
    assert importlib.util.find_spec("qphase_cam.postprocessor.bifurcation") is None


def test_fraction_free_solver_finds_known_physical_double_root():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    output = solver.solve(TwoModeFoldModel(), NumpyBackend())
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


def test_reduction_strategy_exposes_fpgen_work_budgets():
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
    output = solver.solve(TwoModeFoldModel(), NumpyBackend())
    assert (
        output.metadata["coverage"]
        == "regular_reduction_budget_limited_and_bordered_local_search"
    )
    assert not output.metadata["reduction_search"]["complete"]


def test_reduction_structure_is_cached_between_fixed_parameter_cases():
    solver = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)})
    adapter = FPGenDynamicsAdapter.from_model(TwoModeFoldModel())
    _, first = solver._select_reductions(adapter)
    _, second = solver._select_reductions(adapter)
    assert first["compile_cache_hit"] is False
    assert second["compile_cache_hit"] is True


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
    output = solver.solve(TwoModeFoldModel(), NumpyBackend(), data=upstream)
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
    adapter = FPGenDynamicsAdapter.from_model(model)
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
    no_jacobian_model, tmp_path
):
    upstream = object()

    class Solver:
        name = "bifurcation"
        output_kind = "bifurcation_candidates"

        def solve(self, model, backend, *, data=None):
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
                        metadata={"regularity_determinant": 2.0},
                    )
                ],
                target="equilibrium_multiplicity",
                order=2,
                metadata={"control_names": ("control",)},
            )

    result = Engine(
        plugins={
            "backend": NumpyBackend(),
            "model": no_jacobian_model,
            "cam_solver": Solver(),
        }
    ).run(data=upstream)
    assert isinstance(result, CAMBifurcationResult)
    assert isinstance(result, ResultProtocol)
    assert isinstance(result, DatasetResultProtocol)
    np.testing.assert_array_equal(result.axes["candidate"], [0])
    assert result.result_kind == "bifurcation_candidates"
    np.testing.assert_allclose(result.control_values, [[0.5]])
    np.testing.assert_array_equal(result.verification_digits, [0])
    np.testing.assert_array_equal(result.verification_status, ["not_run"])
    target = tmp_path / "bifurcation.npz"
    result.save(target)
    loaded = CAMBifurcationResult.load(target)
    np.testing.assert_allclose(loaded.state_vectors, result.state_vectors)
    np.testing.assert_array_equal(
        loaded.verification_status, result.verification_status
    )
    assert target.with_suffix(".csv").exists()


def test_engine_persists_scaling_branch_table(tmp_path):
    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).solve(
        model, NumpyBackend()
    )
    result = Engine._pack_bifurcation(output, model)
    assert result.branches is not None
    assert result.branches.size == 2
    assert result.to_candidate_table()[0]["signature_count"] == 1
    assert result.to_branch_table()[0]["exponent_denominator"] == 2
    target = tmp_path / "classified.npz"
    result.save(target)
    candidate_header = (
        target.with_suffix(".csv").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "r_diag_0" in candidate_header
    assert "maximum_jacobian_real_part" in candidate_header
    assert target.with_name("classified_branches.csv").exists()
    loaded = CAMBifurcationResult.load(target)
    assert loaded.branches is not None
    np.testing.assert_allclose(
        loaded.branches.leading_state_coefficient,
        result.branches.leading_state_coefficient,
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
                metadata={"scaling_signatures": (signature,)},
            ),
            CAMBifurcationCandidate(
                state_vector=np.asarray([2.0, 0.5, 0.0, 0.0]),
                controls={"gamma": -0.25},
                full_residual_norm=0.0,
                search_residual_norm=0.0,
                success=True,
                status="verified",
                method="test",
                metadata={"scaling_signatures": ()},
            ),
        ],
        target="equilibrium_multiplicity",
        order=3,
        metadata={"control_names": ("gamma",)},
    )
    result = Engine._pack_bifurcation(output, model)
    assert result.diagnostics["scaling_signatures"].shape == (2,)
    assert result.to_candidate_table()[0]["signature_count"] == 1
    assert result.to_candidate_table()[1]["signature_count"] == 0
    target = tmp_path / "ragged.npz"
    result.save(target)
    loaded = CAMBifurcationResult.load(target)
    assert loaded.to_candidate_table()[0]["signature_count"] == 1
    assert loaded.to_candidate_table()[1]["signature_count"] == 0


def test_local_response_validation_solves_and_persists_complete_branches(tmp_path):
    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).solve(
        model, NumpyBackend()
    )
    result = Engine._pack_bifurcation(output, model)
    processor = LocalResponseValidation(
        epsilon_min=1e-10,
        epsilon_max=1e-6,
        epsilon_points=4,
        fit_points=3,
        residual_tolerance=1e-25,
    )
    result.postprocess.update(processor.process(result, model, NumpyBackend()))
    response = result.postprocess["local_response_validation"]
    summary = result.postprocess["local_response_summary"]
    assert len(response["candidate_index"]) == 8
    assert np.all(response["converged"])
    assert np.all(response["continuous"])
    assert np.max(response["full_residual_norm"]) < 1e-25
    np.testing.assert_allclose(response["state_fit_exponent"], 0.5, atol=1e-3)
    assert summary["sample_count"].tolist() == [8]
    assert summary["all_converged"].tolist() == [True]
    assert summary["all_continuous"].tolist() == [True]

    target = tmp_path / "response.npz"
    result.save(target)
    loaded = CAMBifurcationResult.load(target)
    np.testing.assert_allclose(
        loaded.postprocess["local_response_validation"]["delta_state_norm"],
        response["delta_state_norm"],
    )
    assert target.with_name("response_responses.csv").exists()
    assert target.with_name("response_response_summary.csv").exists()


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
        "qphase_cam.postprocessor.stochastic_validity.FPGenDynamicsAdapter.from_model",
        lambda model: Adapter(),
    )
    branches = CAMBifurcationBranchTable(
        candidate_index=np.asarray([0]),
        local_branch_index=np.asarray([0]),
        signature_index=np.asarray([0]),
        state_order=np.asarray([3]),
        perturbation_order=np.asarray([1]),
        coupling_state_order=np.asarray([0]),
        exponent_numerator=np.asarray([1]),
        exponent_denominator=np.asarray([3]),
        epsilon_side=np.asarray([1]),
        amplitude=np.asarray([1.0]),
        real_branch=np.asarray([True]),
        sublinear=np.asarray([True]),
        leading_state_coefficient=np.asarray([[[1.0]]]),
    )
    result = CAMBifurcationResult(
        states=np.asarray([[[1.0 + 0.0j]]]),
        state_vectors=np.asarray([[1.0]]),
        control_values=np.empty((1, 0)),
        control_names=(),
        full_residual_norm=np.asarray([0.0]),
        search_residual_norm=np.asarray([0.0]),
        success=np.asarray([True]),
        status=np.asarray(["verified"]),
        method=np.asarray(["test"]),
        verification_digits=np.asarray([50]),
        verification_status=np.asarray(["verified"]),
        branches=branches,
        meta={
            "perturbation_parameter": "mu",
            "perturbation_scale": 1.0,
        },
    )
    processor = StochasticValidity(probe_epsilon=1.0)
    result.postprocess.update(processor.process(result, Model(), NumpyBackend()))
    diagnostic = result.postprocess["stochastic_validity"]
    assert diagnostic["status"].tolist() == ["complete"]
    np.testing.assert_allclose(diagnostic["projected_noise_intensity"], [2.0])
    np.testing.assert_allclose(diagnostic["normal_form_state_coefficient"], [-1.0])
    np.testing.assert_allclose(diagnostic["epsilon_crossover"], [2.0 * np.sqrt(2.0)])
    assert diagnostic["regime"].tolist() == ["noise_dominated"]
    # the empty-table schema mirrors the non-empty schema field-for-field
    empty = StochasticValidity._empty()
    assert tuple(diagnostic) == tuple(empty)
    for name, values in diagnostic.items():
        assert values.dtype.kind == empty[name].dtype.kind, name
    target = tmp_path / "stochastic.npz"
    result.save(target)
    assert target.with_name("stochastic_stochastic_validity.csv").exists()


def test_stochastic_validity_accepts_empty_candidate_table(monkeypatch, tmp_path):
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
        "qphase_cam.postprocessor.stochastic_validity.FPGenDynamicsAdapter.from_model",
        lambda model: Adapter(),
    )
    result = CAMBifurcationResult(
        states=np.empty((0, 1, 1), dtype=complex),
        state_vectors=np.empty((0, 1)),
        control_values=np.empty((0, 0)),
        control_names=(),
        full_residual_norm=np.empty(0),
        search_residual_norm=np.empty(0),
        success=np.empty(0, dtype=bool),
        status=np.empty(0, dtype=str),
        method=np.empty(0, dtype=str),
        verification_digits=np.empty(0, dtype=int),
        verification_status=np.empty(0, dtype=str),
        meta={
            "perturbation_parameter": "mu",
            "perturbation_scale": 1.0,
        },
    )
    processor = StochasticValidity(probe_epsilon=1.0)
    output = processor.process(result, Model(), NumpyBackend())
    diagnostic = output["stochastic_validity"]
    # the empty table carries the full stable schema, not just candidate_index
    expected = StochasticValidity._empty()
    assert tuple(diagnostic) == tuple(expected)
    assert set(diagnostic) == {
        "candidate_index",
        "branch_index",
        "epsilon_side",
        "status",
        "critical_eigenvalue_real",
        "critical_eigenvalue_imag",
        "noncritical_spectral_gap",
        "critical_mode_condition_number",
        "eigenvector_condition_number",
        "noise_covariance_minimum_eigenvalue",
        "projected_noise_intensity",
        "parameter_forcing",
        "branch_center_coefficient",
        "normal_form_state_coefficient",
        "normal_form_confining",
        "critical_fluctuation_scale",
        "epsilon_crossover",
        "probe_epsilon",
        "regime",
        "noise_semantics",
        "representation",
        "fpe_is_exact",
        "moment_closure",
        "moment_closure_is_exact",
        "deterministic_cam_is_exact",
    }
    for name, values in diagnostic.items():
        assert values.size == 0
        assert values.dtype == expected[name].dtype
    assert diagnostic["candidate_index"].dtype.kind == "i"
    assert diagnostic["status"].dtype.kind == "U"
    assert diagnostic["critical_eigenvalue_real"].dtype.kind == "f"
    assert diagnostic["normal_form_confining"].dtype.kind == "b"
    assert processor.result_metadata["status"] == "no_supported_branches"
    assert processor.result_metadata["row_count"] == 0
    # NPZ round trip preserves the full empty schema and dtypes
    result.postprocess.update(output)
    target = tmp_path / "empty_stochastic.npz"
    result.save(target)
    loaded = CAMBifurcationResult.load(target)
    loaded_table = loaded.postprocess["stochastic_validity"]
    assert tuple(loaded_table) == tuple(diagnostic)
    for name, values in loaded_table.items():
        assert values.size == 0
        assert values.dtype == diagnostic[name].dtype
    # CSV round trip: a header-only file with the full column schema
    csv_path = target.with_name("empty_stochastic_stochastic_validity.csv")
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert next(csv.reader(lines)) == list(expected)


def _fold_condensed_reduction() -> CondensedScalarReduction:
    model = TwoModeFoldModel()
    adapter = FPGenDynamicsAdapter.from_model(model)
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
    output = solver.solve(TwoModeFoldModel(), NumpyBackend())
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
    output = solver.solve(TwoModeFoldModel(), NumpyBackend(), data=upstream)
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
    output = solver.solve(TwoModeFoldModel(), NumpyBackend(), data=upstream)
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
    output = solver.solve(model, NumpyBackend())
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
    result = Engine._pack_bifurcation(output, model)
    result.meta.update(output.metadata)
    scan = CAMBifurcationScanResult(
        case_axes={"gamma_b": np.asarray([1.0])},
        case_shape=(1,),
        case_params={"gamma_b": np.asarray([1.0])},
        candidate_offsets=np.asarray([0, 0]),
        candidates=result,
        case_metadata=(dict(result.meta),),
    )
    target = tmp_path / "empty.npz"
    scan.save(target)
    rows = list(
        csv.DictReader(
            target.with_name("empty_cases.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    (row,) = rows
    for field in (
        "singular_coverage",
        "generated_candidate_count",
        "prefilter_pass_count",
        "prefilter_rejected_count",
        "refinement_start_count",
        "refinement_duplicate_count",
        "accepted_count",
        "rejected_count",
        "physical_status",
        "top_rejection_reasons",
        "near_miss_saved",
        "near_miss_dropped",
        "truncation_reasons",
        "consumer_error_count",
        "materialization_failure_count",
        "result_note",
    ):
        assert field in row
    for removed in ("raw_start_count", "physical_start_count", "coverage_note"):
        assert removed not in row
    assert row["result_note"] == EMPTY_RESULT_NOTE
    assert row["accepted_count"] == "0"
    assert row["physical_status"] == "checked"
    assert int(row["generated_candidate_count"]) == totals["generated_candidate_count"]
    assert int(row["refinement_start_count"]) == totals["refinement_start_count"]
    loaded = CAMBifurcationScanResult.load(target)
    loaded_audit = loaded.case_metadata[0]["audit"]
    assert loaded_audit["result_note"] == EMPTY_RESULT_NOTE
    payloads = loaded.case_metadata[0]["near_misses_json"]
    assert isinstance(payloads, tuple)
    assert len(payloads) == loaded_audit["totals"]["near_miss_saved"]
    for payload in payloads:
        record = json.loads(payload)
        assert record["rejection_reasons"]


def test_cases_csv_exposes_truncation_consumer_and_materialization_failures(
    tmp_path,
):
    solver = _solver(2, {"gamma": ControlRange(min=-0.1, max=0.0)})
    solver.strategy = ReducedStrategy(ReductionStrategyConfig())
    model = TwoModeFoldModel()
    output = solver.solve(model, NumpyBackend())
    result = Engine._pack_bifurcation(output, model)
    result.meta.update(output.metadata)
    # inject the fpgen reduction-contract failure fields (new contract:
    # materialization failures are reported with chart_id/error, outside
    # rejected_reason_counts)
    search = dict(result.meta["reduction_search"])
    search["truncation_reasons"] = ["partition_limit", "materialization_limit"]
    search["consumer_error_count"] = 2
    search["consumer_errors"] = ("('r_diag_0',): boom",)
    search["materialization_failure_count"] = 2
    search["materialization_failures"] = (
        {"chart_id": "ret:r_diag_0|eq:0", "error": "not polynomial"},
        {"chart_id": "ret:r_diag_1|eq:1", "error": "bad denominator"},
    )
    result.meta["reduction_search"] = search
    scan = CAMBifurcationScanResult(
        case_axes={"gamma_b": np.asarray([1.0])},
        case_shape=(1,),
        case_params={"gamma_b": np.asarray([1.0])},
        candidate_offsets=np.asarray([0, 0]),
        candidates=result,
        case_metadata=(dict(result.meta),),
    )
    target = tmp_path / "failures.npz"
    scan.save(target)
    (row,) = list(
        csv.DictReader(
            target.with_name("failures_cases.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    assert "partition_limit" in row["truncation_reasons"]
    assert "materialization_limit" in row["truncation_reasons"]
    assert row["consumer_error_count"] == "2"
    assert "boom" in row["consumer_errors"]
    assert row["materialization_failure_count"] == "2"
    assert "ret:r_diag_0|eq:0: not polynomial" in row["materialization_failures"]
    assert "ret:r_diag_1|eq:1: bad denominator" in row["materialization_failures"]
