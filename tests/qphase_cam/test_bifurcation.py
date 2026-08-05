"""Bifurcation plugin contracts and tagged result tests."""

from __future__ import annotations

import importlib.util
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
from qphase_cam.bifurcation_result import CAMBifurcationResult
from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.core.reduction import CondensedScalarReduction
from qphase_cam.engine import Engine
from qphase_cam.result import CAMResult
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
from qphase_cam.solver.bifurcation_discovery import (
    SeedDiscovery,
    SeedDiscoveryConfig,
)
from qphase_cam.solver.bifurcation_strategy import (
    AutoStrategy,
    FullStrategy,
    FullStrategyConfig,
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
    assert candidate.metadata["is_physical"]
    signature = candidate.metadata["scaling_signatures"][0]
    assert (
        signature["state_order"],
        signature["perturbation_order"],
        signature["coupling_state_order"],
    ) == (2, 1, 0)
    assert signature["exponent"] == pytest.approx(0.5)
    assert output.metadata["reduced_degree"] == 2
    assert (
        output.metadata["coverage"]
        == "regular_reduction_exhaustive_and_bordered_local_search"
    )
    search = output.metadata["reduction_search"]
    assert search["coverage"] == "exhaustive"
    assert search["regular_branches_only"]
    assert not search["returned_prefix"]


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
    plan = adapter.dynamics.linear_reduce(
        candidate=adapter.dynamics.search_linear_reductions(
            retained_dimension=1
        ).candidates[0]
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
