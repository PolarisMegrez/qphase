"""Bifurcation plugin contracts and tagged result tests."""

from __future__ import annotations

import importlib.util
from functools import lru_cache

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
from qphase_cam.bifurcation_result import CAMBifurcationResult
from qphase_cam.engine import Engine
from qphase_cam.result import CAMResult
from qphase_cam.solver.bifurcation import (
    BifurcationSolver,
    BifurcationSolverConfig,
    ControlRange,
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
        gamma, nonlinear_gain, kappa = sp.symbols(
            "gamma Gamma kappa", real=True
        )
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
    target = EquilibriumMultiplicity(
        EquilibriumMultiplicityConfig(order=order)
    )
    return BifurcationSolver(
        BifurcationSolverConfig(controls=controls),
        subplugins={
            "target": target,
            "strategy": AutoStrategy(ReductionStrategyConfig()),
            "discovery": SeedDiscovery(SeedDiscoveryConfig()),
        },
    )


def test_bifurcation_solver_validates_target_codimension():
    solver = _solver(2, {"control": ControlRange(min=-1.0, max=1.0)})
    assert solver.output_kind == "bifurcation_candidates"
    assert solver.target.order == 2
    with pytest.raises(ValueError, match="m-1 controls"):
        _solver(3, {"control": ControlRange(min=-1.0, max=1.0)})


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
    assert candidate.metadata["is_physical"]
    assert output.metadata["reduced_degree"] == 2


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
            controls={"gamma": ControlRange(min=-1.0, max=0.0)}
        ),
        subplugins={
            "target": EquilibriumMultiplicity(
                EquilibriumMultiplicityConfig(order=2)
            ),
            "strategy": FullStrategy(FullStrategyConfig()),
            "discovery": SeedDiscovery(SeedDiscoveryConfig()),
        },
    )
    output = solver.solve(
        TwoModeFoldModel(), NumpyBackend(), data=upstream
    )
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    np.testing.assert_allclose(candidate.controls["gamma"], expected_gamma)
    np.testing.assert_allclose(
        candidate.state_vector, [expected_q, 0.5, 0.0, 0.0]
    )
    assert candidate.method == "bordered_full"
    assert output.metadata["coverage"] == "upstream_bordered_local_search"


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
    assert result.result_kind == "bifurcation_candidates"
    np.testing.assert_allclose(result.control_values, [[0.5]])
    target = tmp_path / "bifurcation.npz"
    result.save(target)
    loaded = CAMBifurcationResult.load(target)
    np.testing.assert_allclose(loaded.state_vectors, result.state_vectors)
    assert target.with_suffix(".csv").exists()
