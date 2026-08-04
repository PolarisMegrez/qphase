"""Bifurcation plugin contracts and tagged result tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.bifurcation_result import CAMBifurcationResult
from qphase_cam.engine import Engine
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
    ReductionStrategyConfig,
)
from qphase_cam.solver.bifurcation_target import (
    EquilibriumMultiplicity,
    EquilibriumMultiplicityConfig,
)
from qphase_cam.state import CAMBifurcationCandidate, CAMBifurcationOutput


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
