"""SciPy root and Cholesky CAM solver plugin."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from qphase_cam.state import CAMSolverOutput

from .base import CAMSolver, CAMSolverConfig
from .common import solve_single_state


class SteadyStateSolverConfig(CAMSolverConfig):
    method: Literal["auto", "root", "cholesky"] = "auto"
    root_method: str = "hybr"
    tolerance: float = Field(1e-10, gt=0.0)
    max_iterations: int = Field(1000, ge=1)
    use_jacobian: bool = True
    initial_guess: Any | None = None


class SteadyStateSolver(CAMSolver):
    name: ClassVar[str] = "steady_state"
    description: ClassVar[str] = "SciPy CAM steady-state solver"
    config_schema: ClassVar[type[SteadyStateSolverConfig]] = SteadyStateSolverConfig

    def solve(self, model: Any, backend: Any) -> CAMSolverOutput:
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("steady_state solver requires the numpy backend")
        solution = solve_single_state(
            model,
            model.params,
            self.config.initial_guess,
            method=self.config.method,
            root_method=self.config.root_method,
            tolerance=self.config.tolerance,
            max_iterations=self.config.max_iterations,
            use_jacobian=self.config.use_jacobian,
        )
        return CAMSolverOutput([solution])
