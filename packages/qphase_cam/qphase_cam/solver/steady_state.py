"""SciPy root and Cholesky CAM solver plugin."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from qphase_cam.state import CAMSolverOutput

from .base import (
    SOLVER_OUTPUT_DECLARATIONS,
    CAMProductFragment,
    CAMSolver,
    CAMSolverConfig,
    CAMSolveRequest,
)
from .common import solve_single_state
from .execution import state_fragment


class SteadyStateSolverConfig(CAMSolverConfig):
    method: Literal["auto", "root", "cholesky"] = "auto"
    root_method: str = "hybr"
    tolerance: float = Field(1e-10, gt=0.0)
    max_iterations: int = Field(1000, ge=1)
    use_jacobian: bool = True
    initial_guess: Any | None = None


class SteadyStateSolver(CAMSolver[SteadyStateSolverConfig]):
    name: ClassVar[str] = "steady_state"
    description: ClassVar[str] = "SciPy CAM steady-state solver"
    config_schema: ClassVar[type[SteadyStateSolverConfig]] = SteadyStateSolverConfig
    supported_tasks: ClassVar[tuple[str, ...]] = ("solve", "scan")
    output_products: ClassVar[tuple[Any, ...]] = (
        SOLVER_OUTPUT_DECLARATIONS["state"],
        SOLVER_OUTPUT_DECLARATIONS["state_scan"],
    )
    progress_unit: ClassVar[str] = "point"

    def solve(self, request: CAMSolveRequest) -> CAMProductFragment:
        return state_fragment(
            request=request,
            run=lambda model, point: self.solve_point(model, request.backend),
            supports_batch=False,
        )

    def solve_point(
        self, model: Any, backend: Any, point: Any | None = None
    ) -> CAMSolverOutput:
        """Solve the model once at its current parameters.

        The numeric core of :meth:`solve`; it returns the raw solver output.
        """
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
