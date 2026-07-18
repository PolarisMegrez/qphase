"""Deflated Newton multi-root CAM solver plugin."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix
from qphase_cam.core.jacobian import JacobianResolver
from qphase_cam.core.liouvillian import residual_vector
from qphase_cam.errors import SolutionCapacityError
from qphase_cam.state import CAMSolution, CAMSolverOutput

from .base import CAMSolver, CAMSolverConfig
from .common import deduplicate_solutions, random_hermitian_guesses


class DeflationSolverConfig(CAMSolverConfig):
    n_guesses: int = Field(32, ge=1)
    guess_scale: float = Field(10.0, gt=0.0)
    seed: int | None = 42
    alpha: float = Field(1e-2, gt=0.0)
    tolerance: float = Field(1e-10, gt=0.0)
    residual_tolerance: float = Field(1e-7, gt=0.0)
    distance_tolerance: float = Field(1e-5, gt=0.0)
    max_iterations: int = Field(100, ge=1)
    damping: float = Field(1.0, gt=0.0, le=1.0)


class DeflationSolver(CAMSolver):
    name: ClassVar[str] = "deflation"
    description: ClassVar[str] = "Deflated Newton CAM multi-root solver"
    config_schema: ClassVar[type[DeflationSolverConfig]] = DeflationSolverConfig

    def solve(self, model: Any, backend: Any) -> CAMSolverOutput:
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("deflation solver requires the numpy backend")
        resolver = JacobianResolver()
        guesses = random_hermitian_guesses(
            int(model.n_modes),
            self.config.n_guesses,
            self.config.guess_scale,
            self.config.seed,
        )
        roots: list[np.ndarray] = []
        solutions: list[CAMSolution] = []
        for guess in guesses:
            vector = np.asarray(matrix_to_vector(guess), dtype=float)
            iteration_count = 0
            for _ in range(self.config.max_iterations):
                iteration_count += 1
                state = vector_to_matrix(vector, int(model.n_modes))
                residual = np.asarray(residual_vector(model, state, model.params))
                norm = float(np.linalg.norm(residual))
                if norm <= self.config.tolerance:
                    break
                jacobian = np.asarray(
                    resolver.resolve(model, state, model.params, backend)
                )
                gradient = np.zeros_like(vector)
                for known in roots:
                    difference = vector - known
                    distance = float(difference @ difference)
                    if distance > 0.0:
                        gradient += (
                            -2.0
                            * self.config.alpha
                            * difference
                            / (distance * (distance + self.config.alpha))
                        )
                operator = jacobian + np.outer(residual, gradient)
                step, *_ = np.linalg.lstsq(operator, -residual, rcond=None)
                vector = vector + self.config.damping * step
            state = np.asarray(vector_to_matrix(vector, int(model.n_modes)))
            norm = float(np.linalg.norm(residual_vector(model, state, model.params)))
            if norm <= self.config.residual_tolerance:
                roots.append(vector.copy())
                solutions.append(
                    CAMSolution(
                        state,
                        norm,
                        True,
                        "deflated-newton",
                        iterations=iteration_count,
                    )
                )
        solutions = deduplicate_solutions(
            solutions, self.config.distance_tolerance
        )
        solutions.sort(
            key=lambda item: model.cam_solution_sort_key(item.state, model.params)
        )
        if len(solutions) > int(model.steady_state_capacity):
            raise SolutionCapacityError(
                f"model {model.name!r} solution capacity exceeded"
            )
        return CAMSolverOutput(solutions)
