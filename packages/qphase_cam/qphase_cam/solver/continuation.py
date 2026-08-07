"""Pseudo-arclength continuation CAM solver plugin."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix
from qphase_cam.core.jacobian import JacobianResolver
from qphase_cam.core.liouvillian import residual_vector
from qphase_cam.state import CAMSolution, CAMSolverOutput

from .base import CAMSolver, CAMSolverConfig
from .common import solve_single_state


class ContinuationSolverConfig(CAMSolverConfig):
    parameter: str
    start: float
    stop: float
    step: float = Field(0.002, gt=0.0)
    min_step: float = Field(1e-7, gt=0.0)
    max_step: float = Field(0.02, gt=0.0)
    max_steps: int = Field(2000, ge=1)
    corrector_iterations: int = Field(16, ge=1)
    tolerance: float = Field(1e-10, gt=0.0)
    parameter_epsilon: float = Field(1e-6, gt=0.0)
    initial_guess: Any | None = None


class ContinuationSolver(CAMSolver[ContinuationSolverConfig]):
    name: ClassVar[str] = "continuation"
    description: ClassVar[str] = "Pseudo-arclength CAM continuation"
    config_schema: ClassVar[type[ContinuationSolverConfig]] = ContinuationSolverConfig

    def solve(self, model: Any, backend: Any) -> CAMSolverOutput:
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("continuation solver requires the numpy backend")
        if self.config.parameter not in model.params:
            raise ValueError(
                f"unknown continuation parameter {self.config.parameter!r}"
            )
        params = dict(model.params)
        params[self.config.parameter] = self.config.start
        initial = solve_single_state(
            model,
            params,
            self.config.initial_guess,
            method="root",
            tolerance=self.config.tolerance,
            use_jacobian=True,
        )
        if not initial.success:
            raise RuntimeError("could not obtain the initial continuation state")
        resolver = JacobianResolver()
        x = np.asarray(matrix_to_vector(initial.state), dtype=float)
        lam = float(self.config.start)
        direction = 1.0 if self.config.stop >= self.config.start else -1.0
        tangent = self._tangent(model, params, x, lam, resolver, backend, direction)
        step = self.config.step
        solutions = [initial]
        values = [lam]
        for _ in range(self.config.max_steps):
            if direction * (lam - self.config.stop) >= 0.0:
                break
            predicted = np.concatenate([x, [lam]]) + step * tangent
            corrected, success, iterations = self._correct(
                model, params, predicted, tangent, resolver, backend
            )
            if not success:
                step *= 0.5
                if step < self.config.min_step:
                    break
                continue
            x, lam = corrected[:-1], float(corrected[-1])
            state = np.asarray(vector_to_matrix(x, int(model.n_modes)))
            current_params = self._params(params, lam)
            residual = float(
                np.linalg.norm(residual_vector(model, state, current_params))
            )
            solutions.append(
                CAMSolution(
                    state,
                    residual,
                    residual <= self.config.tolerance,
                    "pseudo-arclength",
                    iterations=iterations,
                )
            )
            values.append(lam)
            tangent = self._tangent(model, params, x, lam, resolver, backend, tangent)
            step = min(step * 1.1, self.config.max_step)
        return CAMSolverOutput(
            [[solution] for solution in solutions],
            axes={self.config.parameter: np.asarray(values)},
            metadata={"continuation": True},
        )

    def _params(self, params: dict[str, Any], value: float) -> dict[str, Any]:
        updated = dict(params)
        updated[self.config.parameter] = value
        return updated

    def _parameter_derivative(
        self, model: Any, params: dict[str, Any], x: np.ndarray, value: float
    ) -> np.ndarray:
        epsilon = self.config.parameter_epsilon
        state = vector_to_matrix(x, int(model.n_modes))
        plus = residual_vector(model, state, self._params(params, value + epsilon))
        minus = residual_vector(model, state, self._params(params, value - epsilon))
        return np.asarray((plus - minus) / (2.0 * epsilon))

    def _tangent(
        self,
        model: Any,
        params: dict[str, Any],
        x: np.ndarray,
        value: float,
        resolver: JacobianResolver,
        backend: Any,
        preference: Any,
    ) -> np.ndarray:
        state = vector_to_matrix(x, int(model.n_modes))
        jacobian = np.asarray(
            resolver.resolve(model, state, self._params(params, value), backend)
        )
        parameter_derivative = self._parameter_derivative(model, params, x, value)
        operator = np.hstack((jacobian, parameter_derivative[:, None]))
        tangent = np.linalg.svd(operator)[2][-1]
        tangent /= np.linalg.norm(tangent)
        preferred = (
            np.concatenate((np.zeros_like(x), np.asarray([preference], dtype=float)))
            if np.isscalar(preference)
            else np.asarray(preference)
        )
        if np.dot(tangent, preferred) < 0.0:
            tangent = -tangent
        return tangent

    def _correct(
        self,
        model: Any,
        params: dict[str, Any],
        predicted: np.ndarray,
        tangent: np.ndarray,
        resolver: JacobianResolver,
        backend: Any,
    ) -> tuple[np.ndarray, bool, int]:
        current = predicted.copy()
        n_coordinates = len(current) - 1
        for iteration in range(1, self.config.corrector_iterations + 1):
            x, value = current[:n_coordinates], float(current[-1])
            state = vector_to_matrix(x, int(model.n_modes))
            current_params = self._params(params, value)
            residual = np.asarray(residual_vector(model, state, current_params))
            if np.linalg.norm(residual) <= self.config.tolerance:
                return current, True, iteration
            jacobian = np.asarray(
                resolver.resolve(model, state, current_params, backend)
            )
            parameter_derivative = self._parameter_derivative(model, params, x, value)
            operator = np.zeros((n_coordinates + 1, n_coordinates + 1))
            operator[:-1, :-1] = jacobian
            operator[:-1, -1] = parameter_derivative
            operator[-1] = tangent
            arclength_residual = -float(np.dot(tangent, current - predicted))
            right = np.concatenate(
                (-residual, np.asarray([arclength_residual], dtype=float))
            )
            correction, *_ = np.linalg.lstsq(operator, right, rcond=None)
            current += correction
        return current, False, self.config.corrector_iterations
