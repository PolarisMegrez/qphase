"""Backend-native batched Newton CAM solver plugin."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from qphase.backend.xputil import convert_to_numpy, get_xp

from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix
from qphase_cam.core.jacobian import JacobianResolver
from qphase_cam.core.liouvillian import residual_vector
from qphase_cam.errors import SolutionCapacityError
from qphase_cam.state import CAMSolution, CAMSolverOutput

from .base import CAMSolver, CAMSolverConfig
from .common import deduplicate_solutions


class BatchedNewtonSolverConfig(CAMSolverConfig):
    initial_guesses: Any | None = None
    max_iterations: int = Field(50, ge=1)
    tolerance: float = Field(1e-10, gt=0.0)
    line_search: bool = True
    line_search_steps: int = Field(5, ge=1)


class BatchedNewtonSolver(CAMSolver):
    name: ClassVar[str] = "batched_newton"
    description: ClassVar[str] = "Backend-native batched CAM Newton solver"
    config_schema: ClassVar[type[BatchedNewtonSolverConfig]] = BatchedNewtonSolverConfig
    supports_batch: ClassVar[bool] = True

    def solve(self, model: Any, backend: Any) -> CAMSolverOutput:
        xp = get_xp(backend.asarray([0.0]))
        batch_size = self._batch_size(model.params)
        guesses = self._guesses(model, backend, batch_size)
        batch_size, guess_count = int(guesses.shape[0]), int(guesses.shape[1])
        n_modes = int(model.n_modes)
        n_coordinates = n_modes * n_modes
        vector = matrix_to_vector(guesses).reshape(
            batch_size, guess_count, n_coordinates
        )
        params = self._expanded_params(model.params, backend, batch_size, guess_count)
        resolver = JacobianResolver()
        initial_states = vector_to_matrix(
            vector.reshape(batch_size * guess_count, n_coordinates), n_modes
        )
        resolver.resolve(model, initial_states, params, backend)
        converged = xp.zeros((batch_size, guess_count), dtype=bool)
        iterations = 0
        for iteration_index in range(self.config.max_iterations):
            iterations = iteration_index + 1
            flat = vector.reshape(batch_size * guess_count, n_coordinates)
            states = vector_to_matrix(flat, n_modes)
            residual = residual_vector(model, states, params).reshape(
                batch_size, guess_count, n_coordinates
            )
            norms = xp.linalg.norm(residual, axis=-1)
            converged = norms <= self.config.tolerance
            if bool(convert_to_numpy(xp.all(converged))):
                break
            jacobian = resolver.resolve(model, states, params, backend).reshape(
                batch_size, guess_count, n_coordinates, n_coordinates
            )
            try:
                step = xp.linalg.solve(jacobian, -residual[..., None])[..., 0]
            except Exception:
                step = xp.linalg.lstsq(jacobian, -residual[..., None], rcond=None)[0][
                    ..., 0
                ]
            step = xp.where(converged[..., None], 0.0, step)
            vector = self._update_with_line_search(
                vector, step, norms, converged, model, params, n_modes, xp
            )

        flat = vector.reshape(batch_size * guess_count, n_coordinates)
        states = vector_to_matrix(flat, n_modes)
        residual = residual_vector(model, states, params).reshape(
            batch_size, guess_count, n_coordinates
        )
        norms = xp.linalg.norm(residual, axis=-1)
        states_cpu = convert_to_numpy(states).reshape(
            batch_size, guess_count, n_modes, n_modes
        )
        norms_cpu = convert_to_numpy(norms)
        rows: list[list[CAMSolution]] = []
        for batch_index in range(batch_size):
            row = [
                CAMSolution(
                    states_cpu[batch_index, guess_index],
                    float(norms_cpu[batch_index, guess_index]),
                    bool(norms_cpu[batch_index, guess_index] <= self.config.tolerance),
                    "batched-newton",
                    iterations=iterations,
                )
                for guess_index in range(guess_count)
            ]
            row = deduplicate_solutions(row, self.config.tolerance * 100.0)
            row.sort(
                key=lambda item: model.cam_solution_sort_key(item.state, model.params)
            )
            if len(row) > int(model.steady_state_capacity):
                raise SolutionCapacityError(
                    f"model {model.name!r} solution capacity exceeded"
                )
            rows.append(row)
        return CAMSolverOutput(
            rows,
            metadata={
                "batch_size": batch_size,
                "iterations": iterations,
                "jacobian_source": resolver.last_source,
            },
        )

    def _update_with_line_search(
        self,
        vector: Any,
        step: Any,
        norms: Any,
        converged: Any,
        model: Any,
        params: dict[str, Any],
        n_modes: int,
        xp: Any,
    ) -> Any:
        if not self.config.line_search:
            return vector + step
        alpha = xp.ones(vector.shape[:-1], dtype=vector.dtype)
        shape = vector.shape
        updated = vector.copy()
        pending = ~converged
        for _ in range(self.config.line_search_steps):
            candidate = vector + alpha[..., None] * step
            states = vector_to_matrix(candidate.reshape(-1, shape[-1]), n_modes)
            candidate_norms = xp.linalg.norm(
                residual_vector(model, states, params).reshape(shape), axis=-1
            )
            accepted = pending & (candidate_norms < norms)
            updated = xp.where(accepted[..., None], candidate, updated)
            pending = pending & ~accepted
            if bool(convert_to_numpy(xp.all(~pending))):
                return updated
            alpha = xp.where(pending, alpha * 0.5, alpha)
        fallback = vector + alpha[..., None] * step
        return xp.where(pending[..., None], fallback, updated)

    def _batch_size(self, params: dict[str, Any]) -> int:
        sizes = {
            int(np.asarray(value).size)
            for value in params.values()
            if np.asarray(value).ndim > 0
        }
        if not sizes:
            return 1
        if len(sizes) != 1:
            raise ValueError(f"inconsistent batched parameter sizes: {sizes}")
        return sizes.pop()

    def _guesses(self, model: Any, backend: Any, batch_size: int) -> Any:
        n_modes = int(model.n_modes)
        if self.config.initial_guesses is None:
            value = np.broadcast_to(
                np.eye(n_modes, dtype=complex), (batch_size, 1, n_modes, n_modes)
            ).copy()
            return backend.asarray(value)
        value = np.asarray(self.config.initial_guesses, dtype=complex)
        if value.shape == (n_modes, n_modes):
            value = np.broadcast_to(value, (batch_size, 1, n_modes, n_modes)).copy()
        elif value.ndim == 3 and value.shape[-2:] == (n_modes, n_modes):
            value = np.broadcast_to(
                value[None, ...], (batch_size,) + value.shape
            ).copy()
        elif value.ndim != 4 or value.shape[0] != batch_size:
            raise ValueError(
                "initial_guesses must have shape (n,n), (g,n,n), or (b,g,n,n)"
            )
        return backend.asarray(value)

    def _expanded_params(
        self, params: dict[str, Any], backend: Any, batch_size: int, guesses: int
    ) -> dict[str, Any]:
        expanded = {}
        for name, value in params.items():
            array = backend.asarray(value)
            if getattr(array, "ndim", 0) == 0:
                array = backend.asarray(np.full(batch_size, float(value)))
            expanded[name] = backend.repeat(array, guesses)
        return expanded
