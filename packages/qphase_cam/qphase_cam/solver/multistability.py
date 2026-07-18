"""Multi-start CAM solver plugin."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from qphase_cam.errors import SolutionCapacityError
from qphase_cam.state import CAMSolverOutput

from .base import CAMSolver, CAMSolverConfig
from .common import (
    deduplicate_solutions,
    random_hermitian_guesses,
    solve_single_state,
)
from .guess_bounds import GuessBounds, GuessBoundsConfig, infer_guess_bounds


class MultistabilitySolverConfig(CAMSolverConfig):
    n_guesses: int = Field(32, ge=1)
    guess_scale: float = Field(10.0, gt=0.0)
    seed: int | None = 42
    method: Literal["auto", "root", "cholesky"] = "auto"
    tolerance: float = Field(1e-10, gt=0.0)
    residual_tolerance: float = Field(1e-7, gt=0.0)
    distance_tolerance: float = Field(1e-5, gt=0.0)
    use_jacobian: bool = True
    n_workers: int = Field(1, ge=1)
    initial_guesses: Any | None = None
    guess_bounds: GuessBoundsConfig | Literal["auto"] | None = None
    tail_fraction: float = Field(0.25, ge=0.0, le=1.0)
    tail_orders: float = Field(3.0, gt=0.0)
    tile_workers: int = Field(1, ge=1)
    tile_size: int = Field(32, ge=1)


class MultistabilitySolver(CAMSolver):
    name: ClassVar[str] = "multistability"
    description: ClassVar[str] = "Multi-start CAM steady-state search"
    config_schema: ClassVar[type[MultistabilitySolverConfig]] = (
        MultistabilitySolverConfig
    )
    supports_batch: ClassVar[bool] = True

    def solve(self, model: Any, backend: Any) -> CAMSolverOutput:
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("multistability solver requires the numpy backend")
        batch_size = self._batch_size(model.params)
        points = [
            self._point_params(model.params, index, batch_size)
            for index in range(batch_size)
        ]
        if batch_size > 1 and self.config.tile_workers > 1:
            indexed = list(enumerate(points))
            tiles = [
                indexed[start : start + self.config.tile_size]
                for start in range(0, len(indexed), self.config.tile_size)
            ]
            options = self.config.model_dump()
            options["n_workers"] = 1
            payloads = [(model, tile, options) for tile in tiles]
            with ProcessPoolExecutor(max_workers=self.config.tile_workers) as pool:
                completed = list(pool.map(_solve_tile, payloads))
            indexed_rows = [item for tile in completed for item in tile]
            indexed_rows.sort(key=lambda item: item[0])
            rows = [row for _, row, _ in indexed_rows]
            attempted_count = sum(count for _, _, count in indexed_rows)
        else:
            solved = [
                _solve_point(model, params, self.config, index)
                for index, params in enumerate(points)
            ]
            rows = [row for row, _ in solved]
            attempted_count = sum(count for _, count in solved)
        metadata = {
            "attempted": attempted_count,
            "batch_size": batch_size,
            "tile_workers": self.config.tile_workers,
        }
        return CAMSolverOutput(rows if batch_size > 1 else rows[0], metadata=metadata)

    @staticmethod
    def _batch_size(params: dict[str, Any]) -> int:
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

    @staticmethod
    def _point_params(
        params: dict[str, Any], index: int, batch_size: int
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name, value in params.items():
            array = np.asarray(value)
            if array.ndim == 0:
                output[name] = value
            elif array.size == batch_size:
                output[name] = array.reshape(-1)[index].item()
            else:
                raise ValueError(
                    f"parameter {name!r} cannot broadcast to batch size {batch_size}"
                )
        return output


def _solve_tile(
    payload: tuple[Any, list[tuple[int, dict[str, Any]]], dict[str, Any]],
) -> list[tuple[int, list[Any], int]]:
    model, tile, options = payload
    config = MultistabilitySolverConfig.model_validate(options)
    return [
        (index, *_solve_point(model, params, config, index)) for index, params in tile
    ]


def _solve_point(
    model: Any,
    params: dict[str, Any],
    config: MultistabilitySolverConfig,
    point_index: int,
) -> tuple[list[Any], int]:
    guesses = _make_guesses(model, params, config, point_index)

    def solve_guess(guess: Any) -> Any:
        return solve_single_state(
            model,
            params,
            guess,
            method=config.method,
            tolerance=config.tolerance,
            use_jacobian=config.use_jacobian,
        )

    if config.n_workers == 1:
        attempted = [solve_guess(guess) for guess in guesses]
    else:
        with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
            attempted = list(pool.map(solve_guess, guesses))
    accepted = [
        solution
        for solution in attempted
        if solution.success and solution.residual <= config.residual_tolerance
    ]
    accepted = deduplicate_solutions(accepted, config.distance_tolerance)
    accepted.sort(key=lambda item: model.cam_solution_sort_key(item.state, params))
    capacity = int(model.steady_state_capacity)
    if len(accepted) > capacity:
        raise SolutionCapacityError(
            f"model {model.name!r} declares capacity {capacity} but "
            f"{len(accepted)} distinct states were found"
        )
    return accepted, len(attempted)


def _make_guesses(
    model: Any,
    params: dict[str, Any],
    config: MultistabilitySolverConfig,
    point_index: int,
) -> list[np.ndarray]:
    n_modes = int(model.n_modes)
    seed = None if config.seed is None else config.seed + point_index
    explicit: list[np.ndarray] = []
    if config.initial_guesses is not None:
        values = np.asarray(config.initial_guesses, dtype=complex)
        if values.shape == (n_modes, n_modes):
            values = values[None, ...]
        if values.ndim != 3 or values.shape[-2:] != (n_modes, n_modes):
            raise ValueError("initial_guesses must have shape (n,n) or (g,n,n)")
        explicit = [0.5 * (value + value.conj().T) for value in values]
    remaining = max(config.n_guesses - len(explicit), 0)
    if config.guess_bounds == "auto":
        bounds = infer_guess_bounds(model, params, seed=seed)
    elif isinstance(config.guess_bounds, GuessBoundsConfig):
        bounds = GuessBounds.from_config(config.guess_bounds, n_modes)
    else:
        bounds = None
    if bounds is None:
        generated = random_hermitian_guesses(
            n_modes, remaining, config.guess_scale, seed
        )
    else:
        generated = bounds.sample(
            remaining,
            seed,
            config.tail_fraction,
            config.tail_orders,
        )
    return (explicit + generated)[: config.n_guesses]
