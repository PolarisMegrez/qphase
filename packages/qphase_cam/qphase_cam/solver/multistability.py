"""Multi-start CAM solver plugin."""

from __future__ import annotations

import ctypes
import math
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from itertools import product
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.errors import SolutionCapacityError
from qphase_cam.state import CAMSolverOutput

from .base import CAMSolver, CAMSolverConfig
from .common import (
    deduplicate_solutions,
    prepare_root_system,
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
    capacity_patience: int = Field(10, ge=0)
    n_workers: int = Field(1, ge=1)
    initial_guesses: Any | None = None
    guess_bounds: GuessBoundsConfig | Literal["auto"] | None = None
    tail_fraction: float = Field(0.25, ge=0.0, le=1.0)
    tail_orders: float = Field(3.0, gt=0.0)
    tile_workers: int = Field(1, ge=1)
    n_tiles: int | None = Field(default=None, ge=1)
    tile_size: int | None = Field(default=None, ge=1)
    discover_seeds: bool = True
    bounds_inference_starts: int = Field(128, ge=1)
    seed_search_guesses: int = Field(200, ge=1)
    retry_guesses: int = Field(200, ge=1)
    refine_suspicious: bool = True
    refine_guesses: int = Field(50, ge=1)

    @model_validator(mode="after")
    def validate_tile_partition(self) -> MultistabilitySolverConfig:
        if self.n_tiles is not None and self.tile_size is not None:
            raise ValueError("configure only one of n_tiles or tile_size")
        return self


class MultistabilitySolver(CAMSolver[MultistabilitySolverConfig]):
    name: ClassVar[str] = "multistability"
    description: ClassVar[str] = "Multi-start CAM steady-state search"
    config_schema: ClassVar[type[MultistabilitySolverConfig]] = (
        MultistabilitySolverConfig
    )
    supports_batch: ClassVar[bool] = True

    def solve(
        self, model: Any, backend: Any, context: Any | None = None
    ) -> CAMSolverOutput:
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("multistability solver requires the numpy backend")
        batch_size = self._batch_size(model.params)
        points = [
            self._point_params(model.params, index, batch_size)
            for index in range(batch_size)
        ]
        indexed = list(enumerate(points))
        grid_shape = _context_grid_shape(context, batch_size)
        if batch_size > 1:
            global_bounds = _scan_guess_bounds(
                model,
                _representative_params(points, grid_shape),
                self.config,
            )
            global_seeds = self._discover_global_seeds(
                model,
                points,
                grid_shape,
                global_bounds,
                context,
            )
            global_bounds = _expand_bounds_with_seeds(global_bounds, global_seeds)
        else:
            global_bounds = None
            global_seeds = []
        if batch_size > 1:
            default_tile_count = (
                1
                if self.config.tile_workers == 1
                and self.config.n_tiles is None
                and self.config.tile_size is None
                else self.config.n_tiles
            )
            tiles = _partition_points(
                indexed,
                n_tiles=default_tile_count,
                tile_size=self.config.tile_size,
                worker_count=self.config.tile_workers,
                grid_shape=grid_shape,
            )
            completed, worker_count, retry_count = self._solve_tiles(
                model,
                tiles,
                context,
                global_seeds,
                grid_shape,
                global_bounds,
            )
            indexed_rows = [item for tile in completed for item in tile]
            indexed_rows.sort(key=lambda item: item[0])
            refinement_attempts = 0
            refined_points = 0
            if self.config.refine_suspicious:
                indexed_rows, refinement_attempts, refined_points = (
                    self._refine_suspicious_points(
                        model,
                        indexed_rows,
                        points,
                        grid_shape,
                        global_seeds,
                        global_bounds,
                        context,
                    )
                )
            rows = [row for _, row, _ in indexed_rows]
            attempted_count = (
                sum(count for _, _, count in indexed_rows) + refinement_attempts
            )
        else:
            solved = [
                _solve_point(
                    model,
                    params,
                    self.config,
                    index,
                    extra_guesses=global_seeds,
                )
                for index, params in enumerate(points)
            ]
            rows = [row for row, _ in solved]
            attempted_count = sum(count for _, count in solved)
            worker_count = 1
            retry_count = 0
            tiles = [indexed] if batch_size > 1 else []
            refined_points = 0
        metadata = {
            "attempted": attempted_count,
            "batch_size": batch_size,
            "requested_tile_workers": self.config.tile_workers,
            "tile_workers": worker_count,
            "tile_count": len(tiles),
            "worker_retries": retry_count,
            "global_seed_count": len(global_seeds),
            "refined_points": refined_points,
            "neighbor_continuation": batch_size > 1,
        }
        return CAMSolverOutput(rows if batch_size > 1 else rows[0], metadata=metadata)

    def _solve_tiles(
        self,
        model: Any,
        tiles: list[list[tuple[int, dict[str, Any]]]],
        context: Any | None,
        global_seeds: list[np.ndarray],
        grid_shape: tuple[int, ...],
        global_bounds: GuessBounds | None,
    ) -> tuple[list[list[tuple[int, list[Any], int]]], int, int]:
        options = self.config.model_dump()
        options["n_workers"] = 1
        if global_bounds is not None:
            options["guess_bounds"] = _bounds_config(global_bounds)
        worker_count = _effective_worker_count(
            self.config.tile_workers, len(tiles), context
        )
        results: list[list[tuple[int, list[Any], int]] | None] = [None] * len(tiles)
        checkpoint_store = getattr(context, "checkpoints", None)
        pending: list[int] = []
        for tile_index in range(len(tiles)):
            cached = None
            if checkpoint_store is not None and checkpoint_store.enabled:
                cached = checkpoint_store.load_chunk(_checkpoint_key(tile_index))
            if cached is None:
                pending.append(tile_index)
            else:
                results[tile_index] = cached

        completed_count = len(tiles) - len(pending)
        self._report_progress(
            context,
            completed_count,
            len(tiles),
            worker_count,
            self.config.tile_workers,
        )
        retry_count = 0
        while pending:
            if worker_count <= 1:
                config = MultistabilitySolverConfig.model_validate(options)
                for tile_index in pending:
                    tile_result = _solve_tile_with(
                        model,
                        tiles[tile_index],
                        config,
                        global_seeds,
                        grid_shape,
                    )
                    results[tile_index] = tile_result
                    self._save_tile_checkpoint(
                        checkpoint_store, tile_index, tile_result
                    )
                    completed_count += 1
                    self._report_progress(
                        context,
                        completed_count,
                        len(tiles),
                        worker_count,
                        self.config.tile_workers,
                    )
                pending = []
                break

            try:
                with _single_threaded_worker_environment():
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                        initializer=_init_tile_worker,
                        initargs=(model, options, global_seeds, grid_shape),
                    ) as pool:
                        futures = {
                            pool.submit(_solve_tile, tiles[tile_index]): tile_index
                            for tile_index in pending
                        }
                        for future in as_completed(futures):
                            tile_index = futures[future]
                            tile_result = future.result()
                            results[tile_index] = tile_result
                            self._save_tile_checkpoint(
                                checkpoint_store, tile_index, tile_result
                            )
                            completed_count += 1
                            self._report_progress(
                                context,
                                completed_count,
                                len(tiles),
                                worker_count,
                                self.config.tile_workers,
                            )
                pending = []
            except (BrokenProcessPool, OSError) as exc:
                if isinstance(exc, OSError) and getattr(exc, "winerror", None) not in {
                    8,
                    1455,
                }:
                    raise
                pending = [
                    tile_index for tile_index in pending if results[tile_index] is None
                ]
                retry_count += 1
                worker_count = max(1, worker_count // 2)
                self._report_retry(context, worker_count, exc)

        return (
            [result for result in results if result is not None],
            worker_count,
            retry_count,
        )

    def _discover_global_seeds(
        self,
        model: Any,
        points: list[dict[str, Any]],
        grid_shape: tuple[int, ...],
        bounds: GuessBounds | None,
        context: Any | None,
    ) -> list[np.ndarray]:
        explicit = _explicit_guesses(self.config, int(model.n_modes))
        if not self.config.discover_seeds or len(points) <= 1:
            return explicit

        representatives = _representative_indices(grid_shape, len(points))
        structured = _structured_seed_guesses(
            bounds,
            int(model.n_modes),
            self.config.seed_search_guesses,
        )
        options = self.config.model_dump()
        options.update(
            {
                "n_guesses": self.config.seed_search_guesses,
                "initial_guesses": structured or None,
                "n_workers": 1,
                "discover_seeds": False,
                "refine_suspicious": False,
            }
        )
        if bounds is not None:
            options["guess_bounds"] = _bounds_config(bounds)
        search_config = MultistabilitySolverConfig.model_validate(options)
        candidates: list[Any] = []
        for completed, point_index in enumerate(representatives, start=1):
            found, _ = _solve_point(
                model,
                points[point_index],
                search_config,
                point_index,
                extra_guesses=explicit,
            )
            candidates.extend(found)
            self._report_stage_progress(
                context,
                completed,
                len(representatives),
                "seed discovery",
            )
        unique = deduplicate_solutions(candidates, self.config.distance_tolerance)
        return [np.asarray(solution.state) for solution in unique]

    def _refine_suspicious_points(
        self,
        model: Any,
        indexed_rows: list[tuple[int, list[Any], int]],
        points: list[dict[str, Any]],
        grid_shape: tuple[int, ...],
        global_seeds: list[np.ndarray],
        global_bounds: GuessBounds | None,
        context: Any | None,
    ) -> tuple[list[tuple[int, list[Any], int]], int, int]:
        rows = {index: list(row) for index, row, _ in indexed_rows}
        pending: deque[int] = deque()
        queued: set[int] = set()
        attempted_target: dict[int, int] = {}

        def schedule_if_incomplete(index: int) -> None:
            neighbor_counts = [
                len(rows[neighbor])
                for neighbor in _neighbor_indices(index, grid_shape)
                if neighbor in rows
            ]
            target = max(neighbor_counts, default=0)
            if (
                len(rows[index]) < target
                and target > attempted_target.get(index, -1)
                and index not in queued
            ):
                pending.append(index)
                queued.add(index)

        for index in rows:
            schedule_if_incomplete(index)
        if not pending:
            return indexed_rows, 0, 0

        options = self.config.model_dump()
        options.update(
            {
                "n_guesses": self.config.refine_guesses,
                "n_workers": 1,
                "discover_seeds": False,
                "refine_suspicious": False,
            }
        )
        if global_bounds is not None:
            options["guess_bounds"] = _bounds_config(global_bounds)
        refine_config = MultistabilitySolverConfig.model_validate(options)
        attempts = 0
        changed: set[int] = set()
        completed = 0
        while pending:
            index = pending.popleft()
            queued.remove(index)
            neighbor_indices = [
                neighbor
                for neighbor in _neighbor_indices(index, grid_shape)
                if neighbor in rows
            ]
            target = max(
                (len(rows[neighbor]) for neighbor in neighbor_indices), default=0
            )
            if len(rows[index]) >= target or target <= attempted_target.get(index, -1):
                continue
            attempted_target[index] = target
            extra = [solution.state for solution in rows[index]]
            for neighbor in neighbor_indices:
                extra.extend(solution.state for solution in rows.get(neighbor, []))
            extra.extend(global_seeds)
            recovered, count = _solve_point(
                model,
                points[index],
                refine_config,
                index,
                extra_guesses=extra,
            )
            attempts += count
            merged = deduplicate_solutions(
                rows[index] + recovered,
                self.config.distance_tolerance,
            )
            if len(merged) > int(model.steady_state_capacity):
                raise SolutionCapacityError(
                    f"model {model.name!r} capacity "
                    f"{model.steady_state_capacity} exceeded during refinement"
                )
            if len(merged) > len(rows[index]):
                merged.sort(
                    key=lambda item: model.cam_solution_sort_key(
                        item.state, points[index]
                    )
                )
                rows[index] = merged
                changed.add(index)
                for neighbor in neighbor_indices:
                    schedule_if_incomplete(neighbor)
            completed += 1
            self._report_stage_progress(
                context,
                completed,
                completed + len(pending),
                "suspicious-point refinement",
            )
        return (
            [(index, rows[index], count) for index, _, count in indexed_rows],
            attempts,
            len(changed),
        )

    @staticmethod
    def _save_tile_checkpoint(
        checkpoint_store: Any | None,
        tile_index: int,
        tile_result: list[tuple[int, list[Any], int]],
    ) -> None:
        if checkpoint_store is not None and checkpoint_store.enabled:
            checkpoint_store.save_chunk(_checkpoint_key(tile_index), tile_result)

    @staticmethod
    def _report_progress(
        context: Any | None,
        completed: int,
        total: int,
        workers: int,
        requested_workers: int,
    ) -> None:
        if context is None:
            return
        progress = getattr(context, "progress", None)
        if progress is not None:
            worker_text = (
                str(workers)
                if workers == requested_workers
                else f"{workers}/{requested_workers} memory-limited"
            )
            progress.update(
                completed=completed,
                total=total,
                unit="tile",
                message=f"CAM tiles {completed}/{total}, workers={worker_text}",
                stage="solve_tiles",
            )
        cancellation = getattr(context, "cancellation", None)
        if cancellation is not None:
            cancellation.raise_if_cancelled()

    @staticmethod
    def _report_retry(context: Any | None, workers: int, exc: BaseException) -> None:
        if context is None:
            return
        progress = getattr(context, "progress", None)
        if progress is not None:
            progress.status(
                message=(
                    f"CAM worker pool failed ({type(exc).__name__}); "
                    f"retrying with {workers} workers"
                ),
                stage="solve_tiles",
            )

    @staticmethod
    def _report_stage_progress(
        context: Any | None,
        completed: int,
        total: int,
        label: str,
    ) -> None:
        if context is None:
            return
        progress = getattr(context, "progress", None)
        if progress is not None:
            stage = label.replace("-", "_").replace(" ", "_")
            progress.update(
                completed=completed,
                total=total,
                unit="point",
                message=f"CAM {label} {completed}/{total}",
                stage=stage,
            )
        cancellation = getattr(context, "cancellation", None)
        if cancellation is not None:
            cancellation.raise_if_cancelled()

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


_WORKER_MODEL: Any | None = None
_WORKER_CONFIG: MultistabilitySolverConfig | None = None
_WORKER_GLOBAL_SEEDS: list[np.ndarray] = []
_WORKER_GRID_SHAPE: tuple[int, ...] = ()


def _init_tile_worker(
    model: Any,
    options: dict[str, Any],
    global_seeds: list[np.ndarray],
    grid_shape: tuple[int, ...],
) -> None:
    """Initialize shared worker state once instead of pickling it per tile."""
    global _WORKER_MODEL, _WORKER_CONFIG, _WORKER_GLOBAL_SEEDS, _WORKER_GRID_SHAPE
    _WORKER_MODEL = model
    _WORKER_CONFIG = MultistabilitySolverConfig.model_validate(options)
    _WORKER_GLOBAL_SEEDS = global_seeds
    _WORKER_GRID_SHAPE = grid_shape


def _solve_tile(
    tile: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, list[Any], int]]:
    if _WORKER_MODEL is None or _WORKER_CONFIG is None:
        raise RuntimeError("CAM tile worker was not initialized")
    return _solve_tile_with(
        _WORKER_MODEL,
        tile,
        _WORKER_CONFIG,
        _WORKER_GLOBAL_SEEDS,
        _WORKER_GRID_SHAPE,
    )


def _solve_tile_with(
    model: Any,
    tile: list[tuple[int, dict[str, Any]]],
    config: MultistabilitySolverConfig,
    global_seeds: list[np.ndarray],
    grid_shape: tuple[int, ...],
) -> list[tuple[int, list[Any], int]]:
    solved: dict[int, list[Any]] = {}
    output: list[tuple[int, list[Any], int]] = []
    for tile_position, (index, params) in enumerate(tile):
        extra: list[np.ndarray] = []
        for neighbor in _neighbor_indices(index, grid_shape):
            extra.extend(solution.state for solution in solved.get(neighbor, []))
        extra.extend(global_seeds)
        point_config = config
        if (
            tile_position == 0
            and config.discover_seeds
            and config.seed_search_guesses > config.n_guesses
        ):
            point_config = config.model_copy(
                update={"n_guesses": config.seed_search_guesses}
            )
        row, attempted = _solve_point(
            model,
            params,
            point_config,
            index,
            extra_guesses=extra,
        )
        if not row and config.retry_guesses > point_config.n_guesses:
            retry_config = config.model_copy(update={"n_guesses": config.retry_guesses})
            row, retry_attempts = _solve_point(
                model,
                params,
                retry_config,
                index,
                extra_guesses=extra,
            )
            attempted += retry_attempts
        solved[index] = row
        output.append((index, row, attempted))
    return output


def _partition_points(
    indexed: list[tuple[int, dict[str, Any]]],
    *,
    n_tiles: int | None,
    tile_size: int | None,
    worker_count: int,
    grid_shape: tuple[int, ...] | None = None,
) -> list[list[tuple[int, dict[str, Any]]]]:
    if not indexed:
        return []
    if tile_size is not None:
        return [
            indexed[start : start + tile_size]
            for start in range(0, len(indexed), tile_size)
        ]
    target = n_tiles or max(worker_count * 4, 16)
    if grid_shape is not None and len(grid_shape) == 2:
        return _partition_2d_points(indexed, grid_shape, target)
    tile_count = min(target, len(indexed))
    edges = np.linspace(0, len(indexed), tile_count + 1, dtype=int)
    return [
        indexed[int(edges[index]) : int(edges[index + 1])]
        for index in range(tile_count)
        if edges[index + 1] > edges[index]
    ]


def _partition_2d_points(
    indexed: list[tuple[int, dict[str, Any]]],
    grid_shape: tuple[int, ...],
    target_tiles: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    n_rows, n_cols = grid_shape
    tile_rows = min(
        n_rows,
        max(1, int(round(math.sqrt(target_tiles * n_rows / max(n_cols, 1))))),
    )
    tile_cols = min(n_cols, max(1, math.ceil(target_tiles / tile_rows)))
    row_edges = np.linspace(0, n_rows, tile_rows + 1, dtype=int)
    col_edges = np.linspace(0, n_cols, tile_cols + 1, dtype=int)
    tiles: list[list[tuple[int, dict[str, Any]]]] = []
    for row_tile in range(tile_rows):
        for col_tile in range(tile_cols):
            row_start, row_stop = (
                int(row_edges[row_tile]),
                int(row_edges[row_tile + 1]),
            )
            col_start, col_stop = (
                int(col_edges[col_tile]),
                int(col_edges[col_tile + 1]),
            )
            local_rows = row_stop - row_start
            local_cols = col_stop - col_start
            order = _spiral_order(local_rows, local_cols)
            tile = [
                indexed[
                    int(
                        np.ravel_multi_index(
                            (row_start + row, col_start + col), grid_shape
                        )
                    )
                ]
                for row, col in order
            ]
            if tile:
                tiles.append(tile)
    return tiles


def _spiral_order(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    center_row, center_col = n_rows // 2, n_cols // 2
    radius_max = max(
        center_row,
        n_rows - 1 - center_row,
        center_col,
        n_cols - 1 - center_col,
    )
    order: list[tuple[int, int]] = []
    for radius in range(radius_max + 1):
        top, bottom = center_row - radius, center_row + radius
        left, right = center_col - radius, center_col + radius
        if 0 <= top < n_rows:
            order.extend(
                (top, col) for col in range(max(0, left), min(n_cols, right + 1))
            )
        if 0 <= bottom < n_rows and bottom != top:
            order.extend(
                (bottom, col) for col in range(max(0, left), min(n_cols, right + 1))
            )
        if radius > 0 and 0 <= left < n_cols:
            order.extend(
                (row, left) for row in range(max(0, top + 1), min(n_rows, bottom))
            )
        if radius > 0 and 0 <= right < n_cols and right != left:
            order.extend(
                (row, right) for row in range(max(0, top + 1), min(n_rows, bottom))
            )
    return order


def _context_grid_shape(context: Any | None, batch_size: int) -> tuple[int, ...]:
    grid = getattr(context, "parameter_grid", None)
    shape = tuple(getattr(grid, "shape", ()))
    if shape and int(np.prod(shape)) == batch_size:
        return shape
    return (batch_size,)


def _representative_indices(grid_shape: tuple[int, ...], point_count: int) -> list[int]:
    indices = {0, point_count - 1, point_count // 2}
    if grid_shape and int(np.prod(grid_shape)) == point_count:
        corners = product(*((0, size - 1) for size in grid_shape))
        for corner in corners:
            indices.add(int(np.ravel_multi_index(corner, grid_shape)))
            if len(indices) >= 17:
                break
        center = tuple(size // 2 for size in grid_shape)
        indices.add(int(np.ravel_multi_index(center, grid_shape)))
    return sorted(index for index in indices if 0 <= index < point_count)


def _representative_params(
    points: list[dict[str, Any]], grid_shape: tuple[int, ...]
) -> list[dict[str, Any]]:
    return [points[index] for index in _representative_indices(grid_shape, len(points))]


def _scan_guess_bounds(
    model: Any,
    representative_params: list[dict[str, Any]],
    config: MultistabilitySolverConfig,
) -> GuessBounds | None:
    if isinstance(config.guess_bounds, GuessBoundsConfig):
        return GuessBounds.from_config(config.guess_bounds, int(model.n_modes))
    if config.guess_bounds != "auto":
        return None
    inferred = [
        infer_guess_bounds(
            model,
            params,
            seed=None if config.seed is None else config.seed + index,
            starts=config.bounds_inference_starts,
        )
        for index, params in enumerate(representative_params)
    ]
    candidate_sets = [
        bounds.diag_candidates
        for bounds in inferred
        if bounds.diag_candidates is not None
    ]
    return GuessBounds(
        diag_lower=np.min([bounds.diag_lower for bounds in inferred], axis=0),
        diag_upper=np.max([bounds.diag_upper for bounds in inferred], axis=0),
        offdiag_scale=np.max([bounds.offdiag_scale for bounds in inferred], axis=0),
        diag_candidates=(
            _unique_rows(np.vstack(candidate_sets)) if candidate_sets else None
        ),
    )


def _expand_bounds_with_seeds(
    bounds: GuessBounds | None,
    seeds: list[np.ndarray],
    margin: float = 2.0,
) -> GuessBounds | None:
    if not seeds:
        return bounds
    diagonals = np.asarray([np.real(np.diag(state)) for state in seeds])
    seed_lower = np.min(diagonals, axis=0)
    seed_upper = np.max(diagonals, axis=0)
    n_modes = diagonals.shape[1]
    if bounds is None:
        lower = np.minimum(seed_lower, 0.0)
        upper = np.maximum(seed_upper, 0.0)
        offdiag = np.ones((n_modes, n_modes), dtype=float)
        existing = None
    else:
        lower = np.minimum(bounds.diag_lower, seed_lower)
        upper = np.maximum(bounds.diag_upper, seed_upper)
        offdiag = np.array(bounds.offdiag_scale, copy=True)
        existing = bounds.diag_candidates
    lower = np.where(lower < 0.0, margin * lower, lower)
    upper = np.where(upper > 0.0, margin * upper, upper)
    for row in range(n_modes):
        for col in range(row + 1, n_modes):
            scale = margin * max(abs(state[row, col]) for state in seeds)
            offdiag[row, col] = max(offdiag[row, col], scale, 1e-8)
            offdiag[col, row] = offdiag[row, col]
    candidates = (
        diagonals
        if existing is None
        else _unique_rows(np.vstack((existing, diagonals)))
    )
    return GuessBounds(lower, upper, offdiag, candidates)


def _bounds_config(bounds: GuessBounds) -> GuessBoundsConfig:
    return GuessBoundsConfig(
        diag_lower=bounds.diag_lower.tolist(),
        diag_upper=bounds.diag_upper.tolist(),
        offdiag_scale=bounds.offdiag_scale.tolist(),
    )


def _structured_seed_guesses(
    bounds: GuessBounds | None,
    n_modes: int,
    limit: int,
) -> list[np.ndarray]:
    guesses = [
        np.zeros((n_modes, n_modes), dtype=complex),
        np.eye(n_modes, dtype=complex),
    ]
    if bounds is None or limit <= len(guesses):
        return guesses[:limit]
    if bounds.diag_candidates is not None and len(bounds.diag_candidates):
        diagonal_combinations = iter(bounds.diag_candidates)
    else:
        diagonal_levels = [
            np.unique(
                np.asarray(
                    [
                        bounds.diag_lower[mode],
                        0.0,
                        0.5 * (bounds.diag_lower[mode] + bounds.diag_upper[mode]),
                        bounds.diag_upper[mode],
                    ]
                )
            )
            for mode in range(n_modes)
        ]
        diagonal_combinations = product(*diagonal_levels)
    phases = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    for diagonal in diagonal_combinations:
        base = np.diag(np.asarray(diagonal, dtype=float)).astype(complex)
        guesses.append(base)
        if len(guesses) >= limit:
            break
        for fraction in (0.5, 1.0):
            for phase in phases:
                state = base.copy()
                for row in range(n_modes):
                    for col in range(row + 1, n_modes):
                        amplitude = fraction * bounds.offdiag_scale[row, col]
                        value = amplitude * np.exp(1j * phase)
                        state[row, col] = value
                        state[col, row] = value.conjugate()
                guesses.append(state)
                if len(guesses) >= limit:
                    break
            if len(guesses) >= limit:
                break
        if len(guesses) >= limit:
            break
    return guesses[:limit]


def _neighbor_indices(index: int, grid_shape: tuple[int, ...]) -> list[int]:
    if not grid_shape or int(np.prod(grid_shape)) <= 1:
        return []
    coordinate = list(np.unravel_index(index, grid_shape))
    neighbors: list[int] = []
    for axis, size in enumerate(grid_shape):
        for offset in (-1, 1):
            value = coordinate[axis] + offset
            if 0 <= value < size:
                neighbor = list(coordinate)
                neighbor[axis] = value
                neighbors.append(int(np.ravel_multi_index(tuple(neighbor), grid_shape)))
    return neighbors


def _explicit_guesses(
    config: MultistabilitySolverConfig, n_modes: int
) -> list[np.ndarray]:
    if config.initial_guesses is None:
        return []
    values = np.asarray(config.initial_guesses, dtype=complex)
    if values.shape == (n_modes, n_modes):
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-2:] != (n_modes, n_modes):
        raise ValueError("initial_guesses must have shape (n,n) or (g,n,n)")
    return [0.5 * (value + value.conj().T) for value in values]


def _unique_rows(values: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    unique: list[np.ndarray] = []
    for value in np.asarray(values):
        scale = max(1.0, float(np.linalg.norm(value)))
        if all(np.linalg.norm(value - known) > tolerance * scale for known in unique):
            unique.append(np.asarray(value))
    return np.asarray(unique)


def _effective_worker_count(
    requested: int, tile_count: int, context: Any | None
) -> int:
    workers = min(requested, tile_count, os.cpu_count() or requested)
    resources = getattr(context, "resources", None)
    cpu_limit = getattr(resources, "cpu_worker_limit", None)
    if cpu_limit is not None:
        workers = min(workers, int(cpu_limit))

    budgets = [
        value
        for value in (
            _available_memory_mib(),
            getattr(resources, "memory_limit_mib", None),
        )
        if value is not None
    ]
    if budgets:
        usable_memory = max(min(int(value) for value in budgets) - 512, 256)
        workers = min(workers, max(1, usable_memory // 256))
    return max(workers, 1)


def _available_memory_mib() -> int | None:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        try:
            success = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).GlobalMemoryStatusEx(ctypes.byref(status))
        except OSError:
            return None
        if success:
            return int(status.available_physical // (1 << 20))
        return None
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        pages = int(sysconf("SC_AVPHYS_PAGES"))
        page_size = int(sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return None
    return pages * page_size // (1 << 20)


@contextmanager
def _single_threaded_worker_environment() -> Any:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _checkpoint_key(tile_index: int) -> str:
    return f"cam-multistability-tile-{tile_index:08d}"


def _solve_point(
    model: Any,
    params: dict[str, Any],
    config: MultistabilitySolverConfig,
    point_index: int,
    *,
    extra_guesses: list[np.ndarray] | None = None,
) -> tuple[list[Any], int]:
    guesses = _make_guesses(
        model,
        params,
        config,
        point_index,
        extra_guesses=extra_guesses,
    )
    root_system = None
    if config.method in {"auto", "root"}:
        root_system = prepare_root_system(
            model,
            params,
            use_jacobian=config.use_jacobian,
            reference_state=guesses[0] if guesses else None,
        )

    def solve_guess(guess: Any) -> Any:
        return solve_single_state(
            model,
            params,
            guess,
            method=config.method,
            tolerance=config.tolerance,
            residual_tolerance=config.residual_tolerance,
            use_jacobian=config.use_jacobian,
            root_system=root_system,
        )

    capacity = int(model.steady_state_capacity)
    accepted: list[Any] = []
    accepted_vectors: list[np.ndarray] = []
    duplicate_streak = 0
    attempted_count = 0

    def process(solution: Any) -> None:
        nonlocal duplicate_streak
        if not (solution.success and solution.residual <= config.residual_tolerance):
            return
        vector = np.asarray(matrix_to_vector(solution.state))
        matching = next(
            (
                index
                for index, known in enumerate(accepted_vectors)
                if np.linalg.norm(vector - known) < config.distance_tolerance
            ),
            None,
        )
        if matching is not None:
            if solution.residual < accepted[matching].residual:
                accepted[matching] = solution
                accepted_vectors[matching] = vector
            duplicate_streak += 1
            return
        if len(accepted) >= capacity:
            raise SolutionCapacityError(
                f"model {model.name!r} declares capacity {capacity} but more "
                "distinct states were found"
            )
        accepted.append(solution)
        accepted_vectors.append(vector)
        duplicate_streak = 0

    def should_stop() -> bool:
        return len(accepted) >= capacity and (
            duplicate_streak >= config.capacity_patience
        )

    if config.n_workers == 1:
        for guess in guesses:
            process(solve_guess(guess))
            attempted_count += 1
            if should_stop():
                break
    else:
        with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
            chunk_size = max(1, config.n_workers * 2)
            for start in range(0, len(guesses), chunk_size):
                chunk = guesses[start : start + chunk_size]
                for solution in pool.map(solve_guess, chunk):
                    process(solution)
                    attempted_count += 1
                if should_stop():
                    break
    accepted.sort(key=lambda item: model.cam_solution_sort_key(item.state, params))
    return accepted, attempted_count


def _make_guesses(
    model: Any,
    params: dict[str, Any],
    config: MultistabilitySolverConfig,
    point_index: int,
    *,
    extra_guesses: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    n_modes = int(model.n_modes)
    seed = None if config.seed is None else config.seed + point_index
    explicit = _explicit_guesses(config, n_modes)
    if config.guess_bounds == "auto":
        bounds = infer_guess_bounds(model, params, seed=seed)
    elif isinstance(config.guess_bounds, GuessBoundsConfig):
        bounds = GuessBounds.from_config(config.guess_bounds, n_modes)
    else:
        bounds = None
    if bounds is None:
        generated = random_hermitian_guesses(
            n_modes, config.n_guesses, config.guess_scale, seed
        )
    else:
        generated = bounds.sample(
            config.n_guesses,
            seed,
            config.tail_fraction,
            config.tail_orders,
        )
    base_guesses = explicit + generated
    ordered = list(extra_guesses or []) + base_guesses
    unique: list[np.ndarray] = []
    for guess in ordered:
        state = 0.5 * (np.asarray(guess) + np.asarray(guess).conj().T)
        if all(np.linalg.norm(state - known) > 1e-10 for known in unique):
            unique.append(state)
    return unique
