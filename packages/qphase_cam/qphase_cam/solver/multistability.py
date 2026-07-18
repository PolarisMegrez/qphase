"""Multi-start CAM solver plugin."""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

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
    n_tiles: int | None = Field(default=None, ge=1)
    tile_size: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_tile_partition(self) -> MultistabilitySolverConfig:
        if self.n_tiles is not None and self.tile_size is not None:
            raise ValueError("configure only one of n_tiles or tile_size")
        return self


class MultistabilitySolver(CAMSolver):
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
        if batch_size > 1 and self.config.tile_workers > 1:
            tiles = _partition_points(
                indexed,
                n_tiles=self.config.n_tiles,
                tile_size=self.config.tile_size,
                worker_count=self.config.tile_workers,
            )
            completed, worker_count, retry_count = self._solve_tiles(
                model, tiles, context
            )
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
            worker_count = 1
            retry_count = 0
            tiles = [indexed] if batch_size > 1 else []
        metadata = {
            "attempted": attempted_count,
            "batch_size": batch_size,
            "requested_tile_workers": self.config.tile_workers,
            "tile_workers": worker_count,
            "tile_count": len(tiles),
            "worker_retries": retry_count,
        }
        return CAMSolverOutput(rows if batch_size > 1 else rows[0], metadata=metadata)

    def _solve_tiles(
        self,
        model: Any,
        tiles: list[list[tuple[int, dict[str, Any]]]],
        context: Any | None,
    ) -> tuple[list[list[tuple[int, list[Any], int]]], int, int]:
        options = self.config.model_dump()
        options["n_workers"] = 1
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
                    tile_result = _solve_tile_with(model, tiles[tile_index], config)
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
                        initargs=(model, options),
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
            progress.report(
                completed / max(total, 1),
                message=f"CAM tiles {completed}/{total}, workers={worker_text}",
                stage="solve",
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
            progress.report(
                None,
                message=(
                    f"CAM worker pool failed ({type(exc).__name__}); "
                    f"retrying with {workers} workers"
                ),
                stage="solve",
            )

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


def _init_tile_worker(model: Any, options: dict[str, Any]) -> None:
    """Initialize shared worker state once instead of pickling it per tile."""
    global _WORKER_MODEL, _WORKER_CONFIG
    _WORKER_MODEL = model
    _WORKER_CONFIG = MultistabilitySolverConfig.model_validate(options)


def _solve_tile(
    tile: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, list[Any], int]]:
    if _WORKER_MODEL is None or _WORKER_CONFIG is None:
        raise RuntimeError("CAM tile worker was not initialized")
    return _solve_tile_with(_WORKER_MODEL, tile, _WORKER_CONFIG)


def _solve_tile_with(
    model: Any,
    tile: list[tuple[int, dict[str, Any]]],
    config: MultistabilitySolverConfig,
) -> list[tuple[int, list[Any], int]]:
    return [
        (index, *_solve_point(model, params, config, index)) for index, params in tile
    ]


def _partition_points(
    indexed: list[tuple[int, dict[str, Any]]],
    *,
    n_tiles: int | None,
    tile_size: int | None,
    worker_count: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    if not indexed:
        return []
    if tile_size is not None:
        return [
            indexed[start : start + tile_size]
            for start in range(0, len(indexed), tile_size)
        ]
    target = n_tiles or max(worker_count * 4, 16)
    tile_count = min(target, len(indexed))
    edges = np.linspace(0, len(indexed), tile_count + 1, dtype=int)
    return [
        indexed[int(edges[index]) : int(edges[index + 1])]
        for index in range(tile_count)
        if edges[index + 1] > edges[index]
    ]


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
