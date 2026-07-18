"""QPhase engine for coherent-amplitude matrix analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol
from qphase.core.scan import ParameterGrid, execute_pointwise

from qphase_cam.errors import SolutionCapacityError
from qphase_cam.result import CAMResult
from qphase_cam.state import CAMSolution


class EngineConfig(BaseModel):
    """CAM engine configuration; task details belong to solver plugins."""

    model_config = ConfigDict(extra="allow")


class Engine(EngineBase):
    name: ClassVar[str] = "CAM"
    description: ClassVar[str] = "Coherent-amplitude matrix analysis engine"
    config_schema: ClassVar[type[EngineConfig]] = EngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        required_plugins={"backend", "model", "cam_solver"},
        optional_plugins={"cam_postprocessor"},
    )

    def __init__(
        self,
        config: EngineConfig | None = None,
        plugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.config = config or EngineConfig()
        self.plugins = plugins or {}

    def run(
        self,
        data: Any | None = None,
        context: Any | None = None,
        progress_cb: Callable[..., Any] | None = None,
    ) -> ResultProtocol:
        del data
        model = self._required("model")
        backend = self._required("backend")
        solver = self._required("cam_solver")
        reporter = context.progress if context is not None else progress_cb
        if reporter:
            reporter(0.0, None, "Solving CAM fixed points", "solve")
        grid = context.parameter_grid if context is not None else None
        if grid is not None and solver.name == "continuation":
            raise ValueError(
                "continuation cannot be combined with an external ScanSpec"
            )
        output = self._solve_grid(solver, model, backend, grid, context)
        result = self._pack(output, model, grid)
        result.meta.update(
            {
                "engine": "cam",
                "model": model.name,
                "solver": solver.name,
                **output.metadata,
            }
        )
        postprocessors = self.plugins.get("cam_postprocessor", {})
        if postprocessors and not isinstance(postprocessors, dict):
            postprocessors = {postprocessors.name: postprocessors}
        for name, processor in postprocessors.items():
            result.postprocess.update(processor.process(result, model, backend))
            result.meta.setdefault("postprocessors", []).append(name)
            if processor.result_metadata:
                result.meta.setdefault("postprocessor_metadata", {})[name] = dict(
                    processor.result_metadata
                )
        if reporter:
            reporter(1.0, 0.0, "CAM analysis complete", "complete")
        return result

    def _solve_grid(
        self,
        solver: Any,
        model: Any,
        backend: Any,
        grid: ParameterGrid | None,
        context: Any | None,
    ) -> Any:
        if grid is None:
            return solver.solve(model, backend)
        flattened = self._model_scan_params(model, grid)
        base_params = dict(model.params)
        if getattr(solver, "supports_batch", False):
            self._replace_model_params(model, {**base_params, **flattened})
            output = solver.solve(model, backend)
            output.axes = dict(grid.axes)
            output.metadata.update(
                {"scan_shape": grid.shape, "scan_combine": grid.combine}
            )
            return output

        def solve_point(point: Any) -> list[CAMSolution]:
            params = dict(base_params)
            for target, value in point.targets.items():
                params[target.rsplit(".", 1)[-1]] = value
            self._replace_model_params(model, params)
            return list(solver.solve(model, backend).solutions)

        rows = execute_pointwise(grid, solve_point, context=context)
        self._replace_model_params(model, {**base_params, **flattened})
        from qphase_cam.state import CAMSolverOutput

        return CAMSolverOutput(
            rows,
            axes=dict(grid.axes),
            metadata={"scan_shape": grid.shape, "scan_combine": grid.combine},
        )

    @staticmethod
    def _model_scan_params(model: Any, grid: ParameterGrid) -> dict[str, Any]:
        expected_prefix = f"model.{model.name}."
        output: dict[str, Any] = {}
        for target, values in grid.target_arrays(flatten=True).items():
            if not target.startswith(expected_prefix):
                raise ValueError(
                    f"CAM scan target {target!r} must target "
                    f"{expected_prefix}<parameter>"
                )
            parameter = target.removeprefix(expected_prefix)
            if "." in parameter or parameter not in model.params:
                raise ValueError(f"unknown CAM model scan target {target!r}")
            output[parameter] = values
        return output

    @staticmethod
    def _replace_model_params(model: Any, params: dict[str, Any]) -> None:
        if hasattr(model, "_params"):
            model._params = params
            return
        current = getattr(model, "params", None)
        if isinstance(current, dict):
            current.clear()
            current.update(params)
            return
        raise TypeError(f"model {model.name!r} does not expose mutable parameters")

    def _required(self, name: str) -> Any:
        plugin = self.plugins.get(name)
        if plugin is None or isinstance(plugin, dict):
            raise RuntimeError(f"CAM engine requires exactly one {name!r} plugin")
        return plugin

    def _pack(
        self, output: Any, model: Any, grid: ParameterGrid | None = None
    ) -> CAMResult:
        solutions, grid_shape = self._solution_grid(output.solutions, grid)
        capacity = int(model.steady_state_capacity)
        n_modes = int(model.n_modes)
        states = np.full(
            grid_shape + (capacity, n_modes, n_modes), np.nan + 1j * np.nan
        )
        residuals = np.full(grid_shape + (capacity,), np.nan)
        success = np.zeros(grid_shape + (capacity,), dtype=bool)
        valid = np.zeros(grid_shape + (capacity,), dtype=bool)
        counts = np.zeros(grid_shape or (), dtype=int)
        for grid_index, row in solutions:
            if len(row) > capacity:
                raise SolutionCapacityError(
                    f"model {model.name!r} capacity {capacity} exceeded"
                )
            row.sort(
                key=lambda item: model.cam_solution_sort_key(item.state, model.params)
            )
            counts[grid_index] = len(row)
            for slot, solution in enumerate(row):
                states[grid_index + (slot,)] = solution.state
                residuals[grid_index + (slot,)] = solution.residual
                success[grid_index + (slot,)] = solution.success
                valid[grid_index + (slot,)] = True
        params = dict(model.params)
        if grid is not None:
            for name, values in list(params.items()):
                array = np.asarray(values)
                if array.ndim > 0 and array.size == grid.size:
                    params[name] = array.reshape(grid.shape)
        return CAMResult(
            states,
            residuals,
            success,
            valid,
            counts,
            params,
            axes=output.axes,
            meta=dict(output.metadata),
        )

    def _solution_grid(
        self, raw: Any, grid: ParameterGrid | None = None
    ) -> tuple[list[tuple[tuple[int, ...], list[CAMSolution]]], tuple[int, ...]]:
        if grid is not None:
            rows = list(raw)
            if len(rows) != grid.size:
                raise ValueError(
                    f"CAM solver returned {len(rows)} rows for {grid.size} scan points"
                )
            return [
                (
                    tuple(int(value) for value in np.unravel_index(index, grid.shape)),
                    list(row),
                )
                for index, row in enumerate(rows)
            ], grid.shape
        if not raw or isinstance(raw[0], CAMSolution):
            return [((), list(raw))], ()
        rows = list(raw)
        return [((index,), list(row)) for index, row in enumerate(rows)], (len(rows),)
