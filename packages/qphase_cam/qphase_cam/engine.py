"""QPhase engine for coherent-amplitude matrix analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol

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
        progress_cb: Callable[..., Any] | None = None,
    ) -> ResultProtocol:
        del data
        model = self._required("model")
        backend = self._required("backend")
        solver = self._required("cam_solver")
        if progress_cb:
            progress_cb(0.0, None, "Solving CAM fixed points", "solve")
        output = solver.solve(model, backend)
        result = self._pack(output, model)
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
        if progress_cb:
            progress_cb(1.0, 0.0, "CAM analysis complete", "complete")
        return result

    def _required(self, name: str) -> Any:
        plugin = self.plugins.get(name)
        if plugin is None or isinstance(plugin, dict):
            raise RuntimeError(f"CAM engine requires exactly one {name!r} plugin")
        return plugin

    def _pack(self, output: Any, model: Any) -> CAMResult:
        solutions, grid_shape = self._solution_grid(output.solutions)
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
        return CAMResult(
            states,
            residuals,
            success,
            valid,
            counts,
            dict(model.params),
            axes=output.axes,
            meta=dict(output.metadata),
        )

    def _solution_grid(
        self, raw: Any
    ) -> tuple[list[tuple[tuple[int, ...], list[CAMSolution]]], tuple[int, ...]]:
        if not raw or isinstance(raw[0], CAMSolution):
            return [((), list(raw))], ()
        rows = list(raw)
        return [((index,), list(row)) for index, row in enumerate(rows)], (len(rows),)
