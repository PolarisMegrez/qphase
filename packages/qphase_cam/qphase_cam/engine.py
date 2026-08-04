"""QPhase engine for coherent-amplitude matrix analysis."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol
from qphase.core.scan import ParameterGrid, execute_pointwise

from qphase_cam.bifurcation_result import CAMBifurcationResult
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
        model = self._required("model")
        backend = self._required("backend")
        solver = self._required("cam_solver")
        reporter = context.progress if context is not None else None
        if reporter is not None:
            reporter.status("Solving CAM fixed points", stage="solve")
        elif progress_cb is not None:
            progress_cb(None, None, "Solving CAM fixed points", "solve")
        grid = context.parameter_grid if context is not None else None
        output_kind = getattr(solver, "output_kind", "fixed_points")
        if grid is not None and output_kind == "bifurcation_candidates":
            raise ValueError("bifurcation solver cannot be combined with ScanSpec")
        if grid is not None and solver.name == "continuation":
            raise ValueError(
                "continuation cannot be combined with an external ScanSpec"
            )
        output = self._solve_grid(solver, model, backend, grid, context, data)
        result = (
            self._pack_bifurcation(output, model)
            if output_kind == "bifurcation_candidates"
            else self._pack(output, model, grid)
        )
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
            if result.result_kind not in processor.accepted_result_kinds:
                raise ValueError(
                    f"postprocessor {processor.name!r} does not accept "
                    f"result kind {result.result_kind!r}"
                )
            result.postprocess.update(processor.process(result, model, backend))
            result.meta.setdefault("postprocessors", []).append(name)
            if processor.result_metadata:
                result.meta.setdefault("postprocessor_metadata", {})[name] = dict(
                    processor.result_metadata
                )
        if reporter is not None:
            reporter.status("CAM analysis complete", stage="complete")
        elif progress_cb is not None:
            progress_cb(1.0, None, "CAM analysis complete", "complete")
        return result

    def _solve_grid(
        self,
        solver: Any,
        model: Any,
        backend: Any,
        grid: ParameterGrid | None,
        context: Any | None,
        data: Any | None,
    ) -> Any:
        if grid is None:
            return self._invoke_solver(solver, model, backend, context, data)
        flattened = self._model_scan_params(model, grid)
        base_params = dict(model.params)
        if getattr(solver, "supports_batch", False):
            self._replace_model_params(model, {**base_params, **flattened})
            output = self._invoke_solver(solver, model, backend, context, data)
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
            return list(
                self._invoke_solver(solver, model, backend, context, data).solutions
            )

        rows = execute_pointwise(grid, solve_point, context=context)
        self._replace_model_params(model, {**base_params, **flattened})
        from qphase_cam.state import CAMSolverOutput

        return CAMSolverOutput(
            rows,
            axes=dict(grid.axes),
            metadata={"scan_shape": grid.shape, "scan_combine": grid.combine},
        )

    @staticmethod
    def _invoke_solver(
        solver: Any,
        model: Any,
        backend: Any,
        context: Any | None,
        data: Any | None,
    ) -> Any:
        parameters = inspect.signature(solver.solve).parameters
        kwargs = {}
        if "context" in parameters:
            kwargs["context"] = context
        if "data" in parameters:
            kwargs["data"] = data
        return solver.solve(model, backend, **kwargs)

    @staticmethod
    def _pack_bifurcation(output: Any, model: Any) -> CAMBifurcationResult:
        from qphase_cam.core.coordinates import vector_to_matrix

        candidates = list(output.candidates)
        n_state = int(model.n_modes) ** 2
        control_names = tuple(output.metadata.get("control_names", ()))
        vectors = np.asarray(
            [candidate.state_vector for candidate in candidates], dtype=float
        ).reshape((-1, n_state))
        states = np.asarray(vector_to_matrix(vectors, int(model.n_modes)))
        controls = np.asarray(
            [
                [candidate.controls[name] for name in control_names]
                for candidate in candidates
            ],
            dtype=float,
        ).reshape((len(candidates), len(control_names)))
        diagnostic_names = sorted(
            set().union(*(candidate.metadata for candidate in candidates))
        ) if candidates else []
        diagnostics = {
            name: np.asarray(
                [candidate.metadata.get(name, np.nan) for candidate in candidates]
            )
            for name in diagnostic_names
        }
        return CAMBifurcationResult(
            states=states,
            state_vectors=vectors,
            control_values=controls,
            control_names=control_names,
            full_residual_norm=np.asarray(
                [candidate.full_residual_norm for candidate in candidates]
            ),
            search_residual_norm=np.asarray(
                [candidate.search_residual_norm for candidate in candidates]
            ),
            success=np.asarray([candidate.success for candidate in candidates]),
            status=np.asarray([candidate.status for candidate in candidates]),
            method=np.asarray([candidate.method for candidate in candidates]),
            diagnostics=diagnostics,
            meta={
                **output.metadata,
                "target": output.target,
                "order": output.order,
                "model": model.name,
            },
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
