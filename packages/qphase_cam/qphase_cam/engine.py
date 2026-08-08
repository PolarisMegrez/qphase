"""QPhase engine for coherent-amplitude matrix analysis."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol
from qphase.core.scan import ParameterGrid, execute_pointwise

from qphase_cam.bifurcation_result import (
    CAMBifurcationBranchTable,
    CAMBifurcationResult,
    CAMBifurcationScanResult,
)
from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.errors import BifurcationCapabilityError, SolutionCapacityError
from qphase_cam.result import CAMResult
from qphase_cam.state import CAMBifurcationOutput, CAMSolution


class EngineConfig(BaseModel):
    """CAM engine configuration; task details belong to solver plugins."""

    model_config = ConfigDict(extra="allow")
    case_failure_policy: Literal["abort", "record"] = "abort"


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
        if grid is not None and solver.name == "continuation":
            raise ValueError(
                "continuation cannot be combined with an external ScanSpec"
            )
        result: CAMBifurcationScanResult | CAMBifurcationResult | CAMResult
        if grid is not None and output_kind == "bifurcation_candidates":
            result = self._solve_bifurcation_grid(
                solver, model, backend, grid, context, data
            )
        else:
            output = self._solve_grid(solver, model, backend, grid, context, data)
            result = (
                self._pack_bifurcation(output, model)
                if output_kind == "bifurcation_candidates"
                else self._pack(output, model, grid)
            )
            result.meta.update(output.metadata)
        result.meta.update(
            {"engine": "cam", "model": model.name, "solver": solver.name}
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

    def _solve_bifurcation_grid(
        self,
        solver: Any,
        model: Any,
        backend: Any,
        grid: ParameterGrid,
        context: Any | None,
        data: Any | None,
    ) -> CAMBifurcationScanResult:
        flattened = self._model_scan_params(model, grid)
        controls = set(getattr(getattr(solver, "config", None), "controls", ()))
        scanned_parameters = set(flattened)
        overlap = controls & scanned_parameters
        if overlap:
            raise ValueError(
                "bifurcation ScanSpec targets must be fixed case parameters, "
                f"not solver controls: {sorted(overlap)}"
            )
        base_params = dict(model.params)

        def solve_case(point: Any) -> CAMBifurcationResult:
            params = dict(base_params)
            for target, value in point.targets.items():
                params[target.rsplit(".", 1)[-1]] = value
            self._replace_model_params(model, params)
            try:
                output = self._invoke_solver(solver, model, backend, context, data)
            except MemoryError:
                raise
            except Exception as exc:
                if self.config.case_failure_policy == "abort":
                    raise
                target = getattr(solver, "target", None)
                control_names = tuple(
                    getattr(getattr(solver, "config", None), "controls", ())
                )
                output = CAMBifurcationOutput(
                    candidates=[],
                    target=str(getattr(target, "name", "equilibrium_multiplicity")),
                    order=int(getattr(target, "order", 0)),
                    metadata={
                        "control_names": control_names,
                        "case_status": "error",
                        "case_error_type": type(exc).__name__,
                        "case_error_message": str(exc),
                        "case_flat_index": int(point.flat_index),
                        "case_index": tuple(int(value) for value in point.index),
                        "case_parameters": dict(point.values),
                    },
                )
            result = self._pack_bifurcation(output, model)
            result.meta.update(output.metadata)
            result.meta.setdefault("case_status", "complete")
            return result

        try:
            cases = execute_pointwise(grid, solve_case, context=context)
        finally:
            self._replace_model_params(model, base_params)
        offsets = np.zeros(len(cases) + 1, dtype=int)
        offsets[1:] = np.cumsum([len(case.states) for case in cases])
        combined = CAMBifurcationResult.concatenate(cases)
        parameter_arrays = grid.parameter_arrays(flatten=True)
        case_params: dict[str, np.ndarray] = {}
        for name, value in base_params.items():
            if np.asarray(value).ndim == 0:
                case_params[name] = np.full(grid.size, value)
        for axis_name, target in grid.targets.items():
            case_params[target.rsplit(".", 1)[-1]] = np.asarray(
                parameter_arrays[axis_name]
            )
        return CAMBifurcationScanResult(
            case_axes={name: np.asarray(values) for name, values in grid.axes.items()},
            case_shape=grid.shape,
            case_params=case_params,
            candidate_offsets=offsets,
            candidates=combined,
            case_metadata=tuple(dict(case.meta) for case in cases),
            meta={
                "scan_shape": grid.shape,
                "scan_combine": grid.combine,
                "scan_targets": dict(grid.targets),
            },
        )

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
        candidates = list(output.candidates)
        try:
            adapter = FPGenDynamicsAdapter.from_model(model)
        except BifurcationCapabilityError:
            adapter = None
        n_state = adapter.state_size if adapter is not None else int(model.n_modes) ** 2
        control_names = tuple(output.metadata.get("control_names", ()))
        vectors = np.asarray(
            [candidate.state_vector for candidate in candidates], dtype=float
        ).reshape((-1, n_state))
        if candidates and adapter is not None:
            states = np.asarray(
                [
                    adapter.state_matrix(
                        candidate.state_vector,
                        {**model.params, **candidate.controls},
                    )
                    for candidate in candidates
                ]
            )
        elif candidates:
            from qphase_cam.core.coordinates import vector_to_matrix

            states = np.asarray(vector_to_matrix(vectors, int(model.n_modes)))
        elif adapter is not None:
            states = np.empty((0, *adapter.state_matrix_shape), dtype=complex)
        else:
            states = np.empty(
                (0, int(model.n_modes), int(model.n_modes)), dtype=complex
            )
        controls = np.asarray(
            [
                [candidate.controls[name] for name in control_names]
                for candidate in candidates
            ],
            dtype=float,
        ).reshape((len(candidates), len(control_names)))
        diagnostic_names = (
            sorted(set().union(*(candidate.metadata for candidate in candidates)))
            if candidates
            else []
        )
        diagnostics = {
            name: Engine._pack_candidate_metadata(
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
            verification_digits=np.asarray(
                [
                    candidate.metadata.get("verification_digits", 0)
                    for candidate in candidates
                ],
                dtype=int,
            ),
            verification_status=np.asarray(
                [
                    candidate.metadata.get("verification_status", "not_run")
                    for candidate in candidates
                ]
            ),
            branches=Engine._pack_bifurcation_branches(
                candidates,
                adapter.state_matrix_shape
                if adapter is not None
                else (int(model.n_modes), int(model.n_modes)),
            ),
            diagnostics=diagnostics,
            meta={
                **output.metadata,
                "target": output.target,
                "order": output.order,
                "model": model.name,
                "state_layout": (
                    adapter.state_layout if adapter is not None else "canonical"
                ),
                "moment_layout": (
                    adapter.moment_layout if adapter is not None else "normal"
                ),
                "state_ids": adapter.state_ids if adapter is not None else (),
                "state_matrix_shape": (
                    adapter.state_matrix_shape
                    if adapter is not None
                    else (int(model.n_modes), int(model.n_modes))
                ),
                "fixed_params": {
                    name: value
                    for name, value in model.params.items()
                    if np.asarray(value).ndim == 0
                },
            },
        )

    @staticmethod
    def _pack_candidate_metadata(values: list[Any]) -> np.ndarray:
        """Preserve a candidate axis for scalar, fixed-shape, and ragged values."""
        if not values:
            return np.asarray([])
        if any(Engine._contains_mapping(value) for value in values):
            output = np.empty(len(values), dtype=object)
            output[:] = values
            return output
        arrays = [np.asarray(value) for value in values]
        if all(array.ndim == 0 for array in arrays):
            return np.asarray(values)
        shapes = {array.shape for array in arrays}
        if len(shapes) == 1:
            try:
                return np.stack(arrays, axis=0)
            except (TypeError, ValueError):
                pass
        output = np.empty(len(values), dtype=object)
        output[:] = values
        return output

    @staticmethod
    def _contains_mapping(value: Any) -> bool:
        if isinstance(value, Mapping):
            return True
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return any(Engine._contains_mapping(item) for item in value)
        return False

    @staticmethod
    def _pack_bifurcation_branches(
        candidates: list[Any], matrix_shape: tuple[int, int]
    ) -> CAMBifurcationBranchTable | None:
        rows: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            local_branch_index = 0
            for signature_index, signature in enumerate(
                candidate.metadata.get("scaling_signatures", ())
            ):
                branches = signature.get("branches", ())
                if not branches:
                    branches = (
                        {
                            "epsilon_side": 0,
                            "amplitude": np.nan,
                            "leading_state_coefficient": np.full(
                                matrix_shape, np.nan + 0j
                            ),
                        },
                    )
                for branch in branches:
                    rows.append(
                        {
                            "candidate_index": candidate_index,
                            "local_branch_index": local_branch_index,
                            "signature_index": signature_index,
                            "state_order": signature["state_order"],
                            "perturbation_order": signature["perturbation_order"],
                            "coupling_state_order": signature["coupling_state_order"],
                            "exponent_numerator": signature["exponent_numerator"],
                            "exponent_denominator": signature["exponent_denominator"],
                            "epsilon_side": branch["epsilon_side"],
                            "amplitude": branch["amplitude"],
                            "real_branch": bool(branch["epsilon_side"]),
                            "sublinear": signature["sublinear"],
                            "leading_state_coefficient": branch[
                                "leading_state_coefficient"
                            ],
                        }
                    )
                    local_branch_index += 1
        if not rows:
            return None
        fields = CAMBifurcationBranchTable.__dataclass_fields__
        return CAMBifurcationBranchTable(
            **{name: np.asarray([row[name] for row in rows]) for name in fields}
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
