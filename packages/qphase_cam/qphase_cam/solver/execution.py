"""qphase_cam: shared solver execution helpers
---------------------------------------------------------
The pieces every solver needs once it owns its own execution: applying a
``ScanSpec`` grid to the model, running the per-point/per-case loop with the
configured failure policy, and packing the numeric output into typed products.
They used to live inside the engine; they live here so a solver can answer
:class:`~qphase_cam.solver.base.CAMSolveRequest` without the engine reaching into
its numbers.

Nothing here changes a numeric decision: the sort key, the solution capacity
check, the candidate state reconstruction and the diagnostic split are the ones
the engine already used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.core.scan import ParameterGrid, execute_pointwise

from ..data import build_candidate_product, build_state_product
from ..data.candidates import split_candidate_diagnostics
from ..errors import BifurcationCapabilityError, SolutionCapacityError
from ..state import CAMSolution
from .base import CAMProductFragment

__all__ = [
    "CaseOutcome",
    "build_candidate_from_cases",
    "candidate_case_arrays",
    "candidate_product_from_output",
    "case_metadata",
    "grid_model_parameters",
    "model_parameters",
    "run_cases",
    "state_fragment",
]


@dataclass
class CaseOutcome:
    """One case of a scan: either a result or the error the policy recorded."""

    flat_index: int
    index: tuple[int, ...]
    values: dict[str, Any]
    result: Any = None
    error: Exception | None = None

    @property
    def failed(self) -> bool:
        """Whether the case raised."""
        return self.error is not None


def model_parameters(model: Any, params: dict[str, Any]) -> None:
    """Replace the model parameters in place."""
    if hasattr(model, "_params"):
        model._params = params
        return
    current = getattr(model, "params", None)
    if isinstance(current, dict):
        current.clear()
        current.update(params)
        return
    raise TypeError(f"model {model.name!r} does not expose mutable parameters")


@contextmanager
def restored_parameters(model: Any, params: dict[str, Any]) -> Iterator[None]:
    """Apply ``params`` and restore the previous parameters afterwards."""
    previous = dict(getattr(model, "params", {}))
    model_parameters(model, dict(params))
    try:
        yield
    finally:
        model_parameters(model, previous)


def grid_model_parameters(
    model: Any, grid: ParameterGrid, *, forbidden: Sequence[str] = ()
) -> dict[str, Any]:
    """Return the flattened per-point parameters a grid assigns to the model.

    A grid axis must target ``model.<name>.<parameter>``; a parameter that is
    also a solver control cannot be swept, because the control is what the solver
    searches along.
    """
    expected_prefix = f"model.{model.name}."
    output: dict[str, Any] = {}
    blocked = set(forbidden)
    for target, values in grid.target_arrays(flatten=True).items():
        if not target.startswith(expected_prefix):
            raise ValueError(
                f"CAM scan target {target!r} must target {expected_prefix}<parameter>"
            )
        parameter = target.removeprefix(expected_prefix)
        if "." in parameter or parameter not in model.params:
            raise ValueError(f"unknown CAM model scan target {target!r}")
        if parameter in blocked:
            raise ValueError(
                "ScanSpec targets must be fixed case parameters, not solver "
                f"controls: {[parameter]}"
            )
        output[parameter] = values
    return output


def run_cases(
    grid: ParameterGrid,
    *,
    model: Any,
    base_params: Mapping[str, Any],
    call: Callable[[Any, Any], Any],
    context: Any | None,
    policy: str,
) -> list[CaseOutcome]:
    """Run one call per grid point, honouring the case failure policy."""

    def run_point(point: Any) -> CaseOutcome:
        params = dict(base_params)
        for target, value in point.targets.items():
            params[target.rsplit(".", 1)[-1]] = value
        model_parameters(model, params)
        try:
            result = call(model, point)
        except MemoryError:
            raise
        except Exception as exc:
            if policy == "abort":
                raise
            return CaseOutcome(
                flat_index=int(point.flat_index),
                index=tuple(int(value) for value in point.index),
                values=dict(point.values),
                error=exc,
            )
        return CaseOutcome(
            flat_index=int(point.flat_index),
            index=tuple(int(value) for value in point.index),
            values=dict(point.values),
            result=result,
        )

    return list(execute_pointwise(grid, run_point, context=context))


def _solution_rows(
    rows: Sequence[Sequence[Any]],
    *,
    grid_shape: tuple[int, ...],
    capacity: int,
    n_modes: int,
    sort_key: Callable[[Any], Any] | None,
) -> dict[str, Any]:
    """Pack per-point solution rows into the state product arrays."""
    states = np.full(grid_shape + (capacity, n_modes, n_modes), np.nan + 1j * np.nan)
    residuals = np.full(grid_shape + (capacity,), np.nan)
    success = np.zeros(grid_shape + (capacity,), dtype=bool)
    valid = np.zeros(grid_shape + (capacity,), dtype=bool)
    counts = np.zeros(grid_shape or (), dtype=np.int64)
    for position, row in enumerate(rows):
        index = (
            tuple(int(value) for value in np.unravel_index(position, grid_shape))
            if grid_shape
            else ()
        )
        solutions = list(row)
        if len(solutions) > capacity:
            raise SolutionCapacityError(
                f"model solution capacity {capacity} exceeded by {len(solutions)} "
                "solutions"
            )
        if sort_key is not None:
            solutions.sort(key=sort_key)
        counts[index] = len(solutions)
        for slot, solution in enumerate(solutions):
            states[index + (slot,)] = solution.state
            residuals[index + (slot,)] = solution.residual
            success[index + (slot,)] = solution.success
            valid[index + (slot,)] = True
    return {
        "states": states,
        "residuals": residuals,
        "success": success,
        "valid_mask": valid,
        "solution_count": counts,
    }


def state_fragment(
    *,
    request: Any,
    run: Callable[[Any, Any | None], Any],
    supports_batch: bool,
) -> CAMProductFragment:
    """Run one state solve or state scan and pack the typed product.

    ``run(model, point)`` returns a ``CAMSolverOutput``; ``point`` is ``None``
    for a single call (a single point or one batch covering the whole grid).
    """
    model = request.model
    grid = request.parameter_grid
    base_params = dict(model.params)
    capacity = int(model.steady_state_capacity)
    n_modes = int(model.n_modes)

    def sort_key(solution: Any) -> Any:
        return model.cam_solution_sort_key(solution.state, model.params)

    metadata: dict[str, Any] = {}
    sorted_rows = False
    product_params: dict[str, Any] | None = None

    if grid is None:
        output = run(model, None)
        raw = list(output.solutions)
        if raw and not isinstance(raw[0], CAMSolution):
            # A batch solver reports one row per point even without a ScanSpec;
            # the rows are the scan, exactly as the pre-10D2 packing read them.
            rows = [list(row) for row in raw]
            grid_shape: tuple[int, ...] = (len(rows),)
        else:
            rows = [raw]
            grid_shape = ()
        axes = dict(output.axes)
        metadata.update(output.metadata)
    elif supports_batch:
        flattened = grid_model_parameters(model, grid)
        product_params = {**base_params, **flattened}
        with restored_parameters(model, product_params):
            output = run(model, None)
        rows = [list(row) for row in output.solutions]
        grid_shape = tuple(grid.shape)
        axes = dict(grid.axes)
        metadata.update(output.metadata)
        metadata.update({"scan_shape": grid_shape, "scan_combine": grid.combine})
    else:

        def run_point(current: Any, point: Any) -> list[Any]:
            # Sort while the case parameters are still the model's, so the slot
            # order never depends on a later scan-wide parameter array.
            row = list(run(current, point).solutions)
            row.sort(key=sort_key)
            return row

        cases = run_cases(
            grid,
            model=model,
            base_params=base_params,
            call=run_point,
            context=request.context,
            policy=request.case_failure_policy,
        )
        model_parameters(model, {**base_params, **grid_model_parameters(model, grid)})
        rows = [list(case.result) if not case.failed else [] for case in cases]
        failed = [case for case in cases if case.failed]
        if failed:
            metadata["failed_cases"] = [
                {
                    "case_flat_index": case.flat_index,
                    "case_index": list(case.index),
                    "case_error_type": type(case.error).__name__,
                    "case_error_message": str(case.error),
                }
                for case in failed
            ]
        sorted_rows = True
        grid_shape = tuple(grid.shape)
        axes = dict(grid.axes)
        metadata.update({"scan_shape": grid_shape, "scan_combine": grid.combine})

    payload = _solution_rows(
        rows,
        grid_shape=grid_shape,
        capacity=capacity,
        n_modes=n_modes,
        sort_key=None if sorted_rows else sort_key,
    )
    parameters = (
        dict(product_params)
        if product_params is not None
        else dict(getattr(model, "params", {}))
    )
    if grid is not None:
        for name, values in list(parameters.items()):
            array = np.asarray(values)
            if array.ndim > 0 and array.size == grid.size:
                parameters[name] = array.reshape(grid.shape)
    product = build_state_product(
        **payload,
        parameters=parameters,
        axes=axes,
        grid_shape=grid_shape,
        provenance={"solver": request.task},
    )
    role = "state" if product.attributes["product"] == "state" else "state_scan"
    return CAMProductFragment(products={role: product}, metadata=metadata)


#: Diagnostics a candidate carries that are paths, not search metadata.
def case_metadata(
    *,
    case_identity: str,
    search_identity: str,
    outcome: CaseOutcome,
    output: Any | None,
) -> dict[str, Any]:
    """Return the per-case metadata recorded on the candidate product."""
    payload: dict[str, Any] = {
        "case_identity": case_identity,
        "search_identity": search_identity,
        "case_flat_index": int(outcome.flat_index),
        "case_index": list(outcome.index),
        "case_parameters": dict(outcome.values),
        "case_status": "error" if outcome.failed else "complete",
    }
    if outcome.failed:
        payload["case_error_type"] = type(outcome.error).__name__
        payload["case_error_message"] = str(outcome.error)
    elif output is not None:
        for name in ("strategy", "coverage", "target", "order"):
            if name in output.metadata:
                payload[name] = output.metadata[name]
    return payload


def candidate_case_arrays(
    model: Any, output: Any, *, case_axes: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the builder keyword arrays of one candidate case.

    This is the packing the engine used for a ``CAMBifurcationResult``, expressed
    as candidate-product columns: identity, diagnostics split into
    classification/audit JSON, and the typed numeric columns.
    """
    candidates = list(output.candidates)
    try:
        from ..core.qpsgen import QPSGenDynamicsAdapter

        adapter = QPSGenDynamicsAdapter.from_model(model)
    except BifurcationCapabilityError:
        adapter = None
    n_state = adapter.state_size if adapter is not None else int(model.n_modes) ** 2
    control_names = tuple(
        str(name) for name in output.metadata.get("control_names", ())
    )
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
        from ..core.coordinates import vector_to_matrix

        states = np.asarray(vector_to_matrix(vectors, int(model.n_modes)))
    elif adapter is not None:
        states = np.empty((0, *adapter.state_matrix_shape), dtype=complex)
    else:
        states = np.empty((0, int(model.n_modes), int(model.n_modes)), dtype=complex)
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
        name: _candidate_diagnostic(
            [candidate.metadata.get(name, np.nan) for candidate in candidates]
        )
        for name in diagnostic_names
    }
    classification, base_audit, unmapped = split_candidate_diagnostics(
        diagnostics, len(candidates)
    )
    return {
        "states": states,
        "state_vectors": vectors,
        "control_values": controls,
        "control_names": control_names,
        "full_residual_norm": np.asarray(
            [candidate.full_residual_norm for candidate in candidates]
        ),
        "search_residual_norm": np.asarray(
            [candidate.search_residual_norm for candidate in candidates]
        ),
        "success": np.asarray([candidate.success for candidate in candidates]),
        "status": [str(candidate.status) for candidate in candidates],
        "method": [str(candidate.method) for candidate in candidates],
        "verification_digits": np.asarray(
            [
                candidate.metadata.get("verification_digits", 0)
                for candidate in candidates
            ],
            dtype=np.int64,
        ),
        "verification_status": [
            str(candidate.metadata.get("verification_status", "not_run"))
            for candidate in candidates
        ],
        "candidate_identity": [
            str(candidate.metadata["candidate_identity"]) for candidate in candidates
        ],
        "classification": classification,
        "base_audit": base_audit,
        "unmapped_diagnostics": unmapped,
        "case_axes": dict(case_axes or {}),
    }


def _candidate_diagnostic(values: list[Any]) -> np.ndarray:
    """Preserve a candidate axis for scalar, fixed-shape and ragged values."""
    if not values:
        return np.asarray([])
    if any(_contains_mapping(value) for value in values):
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


def _contains_mapping(value: Any) -> bool:
    if isinstance(value, Mapping):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_mapping(item) for item in value)
    return False


def build_candidate_from_cases(
    *,
    case_arrays: Sequence[dict[str, Any]],
    case_meta: Sequence[Mapping[str, Any]],
    case_axes: Mapping[str, Any],
    case_params: Mapping[str, Any],
    case_shape: Sequence[int],
) -> Any:
    """Merge per-case candidate arrays into one ragged candidate product."""
    counts = [len(arrays["candidate_identity"]) for arrays in case_arrays]
    offsets = np.zeros(len(case_arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    merged: dict[str, Any] = {}
    control_names: tuple[str, ...] = ()
    concatenated = (
        "states",
        "state_vectors",
        "control_values",
        "full_residual_norm",
        "search_residual_norm",
        "success",
        "status",
        "method",
        "verification_digits",
        "verification_status",
        "candidate_identity",
    )
    for name in concatenated:
        parts = [arrays[name] for arrays in case_arrays]
        if name in {"status", "method", "verification_status", "candidate_identity"}:
            merged[name] = [item for part in parts for item in part]
        elif name == "control_values":
            merged[name] = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0))
        elif name in {"states"}:
            merged[name] = np.concatenate(parts, axis=0)
        else:
            merged[name] = np.concatenate(parts, axis=0)
    for arrays in case_arrays:
        if arrays["control_names"]:
            control_names = tuple(arrays["control_names"])
            break
    merged["control_names"] = control_names
    merged["classification"] = [
        row for arrays in case_arrays for row in arrays["classification"]
    ]
    merged["base_audit"] = [
        row for arrays in case_arrays for row in arrays["base_audit"]
    ]
    merged["case_metadata"] = (
        list(case_meta) if case_meta else [{} for _ in case_arrays]
    )
    merged["candidate_offsets"] = offsets
    merged["group_scan_index"] = np.arange(len(case_arrays), dtype=np.int64)
    merged["case_axes"] = dict(case_axes)
    merged["case_params"] = dict(case_params)
    merged["case_shape"] = list(case_shape)
    return build_candidate_product(**merged)


def candidate_product_from_output(model: Any, output: Any) -> Any:
    """Pack the candidate output of one search case into the typed product.

    One case, no case axes: this is the product shape a single search produces,
    and the shape a caller uses when it wants the typed product of one solver
    output without running a scan.
    """
    return build_candidate_from_cases(
        case_arrays=[candidate_case_arrays(model, output)],
        case_meta=[],
        case_axes={},
        case_params={},
        case_shape=(),
    )
