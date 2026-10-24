"""qphase_cam: export comparison against the legacy results
---------------------------------------------------------
The importer is only trustworthy if the export can be shown to carry the same
science. These helpers compare a *stored* product with the legacy result it came
from, field by field: numerical payload, coordinates, counts, ragged offsets,
identity uniqueness, and the canonical JSON columns. A pass is not "the file
exists"; it is "every checked quantity and every checked classification row
still matches".

Both the stored and the in-memory product are accepted, because the helpers
read through ``Dataset.handle`` — a test can therefore compare the artifact a
user would load, not just the object the builder returned.

The continuation comparison also checks the legacy *empty* points: a point whose
fixed-capacity slot was never filled must stay a NaN state with ``success``
false, so an import cannot silently upgrade a failed point into a solved one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from qphase.data import Dataset

from ._common import canonical_json_text
from .candidates import split_candidate_diagnostics
from .legacy import (
    _continuation_columns,
    _legacy_candidate_arrays,
    _legacy_candidate_count,
)

if TYPE_CHECKING:
    from ..result import CAMResult

__all__ = [
    "compare_candidates_export",
    "compare_continuation_export",
    "compare_state_export",
]


def _dataset_arrays(product: Dataset) -> dict[str, np.ndarray]:
    """Materialize every variable of a runtime- or artifact-backed product."""
    return {
        variable.name: np.asarray(product.handle(variable.name).materialize())
        for variable in product.variables
    }


def _array_check(
    actual: Any, expected: Any, *, rtol: float, atol: float
) -> tuple[bool, dict[str, Any]]:
    """Compare two arrays, tolerating NaN slots and reporting the error."""
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape:
        return False, {"expected_shape": list(right.shape), "shape": list(left.shape)}
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        matched = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
        with np.errstate(invalid="ignore"):
            difference = np.abs(left - right)
        both_nan = np.isnan(left) & np.isnan(right)
        difference = np.where(both_nan, 0.0, difference)
        finite = difference[np.isfinite(difference)]
        detail: dict[str, Any] = {
            "max_abs_error": float(finite.max()) if finite.size else 0.0
        }
        if difference.size and np.any(np.isinf(difference)):
            detail["infinite_difference"] = True
        return matched, detail
    matched = bool(np.array_equal(left, right))
    return matched, {"entries": int(left.size)}


def _comparison_report(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "passed": True,
        "mismatches": [],
        "checks": {},
        "unmapped": [],
    }


def _record(
    report: dict[str, Any],
    name: str,
    matched: bool,
    detail: Mapping[str, Any] | None = None,
) -> None:
    report["checks"][name] = {"matched": bool(matched), **dict(detail or {})}
    if not matched:
        report["passed"] = False
        report["mismatches"].append(name)


def compare_state_export(
    product: Dataset,
    result: CAMResult,
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Compare one state product against the legacy result it came from.

    Checks the numerical payload, the parameter/axis coordinates, the counts
    and the invariant that ``valid_mask`` agrees with ``solution_count`` -
    not merely that the arrays have the right names.
    """
    arrays = _dataset_arrays(product)
    shape = tuple(int(value) for value in product.attributes["grid_shape"])
    scan_size = int(product.attributes["scan_size"])
    capacity = int(product.attributes["solution_capacity"])
    report = _comparison_report("state")

    def restore(name: str, trailing: tuple[int, ...]) -> np.ndarray:
        return arrays[name].reshape(shape + (capacity, *trailing))

    for name, expected in (
        ("states", np.asarray(result.states)),
        ("residuals", np.asarray(result.residuals)),
        ("success", np.asarray(result.success)),
        ("valid_mask", np.asarray(result.valid_mask)),
    ):
        matched, detail = _array_check(
            restore(name, expected.shape[len(shape) + 1 :]),
            expected,
            rtol=rtol,
            atol=atol,
        )
        _record(report, name, matched, detail)
    matched, detail = _array_check(
        arrays["solution_count"].reshape(shape),
        np.asarray(result.solution_count),
        rtol=rtol,
        atol=atol,
    )
    _record(report, "solution_count", matched, detail)

    valid = arrays["valid_mask"]
    _record(
        report,
        "valid_mask_matches_solution_count",
        bool(np.array_equal(valid.sum(axis=-1), arrays["solution_count"])),
        {"scan_size": scan_size},
    )

    for prefix, mapping in (
        ("parameter.", result.params),
        ("axis.", result.axes),
    ):
        for name, value in mapping.items():
            column = arrays.get(f"{prefix}{name}")
            if column is None:
                report["unmapped"].append(f"{prefix}{name}")
                continue
            array = np.asarray(value)
            flattened = (
                np.full(scan_size, array.item())
                if array.ndim == 0
                else array.reshape(-1)
            )
            matched, detail = _array_check(column, flattened, rtol=rtol, atol=atol)
            _record(report, f"{prefix}{name}", matched, detail)
    return report


def compare_candidates_export(
    product: Dataset,
    result: Any,
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Compare one candidate product against its legacy result.

    Both the single candidate result and the ragged scan are accepted; the
    legacy side is normalized exactly as the importer normalizes it, so the
    comparison exercises the persisted columns (counts, coordinates, offsets,
    classification and audit rows) rather than the in-memory objects.
    """
    from ..bifurcation_result import (
        CAMBifurcationResult,
        CAMBifurcationScanResult,
    )

    arrays = _dataset_arrays(product)
    report = _comparison_report("candidates")
    if isinstance(result, CAMBifurcationScanResult):
        candidates = result.candidates
        expected_offsets = np.asarray(result.candidate_offsets, dtype=np.int64)
        case_axes = result.case_axes
        case_params = result.case_params
        case_metadata = list(result.case_metadata)
    elif isinstance(result, CAMBifurcationResult):
        candidates = result
        expected_offsets = np.asarray(
            [0, len(np.asarray(result.states))], dtype=np.int64
        )
        case_axes = {}
        case_params = {}
        case_metadata = [{}]
    else:
        raise TypeError(
            "expected a CAMBifurcationResult or a CAMBifurcationScanResult, "
            f"got {type(result).__name__}"
        )
    expected = _legacy_candidate_arrays(candidates)
    n_candidate = _legacy_candidate_count(candidates)

    matched, detail = _array_check(
        arrays["candidate_offsets"], expected_offsets, rtol=rtol, atol=atol
    )
    _record(report, "candidate_offsets", matched, detail)
    _record(
        report,
        "candidate_count",
        len(arrays["candidate_identity"]) == n_candidate,
        {
            "expected": n_candidate,
            "candidate": int(len(arrays["candidate_identity"])),
            "per_case": np.diff(expected_offsets).tolist(),
        },
    )
    identity = [str(value) for value in arrays["candidate_identity"]]
    _record(
        report,
        "candidate_identity",
        bool(
            len(identity) == n_candidate
            and all(value.strip() for value in identity)
            and len(set(identity)) == len(identity)
        ),
        {"entries": len(identity), "unique": len(set(identity))},
    )

    for name in (
        "states",
        "state_vectors",
        "control_values",
        "control_names",
        "full_residual_norm",
        "search_residual_norm",
        "success",
        "verification_digits",
    ):
        matched, detail = _array_check(
            arrays[name], expected[name], rtol=rtol, atol=atol
        )
        _record(report, name, matched, detail)
    for name in ("status", "method", "verification_status"):
        matched, detail = _array_check(
            arrays[name], expected[name], rtol=rtol, atol=atol
        )
        _record(report, name, matched, detail)

    classification, base_audit, _unmapped = split_candidate_diagnostics(
        candidates.diagnostics, n_candidate
    )
    for name, expected_rows in (
        ("classification", classification),
        ("base_audit", base_audit),
    ):
        actual_rows = [json.loads(str(value)) for value in arrays[name]]
        expected_text = [canonical_json_text(row) for row in expected_rows]
        actual_text = [canonical_json_text(row) for row in actual_rows]
        _record(
            report,
            name,
            actual_text == expected_text,
            {"keys": sorted({key for row in expected_rows for key in row})},
        )

    actual_case_metadata = [json.loads(str(value)) for value in arrays["case_metadata"]]
    _record(
        report,
        "case_metadata",
        [canonical_json_text(row) for row in actual_case_metadata]
        == [canonical_json_text(row) for row in case_metadata],
        {"cases": len(actual_case_metadata)},
    )

    for prefix, mapping in (
        ("parameter.", case_axes),
        ("case_parameter.", case_params),
    ):
        for name, value in mapping.items():
            column = arrays.get(f"{prefix}{name}")
            if column is None:
                report["unmapped"].append(f"{prefix}{name}")
                continue
            matched, detail = _array_check(
                column, np.asarray(value).reshape(-1), rtol=rtol, atol=atol
            )
            _record(report, f"{prefix}{name}", matched, detail)
    return report


def compare_continuation_export(
    product: Dataset,
    result: Any,
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Compare one continuation product against the legacy run it came from.

    The legacy side is normalized exactly as the importer normalizes it (filled
    slot per point, swept parameter as the coordinate), so the comparison
    exercises the persisted columns rather than the in-memory object.
    """
    from ..result import CAMResult

    if not isinstance(result, CAMResult):
        raise TypeError(f"expected a CAMResult, got {type(result).__name__}")
    coordinate, states, residuals, success, name, counts = _continuation_columns(result)
    arrays = _dataset_arrays(product)
    report = _comparison_report("continuation")

    for field, expected in (
        ("states", states),
        ("residuals", residuals),
        ("success", success),
        ("continuation_coordinate", coordinate),
    ):
        matched, detail = _array_check(arrays[field], expected, rtol=rtol, atol=atol)
        _record(report, field, matched, detail)

    offsets = np.asarray([0, coordinate.size], dtype=np.int64)
    matched, detail = _array_check(
        arrays["path_offsets"], offsets, rtol=rtol, atol=atol
    )
    _record(report, "path_offsets", matched, detail)
    identity = [str(value) for value in arrays["path_identity"]]
    _record(
        report,
        "path_identity",
        len(identity) == 1 and bool(identity[0].strip()),
        {"entries": len(identity)},
    )
    _record(
        report,
        "empty_points",
        int(np.count_nonzero(~np.any(np.isfinite(states.real), axis=(1, 2))))
        == int(np.count_nonzero(counts == 0)),
        {"legacy_empty": int(np.count_nonzero(counts == 0))},
    )
    events = [json.loads(str(value)) for value in arrays["events"]]
    _record(report, "events", events == [{} for _ in range(coordinate.size)])

    for key, value in (result.params or {}).items():
        column = arrays.get(f"parameter.{key}")
        if column is None:
            report["unmapped"].append(f"parameter.{key}")
            continue
        matched, detail = _array_check(
            column,
            np.broadcast_to(np.asarray(value), column.shape),
            rtol=rtol,
            atol=atol,
        )
        _record(report, f"parameter.{key}", matched, detail)
    return report
