"""qphase_cam: the ragged candidate product builder
---------------------------------------------------------
Bifurcation/search candidates as one ragged table: a ``scan`` axis over the
flattened case grid, a ``group``/``group_offset`` pairing and a
``candidate_offsets`` block of length ``n_case + 1``. Empty cases and a scan
with no candidate at all are first-class: the offsets say so, so nothing has to
be inferred from a zero length.

Three invariants are enforced here rather than documented and hoped for:

* ``candidate_identity`` is explicit, non-empty and **unique within one case**
  (the block ``candidate_offsets`` delimits). It names the physical candidate,
  not a slot: the same candidate found by two cases of one scan may appear twice
  with the same identity, which is why uniqueness is scoped per case. The builder
  refuses to invent an identity from slot positions. The identity is solver-owned
  and content-derived (``qphase_cam.solver.identity``); the legacy
  ``legacy:<fingerprint>:<case>:<slot>`` form exists only in the one-off importer
  and describes an imported snapshot.
* Deterministic ``candidate_offsets``: start at zero, monotonic non-decreasing,
  end at the candidate count, and cover the declared ``case_shape``.
* JSON reaches the product only as canonical text: ``classification``,
  ``base_audit`` and ``case_metadata`` are ``U`` columns, never object arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qphase.data import Dataset, StatisticsDataset

from ._common import (
    closed_product,
    coordinate_columns,
    json_text_column,
    json_value,
    string_column,
    string_widths,
)
from .declarations import CAM_CANDIDATE_PRODUCT, _candidate_variables

__all__ = [
    "build_candidate_product",
    "is_classification_diagnostic",
    "split_candidate_diagnostics",
]


def is_classification_diagnostic(name: str) -> bool:
    lowered = name.lower()
    return "signature" in lowered or "classif" in lowered


def split_candidate_diagnostics(
    diagnostics: Mapping[str, Any] | None, n_candidate: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Split per-candidate diagnostics into classification and audit rows.

    Keys whose name mentions a scaling signature or a classification decision
    go to ``classification``; every other per-candidate key goes to
    ``base_audit``. Values are converted recursively, so a legacy object array
    of tuples/dicts — how the solver stores ``scaling_signatures`` — becomes
    JSON instead of being refused. Only a *value* that has no JSON
    representation is reported, as ``<key>[<candidate>]``, and a key that
    cannot be expressed per candidate at all is reported by name.
    """
    classification: list[dict[str, Any]] = [dict() for _ in range(n_candidate)]
    base_audit: list[dict[str, Any]] = [dict() for _ in range(n_candidate)]
    unmapped: list[str] = []
    for raw_name, value in (diagnostics or {}).items():
        name = str(raw_name)
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != n_candidate:
            unmapped.append(name)
            continue
        rows = classification if is_classification_diagnostic(name) else base_audit
        for position in range(n_candidate):
            try:
                rows[position][name] = json_value(array[position])
            except (TypeError, ValueError):
                unmapped.append(f"{name}[{position}]")
    return classification, base_audit, sorted(unmapped)


def _candidate_states(values: Any, n_candidate: int) -> np.ndarray:
    """Return candidate states as ``(candidate, state_row, state_col)``."""
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 1 and array.size == 0 and n_candidate == 0:
        return np.zeros((0, 0, 0), dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError(
            "states must span (candidate, state_row, state_col); got shape "
            f"{array.shape}"
        )
    if array.shape[0] != n_candidate:
        raise ValueError(
            f"states has {array.shape[0]} rows for {n_candidate} candidates"
        )
    return array


def _candidate_matrix(
    values: Any, n_candidate: int, name: str, dtype: Any
) -> np.ndarray:
    """Return a two-dimensional per-candidate column block."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1 and array.size == 0 and n_candidate == 0:
        return np.zeros((0, 0), dtype=dtype)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional over the candidate axis; got "
            f"shape {array.shape}"
        )
    if array.shape[0] != n_candidate:
        raise ValueError(
            f"{name} has {array.shape[0]} rows for {n_candidate} candidates"
        )
    return array


def _candidate_column(
    values: Any, n_candidate: int, name: str, dtype: Any
) -> np.ndarray:
    """Return one value per candidate."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.shape[0] != n_candidate:
        raise ValueError(
            f"{name} must have one entry per candidate; got shape {array.shape}"
        )
    return array


def _candidate_text_column(
    values: Sequence[str], n_candidate: int, name: str
) -> np.ndarray:
    column = string_column(list(values))
    if column.shape[0] != n_candidate:
        raise ValueError(
            f"{name} has {column.shape[0]} entries for {n_candidate} candidates"
        )
    return column


def _json_rows(rows: Sequence[Any] | None, count: int, name: str) -> list[Any]:
    if rows is None:
        return [{} for _ in range(count)]
    items = list(rows)
    if len(items) != count:
        raise ValueError(f"{name} has {len(items)} entries for {count} rows")
    return items


def build_candidate_product(
    *,
    states: Any,
    state_vectors: Any,
    control_values: Any,
    control_names: Sequence[str] | None,
    full_residual_norm: Any,
    search_residual_norm: Any,
    success: Any,
    status: Sequence[str],
    method: Sequence[str],
    verification_digits: Any,
    verification_status: Sequence[str],
    candidate_identity: Sequence[str],
    classification: Sequence[Any] | None = None,
    base_audit: Sequence[Any] | None = None,
    case_metadata: Sequence[Any] | None = None,
    candidate_offsets: Any | None = None,
    group_scan_index: Any | None = None,
    case_axes: Mapping[str, Any] | None = None,
    case_params: Mapping[str, Any] | None = None,
    case_shape: Sequence[int] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build the ragged candidate product of one case or one case scan.

    ``candidate_identity`` is required: identity is the solver's to own, and a
    builder that invented one from slot positions would recreate exactly the
    slot-as-branch-id confusion this product exists to remove. Pass ``[]`` for
    an empty candidate list.

    ``classification``, ``base_audit`` and ``case_metadata`` accept mappings,
    lists or canonical JSON text; each entry is re-serialized canonically, so
    a JSON text column is always valid JSON. Defaults are ``{}``.

    ``candidate_offsets`` defaults to ``[0, n_candidate]`` (one case). When it
    is given it must have ``n_case + 1`` entries, start at zero, be monotonic
    non-decreasing and end at the candidate count; empty cases are otherwise
    indistinguishable from missing ones.

    Raises
    ------
    ValueError
        If an identity is empty or duplicated, an offset block is
        inconsistent, or a column does not cover the candidate axis.

    """
    identity = [str(value) for value in candidate_identity]
    n_candidate = len(identity)
    for position, value in enumerate(identity):
        if not value.strip():
            raise ValueError(
                f"candidate_identity[{position}] is empty; every candidate "
                "needs an explicit identity"
            )
    if candidate_offsets is None:
        offsets = np.asarray([0, n_candidate], dtype=np.int64)
    else:
        offsets = np.asarray(candidate_offsets, dtype=np.int64)
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError(
                "candidate_offsets must be one-dimensional with n_case + 1 "
                f"entries; got shape {offsets.shape}"
            )
        if int(offsets[0]) != 0:
            raise ValueError("candidate_offsets must start at 0")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("candidate_offsets must be monotonic non-decreasing")
        if int(offsets[-1]) != n_candidate:
            raise ValueError(
                f"candidate_offsets must end at the candidate count "
                f"{n_candidate}; got {int(offsets[-1])}"
            )
    n_case = int(offsets.size) - 1
    # A candidate identity names the physical candidate, not its position: the
    # same candidate found by two search cases may legitimately appear twice in
    # one product, so uniqueness is required inside a case, not across cases.
    for case in range(n_case):
        start, stop = int(offsets[case]), int(offsets[case + 1])
        block = identity[start:stop]
        duplicates = sorted({value for value in block if block.count(value) > 1})
        if duplicates:
            raise ValueError(
                f"candidate_identity must be unique within one case; case {case} "
                f"duplicates {duplicates}"
            )
    case_shape_list = [int(value) for value in (case_shape or ())] or [n_case]
    if int(np.prod(case_shape_list, dtype=np.int64)) != n_case:
        raise ValueError(
            f"case_shape {case_shape_list} does not cover the {n_case} scan cases"
        )

    group_index = (
        np.arange(n_case, dtype=np.int64)
        if group_scan_index is None
        else np.asarray(group_scan_index, dtype=np.int64)
    )
    if group_index.shape != (n_case,):
        raise ValueError(
            f"group_scan_index must have shape ({n_case},); got {group_index.shape}"
        )
    if n_case and (int(group_index.min()) < 0 or int(group_index.max()) >= n_case):
        raise ValueError("group_scan_index must index the scan axis")

    state_array = _candidate_states(states, n_candidate)
    vector_array = _candidate_matrix(
        state_vectors, n_candidate, "state_vectors", np.float64
    )
    control_array = _candidate_matrix(
        control_values, n_candidate, "control_values", np.float64
    )
    names = [str(value) for value in (control_names or ())]
    if control_array.shape[1] != len(names):
        raise ValueError(
            f"control_values has {control_array.shape[1]} columns for "
            f"{len(names)} control_names"
        )

    reserved: set[str] = set()
    axis_arrays, axis_variables, coordinates, unmapped_case_axes = coordinate_columns(
        case_axes,
        prefix="parameter.",
        shape=tuple(case_shape_list),
        count=n_case,
        reserved=reserved,
    )
    # ``case_parameter.*`` columns describe a case, not a parameter axis, so
    # they stay out of ``coordinates`` and out of the coordinate namespace.
    parameter_arrays, parameter_variables, _, unmapped_case_params = coordinate_columns(
        case_params,
        prefix="case_parameter.",
        shape=(n_case,),
        count=n_case,
        reserved=set(),
    )

    arrays: dict[str, np.ndarray] = {
        "group_scan_index": group_index,
        "candidate_offsets": offsets,
        "candidate_identity": string_column(identity),
        "states": state_array,
        "state_vectors": vector_array,
        "control_values": control_array,
        "control_names": string_column(names),
        "full_residual_norm": _candidate_column(
            full_residual_norm, n_candidate, "full_residual_norm", np.float64
        ),
        "search_residual_norm": _candidate_column(
            search_residual_norm, n_candidate, "search_residual_norm", np.float64
        ),
        "success": _candidate_column(success, n_candidate, "success", bool),
        "status": _candidate_text_column(status, n_candidate, "status"),
        "method": _candidate_text_column(method, n_candidate, "method"),
        "verification_digits": _candidate_column(
            verification_digits, n_candidate, "verification_digits", np.int64
        ),
        "verification_status": _candidate_text_column(
            verification_status, n_candidate, "verification_status"
        ),
        "classification": json_text_column(
            _json_rows(classification, n_candidate, "classification")
        ),
        "base_audit": json_text_column(
            _json_rows(base_audit, n_candidate, "base_audit")
        ),
        "case_metadata": json_text_column(
            _json_rows(case_metadata, n_case, "case_metadata")
        ),
        **axis_arrays,
        **parameter_arrays,
    }
    variables = [
        *_candidate_variables(string_widths(arrays)),
        *axis_variables,
        *parameter_variables,
    ]
    attributes = {
        **CAM_CANDIDATE_PRODUCT.attributes,
        "case_shape": case_shape_list,
        "case_count": n_case,
        "candidate_count": n_candidate,
    }
    schema = closed_product(
        CAM_CANDIDATE_PRODUCT,
        sizes={
            "scan": n_case,
            "group": n_case,
            "group_offset": n_case + 1,
            "candidate": n_candidate,
            "state_row": int(state_array.shape[1]),
            "state_col": int(state_array.shape[2]),
            "state_vector": int(vector_array.shape[1]),
            "control": int(control_array.shape[1]),
        },
        variables=variables,
        coordinates=coordinates,
        attributes=attributes,
    )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.cam",
        provenance={
            **dict(provenance or {}),
            "unmapped_case_axis_names": unmapped_case_axes,
            "unmapped_case_parameter_names": unmapped_case_params,
        },
    )
