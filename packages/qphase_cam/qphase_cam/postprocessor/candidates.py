"""Typed views of a candidate product for candidate-level annotations.

The candidate product stores its classification decision and its audit rows as
canonical JSON columns. A candidate annotation such as ``local_response`` or
``stochastic_validity`` needs the scaling-branch structure that lives inside those
columns, so this module turns the product into explicit typed rows:

* :class:`CandidateRow` — one candidate: identity, state, controls, residuals,
  status and the parsed classification/audit mappings.
* :class:`CandidateBranchTable` — the scaling branches of those candidates, with
  the same field names the solver's branch table used, derived from
  ``classification["scaling_signatures"]``.
* :class:`CandidateView` — both, plus the per-case parameters, so an annotation
  can ask for the parameters of the candidate's case.

These are read-only views over one product for the duration of one annotation
call. They are never persisted, never handed to a solver and never replace a
product: the branch structure reaches the artifact as typed annotation fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.data import Dataset

__all__ = [
    "CandidateBranchTable",
    "CandidateRow",
    "CandidateView",
    "branch_axis_size",
    "candidate_view",
    "complex_from_json",
    "matrix_from_json",
    "pivot_branch_rows",
]

_BRANCH_FIELDS: tuple[str, ...] = (
    "candidate_index",
    "local_branch_index",
    "signature_index",
    "state_order",
    "perturbation_order",
    "coupling_state_order",
    "exponent_numerator",
    "exponent_denominator",
    "epsilon_side",
    "amplitude",
    "real_branch",
    "sublinear",
    "leading_state_coefficient",
)


@dataclass(frozen=True)
class CandidateRow:
    """One candidate of a candidate product."""

    index: int
    identity: str
    state: np.ndarray
    state_vector: np.ndarray
    controls: dict[str, float]
    full_residual_norm: float
    search_residual_norm: float
    success: bool
    status: str
    method: str
    verification_digits: int
    verification_status: str
    classification: dict[str, Any]
    base_audit: dict[str, Any]


@dataclass(frozen=True)
class CandidateBranchTable:
    """The scaling branches of a candidate set, as typed arrays.

    ``size`` branches with the solver's field names; a signature without an
    explicit branch list still contributes one placeholder branch (the same shape
    the solver's branch table produced), so an annotation sees the same rows it
    always did.
    """

    candidate_index: np.ndarray
    local_branch_index: np.ndarray
    signature_index: np.ndarray
    state_order: np.ndarray
    perturbation_order: np.ndarray
    coupling_state_order: np.ndarray
    exponent_numerator: np.ndarray
    exponent_denominator: np.ndarray
    epsilon_side: np.ndarray
    amplitude: np.ndarray
    real_branch: np.ndarray
    sublinear: np.ndarray
    leading_state_coefficient: np.ndarray

    @property
    def size(self) -> int:
        """Number of branches."""
        return int(len(self.candidate_index))


@dataclass(frozen=True)
class CandidateView:
    """Typed view of one candidate product."""

    rows: list[CandidateRow]
    branches: CandidateBranchTable
    meta: dict[str, Any]
    diagnostics: dict[str, Any]
    state_vectors: np.ndarray
    case_params: dict[str, np.ndarray]
    candidate_offsets: np.ndarray
    fixed_params: dict[str, Any]

    def params_at(self, candidate_index: int) -> dict[str, Any]:
        """Return the parameters of the case one candidate belongs to."""
        params = dict(self.fixed_params)
        if self.candidate_offsets.size:
            case = int(
                np.searchsorted(self.candidate_offsets, candidate_index, side="right")
                - 1
            )
            for name, values in self.case_params.items():
                flat = np.asarray(values).reshape(-1)
                if case < flat.size:
                    params[name] = flat[case]
        params.update(self.rows[candidate_index].controls)
        return params

    def branches_of(self, candidate_index: int) -> list[int]:
        """Return the real branch indices of one candidate."""
        return [
            index
            for index in range(self.branches.size)
            if int(self.branches.candidate_index[index]) == candidate_index
            and bool(self.branches.real_branch[index])
        ]


def complex_from_json(value: Any) -> complex:
    """Return a complex number from its JSON ``{"real", "imag"}`` form."""
    if isinstance(value, dict):
        return complex(float(value.get("real") or 0.0), float(value.get("imag") or 0.0))
    return complex(value)


def matrix_from_json(value: Any) -> np.ndarray:
    """Return a complex array from a nested ``{"real", "imag"}`` JSON value."""
    if isinstance(value, list):
        return np.asarray([matrix_from_json(item) for item in value], dtype=complex)
    return np.asarray(complex_from_json(value))


def _json_column(product: Dataset, name: str) -> list[dict[str, Any]]:
    column = np.asarray(product.handle(name).materialize())
    return [json.loads(str(value)) for value in np.atleast_1d(column)]


def _diagnostics(rows: list[CandidateRow]) -> dict[str, Any]:
    """Return the per-candidate diagnostics as arrays, as the solver exposed them."""
    names = sorted(
        {name for row in rows for name in (*row.base_audit, *row.classification)}
    )
    diagnostics: dict[str, Any] = {}
    for name in names:
        values = [
            row.classification.get(name, row.base_audit.get(name, np.nan))
            for row in rows
        ]
        if name == "state_tangent_vector":
            diagnostics[name] = np.asarray(values, dtype=float)
        elif name == "state_tangent_matrix":
            diagnostics[name] = np.asarray(
                [matrix_from_json(value) for value in values]
            )
        elif name == "verified_full_state_decimal_values":
            diagnostics[name] = np.asarray(
                [tuple(str(item) for item in (value or ())) for value in values],
                dtype=object,
            )
        else:
            diagnostics[name] = values
    return diagnostics


def _branch_table(rows: list[CandidateRow]) -> CandidateBranchTable:
    fields: dict[str, list[Any]] = {name: [] for name in _BRANCH_FIELDS}
    for row in rows:
        signatures = row.classification.get("scaling_signatures") or ()
        local = 0
        for signature_index, signature in enumerate(signatures):
            branches = signature.get("branches") or (
                {
                    "epsilon_side": 0,
                    "amplitude": np.nan,
                    "leading_state_coefficient": None,
                },
            )
            for branch in branches:
                coefficient = branch.get("leading_state_coefficient")
                fields["candidate_index"].append(row.index)
                fields["local_branch_index"].append(local)
                fields["signature_index"].append(signature_index)
                fields["state_order"].append(int(signature["state_order"]))
                fields["perturbation_order"].append(
                    int(signature["perturbation_order"])
                )
                fields["coupling_state_order"].append(
                    int(signature["coupling_state_order"])
                )
                fields["exponent_numerator"].append(
                    int(signature["exponent_numerator"])
                )
                fields["exponent_denominator"].append(
                    int(signature["exponent_denominator"])
                )
                fields["epsilon_side"].append(int(branch.get("epsilon_side") or 0))
                amplitude = branch.get("amplitude")
                fields["amplitude"].append(
                    float(amplitude) if amplitude is not None else np.nan
                )
                fields["real_branch"].append(bool(branch.get("epsilon_side", 0)))
                fields["sublinear"].append(bool(signature.get("sublinear", False)))
                fields["leading_state_coefficient"].append(
                    np.full((0, 0), np.nan + 0j)
                    if coefficient is None
                    else matrix_from_json(coefficient)
                )
                local += 1
    return CandidateBranchTable(
        candidate_index=np.asarray(fields["candidate_index"], dtype=np.int64),
        local_branch_index=np.asarray(fields["local_branch_index"], dtype=np.int64),
        signature_index=np.asarray(fields["signature_index"], dtype=np.int64),
        state_order=np.asarray(fields["state_order"], dtype=np.int64),
        perturbation_order=np.asarray(fields["perturbation_order"], dtype=np.int64),
        coupling_state_order=np.asarray(fields["coupling_state_order"], dtype=np.int64),
        exponent_numerator=np.asarray(fields["exponent_numerator"], dtype=np.int64),
        exponent_denominator=np.asarray(fields["exponent_denominator"], dtype=np.int64),
        epsilon_side=np.asarray(fields["epsilon_side"], dtype=np.int64),
        amplitude=np.asarray(fields["amplitude"], dtype=float),
        real_branch=np.asarray(fields["real_branch"], dtype=bool),
        sublinear=np.asarray(fields["sublinear"], dtype=bool),
        leading_state_coefficient=np.asarray(
            fields["leading_state_coefficient"], dtype=object
        ),
    )


def candidate_view(
    product: Dataset, *, metadata: dict[str, Any] | None = None
) -> CandidateView:
    """Return the typed view of one candidate product."""
    identities = np.asarray(product.handle("candidate_identity").materialize())
    states = np.asarray(product.handle("states").materialize())
    vectors = np.asarray(product.handle("state_vectors").materialize())
    controls = np.asarray(product.handle("control_values").materialize())
    control_names = [
        str(value)
        for value in np.asarray(product.handle("control_names").materialize())
    ]
    full = np.asarray(product.handle("full_residual_norm").materialize())
    search = np.asarray(product.handle("search_residual_norm").materialize())
    success = np.asarray(product.handle("success").materialize(), dtype=bool)
    status = [
        str(value) for value in np.asarray(product.handle("status").materialize())
    ]
    method = [
        str(value) for value in np.asarray(product.handle("method").materialize())
    ]
    digits = np.asarray(product.handle("verification_digits").materialize())
    verification = [
        str(value)
        for value in np.asarray(product.handle("verification_status").materialize())
    ]
    classification = _json_column(product, "classification")
    base_audit = _json_column(product, "base_audit")
    rows = [
        CandidateRow(
            index=index,
            identity=str(identities[index]),
            state=states[index],
            state_vector=vectors[index],
            controls={
                name: float(controls[index, position])
                for position, name in enumerate(control_names)
            },
            full_residual_norm=float(full[index]),
            search_residual_norm=float(search[index]),
            success=bool(success[index]),
            status=status[index],
            method=method[index],
            verification_digits=int(digits[index]),
            verification_status=verification[index],
            classification=classification[index],
            base_audit=base_audit[index],
        )
        for index in range(len(states))
    ]
    case_params = {
        variable.name.removeprefix("case_parameter."): np.asarray(
            product.handle(variable.name).materialize()
        )
        for variable in product.variables
        if variable.name.startswith("case_parameter.")
    }
    offsets = (
        np.asarray(product.handle("candidate_offsets").materialize(), dtype=np.int64)
        if "candidate_offsets" in {variable.name for variable in product.variables}
        else np.asarray([], dtype=np.int64)
    )
    resolved = dict(metadata or {})
    fixed = dict(resolved.get("fixed_params", {}))
    return CandidateView(
        rows=rows,
        branches=_branch_table(rows),
        meta=resolved,
        diagnostics=_diagnostics(rows),
        state_vectors=vectors,
        case_params=case_params,
        candidate_offsets=offsets,
        fixed_params=fixed,
    )


def branch_axis_size(view: CandidateView) -> int:
    """Return the widest per-candidate branch count in the view."""
    if not view.rows:
        return 0
    counts = [len(view.branches_of(row.index)) for row in view.rows]
    return max(counts, default=0)


def _column_dtype(values: list[Any]) -> Any:
    kinds = {type(value) for value in values if value is not None}
    if kinds <= {bool}:
        return bool
    if kinds <= {int}:
        return np.int64
    if kinds <= {complex}:
        return np.complex128
    if kinds <= {str}:
        width = max((len(str(value)) for value in values), default=0)
        return f"U{max(width, 1)}"
    return np.float64


def pivot_branch_rows(
    view: CandidateView,
    rows: list[dict[str, Any]],
    *,
    sample_axis: str | None = None,
    rename: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]], dict[str, int]]:
    """Pivot per-(candidate, branch[, sample]) rows into typed field columns.

    The rows are anchored to a candidate by ``candidate_index`` and to a branch by
    its global ``branch_index`` in the view's branch table; the branch's local
    index inside its candidate comes from the table. With ``sample_axis`` set the
    rows of one branch become that padded axis, otherwise each branch is one entry
    of the branch axis. Missing entries are padded, so a ragged branch set is
    representable without a JSON payload.
    """
    # The row's ``branch_index`` is the branch's position in the view's branch
    # table; its local index is the position inside its own candidate.
    local = {
        position: int(view.branches.local_branch_index[position])
        for position in range(view.branches.size)
    }
    positions = {row.index: position for position, row in enumerate(view.rows)}
    counts: dict[tuple[int, int], int] = {}
    placed: list[tuple[int, int, int, dict[str, Any]]] = []
    for row in rows:
        candidate = int(row["candidate_index"])
        branch = local[int(row["branch_index"])]
        key = (candidate, branch)
        sample = counts.get(key, 0)
        counts[key] = sample + 1
        placed.append((positions[candidate], branch, sample, row))
    n_branch = branch_axis_size(view)
    n_sample = max(counts.values(), default=1) if sample_axis is not None else 1
    tail: tuple[str, ...] = (
        ("branch",) if sample_axis is None else ("branch", sample_axis)
    )
    axes: dict[str, int] = {"branch": n_branch}
    if sample_axis is not None:
        axes[sample_axis] = n_sample
    rename = dict(rename or {})
    names = sorted({name for _, _, _, row in placed for name in row})
    fields: dict[str, Any] = {}
    field_axes: dict[str, tuple[str, ...]] = {}
    for name in names:
        dtype = _column_dtype([row.get(name) for _, _, _, row in placed])
        kind = np.dtype(dtype).kind
        shape: tuple[int, ...] = (len(view.rows), n_branch)
        if sample_axis is not None:
            shape = (len(view.rows), n_branch, n_sample)
        if kind in "fc":
            array = np.full(shape, np.nan, dtype=dtype)
        elif kind == "U":
            array = np.full(shape, "", dtype=dtype)
        elif kind == "b":
            array = np.zeros(shape, dtype=bool)
        else:
            array = np.zeros(shape, dtype=dtype)
        for candidate, branch, sample, row in placed:
            value = row.get(name)
            if value is None:
                continue
            if sample_axis is None:
                array[candidate, branch] = value
            else:
                array[candidate, branch, sample] = value
        resolved = rename.get(name, name)
        fields[resolved] = array
        field_axes[resolved] = tail
    return fields, field_axes, axes
