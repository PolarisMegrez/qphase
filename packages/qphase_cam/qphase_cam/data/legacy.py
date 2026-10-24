"""qphase_cam: one-off import of the pre-2.0 CAM results
---------------------------------------------------------
This module exists to move historical results into the typed products of
Phase 10C1 and to be deleted once nothing old is left to read. It is a
*compatibility* layer, not a second product implementation: it normalizes a
legacy object (or its trusted NPZ payload) into the same builder arguments a
future solver will produce, and it never keeps a legacy structure alive.

Identity
--------
An imported candidate gets ``legacy:<fingerprint>:<case>:<candidate>``, where
``<fingerprint>`` is a SHA-256 content digest of the source payload (core
arrays, offsets, coordinates, diagnostics, metadata — never a path, an
``id()`` or a ``hash()``). Two consequences matter:

* importing the same artifact twice yields exactly the same identities;
* two artifacts with different scientific content cannot collide, even when
  they share a label, a result kind or a case index — which is what allows two
  imports to be merged into one product without inventing a conflict.

The fingerprint identifies an *imported snapshot*. It is not a branch id: a
recomputation, a different precision or another run is a different snapshot and
therefore a different identity. Phase 10D supplies the solver-owned identity
that survives those.

The continuation importer reads the *fixed-point* legacy result whose metadata
marks it as a continuation run; it takes the filled solution slot of every point
and carries the swept parameter as ``continuation_coordinate``.

What is not mapped
------------------
``postprocess`` tables, ``branches`` tables and any ``meta``/diagnostic value
that has no JSON representation are reported by key name in the bundle
provenance (``unmapped_legacy_*``). Per-candidate diagnostics *are* mapped,
including object arrays such as ``scaling_signatures``: the classification
evidence survives as a canonical JSON text column instead of being dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qphase.data import GenericDataBundle

from ._common import content_digest, json_mapping
from .bundle import cam_bundle
from .candidates import build_candidate_product, split_candidate_diagnostics
from .continuation import build_continuation_product
from .state import build_state_product

if TYPE_CHECKING:
    from ..bifurcation_result import (
        CAMBifurcationResult,
        CAMBifurcationScanResult,
    )
    from ..result import CAMResult

__all__ = [
    "import_bifurcation_result",
    "import_bifurcation_scan",
    "import_cam_result",
    "import_continuation_result",
    "import_legacy_artifact",
]


# --- source identity -------------------------------------------------------


def _branch_payload(result: Any) -> dict[str, Any]:
    """Return the branch table of a legacy result as a plain mapping."""
    branches = getattr(result, "branches", None)
    if branches is None:
        return {}
    return {
        str(name): getattr(branches, name) for name in branches.__dataclass_fields__
    }


def _state_payload(result: Any) -> dict[str, Any]:
    """Return the identifying content of a legacy fixed-point result."""
    return {
        "states": result.states,
        "residuals": result.residuals,
        "success": result.success,
        "valid_mask": result.valid_mask,
        "solution_count": result.solution_count,
        "params": result.params,
        "axes": result.axes,
        "postprocess": result.postprocess,
        "meta": result.meta,
    }


def _candidate_payload(result: Any) -> dict[str, Any]:
    """Return the identifying content of a legacy candidate result."""
    return {
        "states": result.states,
        "state_vectors": result.state_vectors,
        "control_values": result.control_values,
        "control_names": result.control_names,
        "full_residual_norm": result.full_residual_norm,
        "search_residual_norm": result.search_residual_norm,
        "success": result.success,
        "status": result.status,
        "method": result.method,
        "verification_digits": result.verification_digits,
        "verification_status": result.verification_status,
        "branches": _branch_payload(result),
        "diagnostics": result.diagnostics,
        "postprocess": result.postprocess,
        "meta": result.meta,
    }


def _scan_payload(result: Any) -> dict[str, Any]:
    """Return the identifying content of a legacy candidate scan."""
    return {
        "case_axes": result.case_axes,
        "case_shape": result.case_shape,
        "case_params": result.case_params,
        "candidate_offsets": result.candidate_offsets,
        "case_metadata": result.case_metadata,
        "candidates": _candidate_payload(result.candidates),
        "postprocess": result.postprocess,
        "meta": result.meta,
    }


def _source_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the content fingerprint used inside a legacy identity."""
    return content_digest(payload)


def _resolve_token(
    payload: Mapping[str, Any], source_id: str | None
) -> tuple[str, str]:
    """Return ``(identity token, how it was obtained)``.

    A caller that already owns a stable artifact id passes it as ``source_id``;
    otherwise the token is the content fingerprint, which is deterministic for
    an in-memory object as well as for a file.
    """
    if source_id is not None:
        token = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in source_id.strip()
        )
        if not token:
            raise ValueError("source_id must contain at least one usable character")
        return token, "source_id"
    return _source_fingerprint(payload), "content_fingerprint"


# --- legacy payload helpers ------------------------------------------------


def _state_matrix_shape(result: Any, n_candidate: int) -> tuple[int, int]:
    """Return the matrix shape of a candidate state block."""
    array = np.asarray(result.states)
    if array.ndim == 3:
        return int(array.shape[1]), int(array.shape[2])
    declared = result.meta.get("state_matrix_shape") if result.meta else None
    if declared:
        shape = tuple(int(value) for value in declared)
        if len(shape) == 2:
            return shape
    if n_candidate == 0:
        return (0, 0)
    raise ValueError(
        f"cannot infer the candidate state matrix shape from {array.shape}"
    )


def _legacy_matrix(values: Any, n_candidate: int, width: int, dtype: Any) -> np.ndarray:
    """Return one per-candidate matrix block, tolerating an empty payload."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 2 and array.shape[0] == n_candidate:
        return array
    if array.size == 0:
        return np.zeros((n_candidate, width), dtype=dtype)
    return array.reshape(n_candidate, -1 if width < 0 else width)


def _legacy_candidate_count(result: Any) -> int:
    """Return how many candidates a legacy result carries."""
    return int(len(np.asarray(result.states)))


def _legacy_candidate_arrays(result: Any) -> dict[str, Any]:
    """Normalize a legacy candidate result into builder keyword arrays."""
    n_candidate = _legacy_candidate_count(result)
    matrix_shape = _state_matrix_shape(result, n_candidate)
    control_names = tuple(str(value) for value in result.control_names)
    return {
        "states": np.asarray(result.states, dtype=np.complex128).reshape(
            (n_candidate, *matrix_shape)
        ),
        "state_vectors": _legacy_matrix(
            result.state_vectors, n_candidate, -1, np.float64
        ),
        "control_values": _legacy_matrix(
            result.control_values, n_candidate, len(control_names), np.float64
        ),
        "control_names": control_names,
        "full_residual_norm": np.asarray(result.full_residual_norm, dtype=np.float64),
        "search_residual_norm": np.asarray(
            result.search_residual_norm, dtype=np.float64
        ),
        "success": np.asarray(result.success, dtype=bool),
        "status": [str(value) for value in result.status],
        "method": [str(value) for value in result.method],
        "verification_digits": np.asarray(result.verification_digits, dtype=np.int64),
        "verification_status": [str(value) for value in result.verification_status],
    }


def _branch_names(result: Any) -> list[str]:
    """Return the branch-table field names that were not mapped."""
    branches = getattr(result, "branches", None)
    if branches is None:
        return []
    return sorted(str(name) for name in branches.__dataclass_fields__)


def _legacy_path(path: str | Path) -> Path:
    """Resolve the legacy ``.npz`` payload of an unsuffixed result path."""
    target = Path(path)
    if not target.exists() and target.suffix != ".npz":
        return target.with_suffix(".npz")
    return target


def _trusted_npz(path: str | Path) -> Any:
    """Open one trusted pre-2.0 NPZ payload.

    ``allow_pickle`` is deliberate and bounded: a pre-2.0 artifact stores its
    ``params``/``meta``/``diagnostics`` mappings as object arrays, and this
    importer exists to read exactly those historical files. Every value it
    keeps afterwards is typed; only the trusted-input step unpickles.
    """
    return np.load(_legacy_path(path), allow_pickle=True)


# --- importers -------------------------------------------------------------


def import_cam_result(
    source: CAMResult | str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    source_id: str | None = None,
) -> GenericDataBundle:
    """Import one :class:`~qphase_cam.result.CAMResult`.

    The legacy ``postprocess`` mapping is not mapped: it is reported by key
    name in ``unmapped_legacy_postprocess``. ``params`` and ``axes`` entries
    that are neither scalar nor grid-shaped are reported in
    ``unmapped_legacy_parameters``/``unmapped_legacy_axes``. A fixed-point
    product carries no candidate identity, so ``source_id`` only labels the
    provenance fingerprint.
    """
    from ..result import CAMResult

    result = (
        CAMResult.load(_legacy_path(source))
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(result, CAMResult):
        raise TypeError(f"expected a CAMResult, got {type(result).__name__}")
    dataset = build_state_product(
        states=result.states,
        residuals=result.residuals,
        success=result.success,
        valid_mask=result.valid_mask,
        solution_count=result.solution_count,
        parameters=result.params,
        axes=result.axes,
        grid_shape=result.grid_shape,
        provenance={"legacy_source": "qphase_cam.result:CAMResult"},
    )
    product_provenance = dict(dataset.provenance)
    meta, unmapped_meta = json_mapping(result.meta or {})
    token, token_source = _resolve_token(_state_payload(result), source_id)
    name = str(dataset.attributes["product"])
    return cam_bundle(
        {name: dataset},
        provenance={
            **dict(provenance or {}),
            "legacy_source": "qphase_cam.result:CAMResult",
            "legacy_result_kind": result.result_kind,
            "legacy_source_fingerprint": token,
            "legacy_source_token": token_source,
            "legacy_meta": meta,
            "unmapped_legacy_postprocess": sorted(
                str(key) for key in (result.postprocess or {})
            ),
            "unmapped_legacy_meta": unmapped_meta,
            "unmapped_legacy_parameters": product_provenance[
                "unmapped_parameter_names"
            ],
            "unmapped_legacy_axes": product_provenance["unmapped_axis_names"],
        },
        metadata={"label": result.label},
    )


def import_bifurcation_result(
    source: CAMBifurcationResult | str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    source_id: str | None = None,
) -> GenericDataBundle:
    """Import one :class:`~qphase_cam.bifurcation_result.CAMBifurcationResult`.

    A single case is normalized to a scan of size one, so both importers
    produce the same schema. Candidate identities are
    ``legacy:<fingerprint>:0:<candidate>``; pass ``source_id`` to use an
    artifact id you already own instead of the content fingerprint.
    """
    from ..bifurcation_result import CAMBifurcationResult

    result = (
        CAMBifurcationResult.load(_legacy_path(source))
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(result, CAMBifurcationResult):
        raise TypeError(f"expected a CAMBifurcationResult, got {type(result).__name__}")
    token, token_source = _resolve_token(_candidate_payload(result), source_id)
    n_candidate = _legacy_candidate_count(result)
    arrays = _legacy_candidate_arrays(result)
    classification, base_audit, unmapped_diagnostics = split_candidate_diagnostics(
        result.diagnostics, n_candidate
    )
    dataset = build_candidate_product(
        candidate_identity=[
            f"legacy:{token}:0:{position}" for position in range(n_candidate)
        ],
        classification=classification,
        base_audit=base_audit,
        provenance={
            "legacy_source": "qphase_cam.bifurcation_result:CAMBifurcationResult"
        },
        **arrays,
    )
    meta, unmapped_meta = json_mapping(result.meta or {})
    return cam_bundle(
        {"candidates": dataset},
        provenance={
            **dict(provenance or {}),
            "legacy_source": "qphase_cam.bifurcation_result:CAMBifurcationResult",
            "legacy_result_kind": result.result_kind,
            "legacy_source_fingerprint": token,
            "legacy_source_token": token_source,
            "legacy_meta": meta,
            "unmapped_legacy_postprocess": sorted(
                str(key) for key in (result.postprocess or {})
            ),
            "unmapped_legacy_diagnostics": unmapped_diagnostics,
            "unmapped_legacy_branches": _branch_names(result),
            "unmapped_legacy_meta": unmapped_meta,
        },
        metadata={"label": result.label},
    )


def import_bifurcation_scan(
    source: CAMBifurcationScanResult | str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    source_id: str | None = None,
) -> GenericDataBundle:
    """Import one ragged scan of bifurcation candidates.

    The case grid is flattened into the ``scan`` axis and the original shape is
    kept as the ``case_shape`` attribute. ``group_scan_index`` is the flattened
    case index, so a consumer can rebuild a case's candidates from
    ``candidate_offsets`` alone.
    """
    from ..bifurcation_result import CAMBifurcationScanResult

    result = (
        CAMBifurcationScanResult.load(_legacy_path(source))
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(result, CAMBifurcationScanResult):
        raise TypeError(
            f"expected a CAMBifurcationScanResult, got {type(result).__name__}"
        )
    token, token_source = _resolve_token(_scan_payload(result), source_id)
    offsets = np.asarray(result.candidate_offsets, dtype=np.int64)
    n_case = int(offsets.size) - 1
    case_metadata = list(result.case_metadata)
    if len(case_metadata) != n_case:
        raise ValueError(
            f"case_metadata has {len(case_metadata)} entries for {n_case} cases"
        )
    identity: list[str] = []
    for case in range(n_case):
        start = int(offsets[case])
        stop = int(offsets[case + 1])
        identity.extend(
            f"legacy:{token}:{case}:{position}" for position in range(stop - start)
        )
    arrays = _legacy_candidate_arrays(result.candidates)
    n_candidate = _legacy_candidate_count(result.candidates)
    if n_candidate != len(identity):
        raise ValueError(
            f"the candidate payload has {n_candidate} rows but the offsets "
            f"describe {len(identity)}"
        )
    classification, base_audit, unmapped_diagnostics = split_candidate_diagnostics(
        result.candidates.diagnostics, n_candidate
    )
    dataset = build_candidate_product(
        candidate_identity=identity,
        classification=classification,
        base_audit=base_audit,
        case_metadata=case_metadata,
        candidate_offsets=offsets,
        group_scan_index=np.arange(n_case, dtype=np.int64),
        case_axes=result.case_axes,
        case_params=result.case_params,
        case_shape=result.case_shape,
        provenance={
            "legacy_source": "qphase_cam.bifurcation_result:CAMBifurcationScanResult"
        },
        **arrays,
    )
    product_provenance = dict(dataset.provenance)
    scan_meta, unmapped_scan_meta = json_mapping(result.meta or {})
    candidate_meta, unmapped_candidate_meta = json_mapping(result.candidates.meta or {})
    return cam_bundle(
        {"candidates": dataset},
        provenance={
            **dict(provenance or {}),
            "legacy_source": "qphase_cam.bifurcation_result:CAMBifurcationScanResult",
            "legacy_result_kind": result.result_kind,
            "legacy_source_fingerprint": token,
            "legacy_source_token": token_source,
            "legacy_meta": scan_meta,
            "legacy_candidate_meta": candidate_meta,
            "unmapped_legacy_postprocess": sorted(
                {
                    str(key)
                    for key in (
                        *(result.postprocess or {}),
                        *(result.candidates.postprocess or {}),
                    )
                }
            ),
            "unmapped_legacy_diagnostics": unmapped_diagnostics,
            "unmapped_legacy_branches": _branch_names(result.candidates),
            "unmapped_legacy_meta": sorted(
                {*unmapped_scan_meta, *unmapped_candidate_meta}
            ),
            "unmapped_legacy_case_axes": product_provenance["unmapped_case_axis_names"],
            "unmapped_legacy_case_params": product_provenance[
                "unmapped_case_parameter_names"
            ],
        },
        metadata={"label": result.label},
    )


def _continuation_columns(
    result: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    """Flatten a legacy continuation result into per-point columns.

    A legacy continuation result keeps its points on the leading grid axis with a
    fixed-capacity solution axis behind it, and exactly one of those slots filled.
    The product has one state per point, so the filled slot is taken and a point
    without a solution becomes NaN/False instead of disappearing.
    """
    meta = dict(result.meta or {})
    if not meta.get("continuation"):
        raise ValueError("not a continuation result: meta['continuation'] must be true")
    axes = {
        str(name): np.asarray(values) for name, values in (result.axes or {}).items()
    }
    if len(axes) != 1:
        raise ValueError(
            "a continuation result carries exactly one continuation coordinate "
            f"axis; got {sorted(axes)}"
        )
    name, coordinate = next(iter(axes.items()))
    coordinate = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    shape = result.grid_shape
    n_point = int(np.prod(shape, dtype=np.int64))
    if coordinate.size != n_point:
        raise ValueError(
            f"the continuation axis {name!r} has {coordinate.size} values for "
            f"{n_point} points"
        )
    states = np.asarray(result.states, dtype=np.complex128)
    if states.ndim != 4 or states.shape[0] != n_point:
        raise ValueError(
            "states must span (point, solution, state_row, state_col); got "
            f"shape {states.shape}"
        )
    capacity = int(states.shape[1])
    valid = np.asarray(result.valid_mask, dtype=bool).reshape(n_point, capacity)
    counts = np.asarray(result.solution_count, dtype=np.int64).reshape(n_point)
    if np.any(counts > 1):
        raise ValueError(
            "a continuation point carries at most one state, but solution_count "
            f"reports up to {int(counts.max())}"
        )
    if np.any(valid[:, 1:].any(axis=1)):
        raise ValueError(
            "a continuation point fills only its first solution slot; the legacy "
            "payload has a later slot set"
        )
    filled = valid[:, 0]
    residuals = np.asarray(result.residuals, dtype=np.float64).reshape(
        n_point, capacity
    )[:, 0]
    success = np.asarray(result.success, dtype=bool).reshape(n_point, capacity)[:, 0]
    point_states = np.where(filled[:, None, None], states[:, 0], np.nan + 1j * np.nan)
    residuals = np.where(filled, residuals, np.nan)
    success = np.logical_and(success, filled)
    return coordinate, point_states, residuals, success, name, counts


def import_continuation_result(
    source: CAMResult | str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    source_id: str | None = None,
) -> GenericDataBundle:
    """Import one legacy continuation run as a single local path.

    The legacy result keeps the swept parameter in one axis and the states in the
    fixed-capacity slots of the leading grid axis, so the import takes the filled
    slot of every point, names the path ``legacy:<fingerprint>:0`` and carries the
    swept values as ``continuation_coordinate``. The legacy result has no event
    table, so every point gets an empty ``{}`` event row.

    A result whose ``meta['continuation']`` is not true, or which does not carry
    exactly one continuation axis, is rejected: importing it as a path would
    invent a continuation that never happened.
    """
    from ..result import CAMResult

    result = (
        CAMResult.load(_legacy_path(source))
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(result, CAMResult):
        raise TypeError(f"expected a CAMResult, got {type(result).__name__}")
    coordinate, states, residuals, success, name, counts = _continuation_columns(result)
    token, token_source = _resolve_token(_state_payload(result), source_id)
    dataset = build_continuation_product(
        path_identity=[f"legacy:{token}:0"],
        continuation_coordinate=coordinate,
        continuation_coordinate_name=name,
        states=states,
        residuals=residuals,
        success=success,
        parameters=result.params,
        axes=result.axes,
        provenance={"legacy_source": "qphase_cam.result:CAMResult"},
    )
    product_provenance = dict(dataset.provenance)
    meta, unmapped_meta = json_mapping(result.meta or {})
    return cam_bundle(
        {"continuation": dataset},
        provenance={
            **dict(provenance or {}),
            "legacy_source": "qphase_cam.result:CAMResult",
            "legacy_result_kind": result.result_kind,
            "legacy_source_fingerprint": token,
            "legacy_source_token": token_source,
            "legacy_meta": meta,
            "legacy_continuation_parameter": name,
            "legacy_point_count": int(coordinate.size),
            "legacy_empty_point_count": int(np.count_nonzero(counts == 0)),
            "unmapped_legacy_postprocess": sorted(
                str(key) for key in (result.postprocess or {})
            ),
            "unmapped_legacy_meta": unmapped_meta,
            "unmapped_legacy_parameters": product_provenance[
                "unmapped_parameter_names"
            ],
            "unmapped_legacy_axes": product_provenance["unmapped_axis_names"],
        },
        metadata={"label": result.label},
    )


def import_legacy_artifact(
    path: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
    source_id: str | None = None,
) -> GenericDataBundle:
    """Import a trusted pre-2.0 NPZ payload, dispatching on its keys.

    A legacy artifact records its own result type only in the file layout and in
    its metadata, so the dispatcher reads both: ``case_axes`` is a candidate
    scan, ``control_names`` a single candidate set, and among the fixed-point
    payloads (``valid_mask``) ``meta['continuation']`` marks a continuation run.
    """
    with _trusted_npz(path) as data:
        keys = set(data.files)
        meta = data["meta"].item() if "meta" in keys else {}
    target = _legacy_path(path)
    if "valid_mask" in keys:
        if isinstance(meta, Mapping) and meta.get("continuation"):
            return import_continuation_result(
                target, provenance=provenance, source_id=source_id
            )
        return import_cam_result(target, provenance=provenance, source_id=source_id)
    if "case_axes" in keys:
        return import_bifurcation_scan(
            target, provenance=provenance, source_id=source_id
        )
    if "control_names" in keys:
        return import_bifurcation_result(
            target, provenance=provenance, source_id=source_id
        )
    raise ValueError(
        f"{target} is not a recognizable pre-2.0 CAM artifact; keys: {sorted(keys)}"
    )
