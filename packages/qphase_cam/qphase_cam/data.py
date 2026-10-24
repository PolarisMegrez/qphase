"""qphase_cam: Typed Data Products and Legacy Result Import (2.0)
---------------------------------------------------------
The CAM product vocabulary in three layers:

* the six :class:`qphase.data.ProductDeclaration` records the task profiles
  reference: a name, a data kind, a quantity and the field vocabulary of one
  product. Phase 10C1 replaced the two placeholder field names of the 10B
  planning audit with the fields the implemented schemas actually carry.
* three concrete :class:`qphase.data.ProductSchema` constants —
  :data:`CAM_STATE_PRODUCT` (one parameter point), :data:`CAM_STATE_SCAN_PRODUCT`
  (a scan of parameter points) and :data:`CAM_CANDIDATE_PRODUCT`
  (bifurcation/search candidates) — plus the builders that close their axes
  and materialize a :class:`~qphase.data.StatisticsDataset`.
* a one-off compatibility layer importing the pre-2.0 result objects
  (``CAMResult``, ``CAMBifurcationResult``, ``CAMBifurcationScanResult``) or
  their trusted NPZ payloads into a
  :class:`~qphase.data.GenericDataBundle`, and the export comparisons that
  show nothing was lost on the way.

Schema rules kept here
----------------------
* Payloads are typed NumPy arrays only: no object arrays, no pickles, no
  ``dict`` payload columns. Classification decisions, audit rows and case
  metadata are canonical-JSON *text* columns, so they stay typed and diffable
  without smuggling arbitrary Python objects into a product.
* The ``solution`` axis is a fixed-capacity slot, never a branch identity. Two
  scan points that both solve three states share three slots; nothing in the
  product claims those states continue into each other along the parameter
  axis. Cross-parameter branch identity is the continuation product's job
  (Phase 10C2).
* ``candidate_identity`` is an independent ``U`` string variable holding one
  non-empty, globally unique value per candidate. It is never derived from an
  array slot. Imported legacy candidates get
  ``legacy:<source>:<case>:<candidate>``, where ``<source>`` is the result
  label when it has one and the result kind otherwise. That identity is stable
  *inside the imported artifact only*; the solver-owned identity arrives with
  Phase 10D.
* Ragged candidates use the core ragged contract: a ``scan`` axis over the
  flattened case grid, a ``group``/``group_offset`` pairing and a
  ``candidate_offsets`` variable of length ``n_case + 1`` that starts at zero,
  is monotonic and ends at the candidate count. Empty cases and empty
  candidate lists are representable. The original (possibly
  multi-dimensional) case grid shape is preserved as the ``case_shape``
  attribute.
* Parameter coordinates are ordinary typed ``parameter.<name>`` variables
  plus a :class:`~qphase.data.CoordinateSchema`; there is no arbitrary dict
  payload anywhere in a product.

What the importer does *not* map
--------------------------------
Legacy ``postprocess`` tables, per-candidate ``diagnostics`` that cannot be
expressed per candidate, ``branches`` tables and unmappable metadata keys are
listed by key name in the bundle provenance
(``unmapped_legacy_postprocess``, ``unmapped_legacy_diagnostics``,
``unmapped_legacy_branches``, ``unmapped_legacy_meta``). They are never
dropped silently and never squeezed into a typed column; Phase 10F adopts them
as their own products.

Public API
----------
CAM_STATE_PRODUCT
    Schema template of one solved parameter point.
CAM_STATE_SCAN_PRODUCT
    Schema template of steady states resolved across a scan axis.
CAM_CANDIDATE_PRODUCT
    Schema template of ragged bifurcation/search candidates.
CAM_BUNDLE_TYPE_ID
    Bundle descriptor type id of a CAM product collection.
build_state_product
    Build the state (or state-scan) dataset from fixed-capacity arrays.
build_candidate_product
    Build the ragged candidate dataset from flattened case arrays.
cam_bundle
    Group CAM products into a :class:`~qphase.data.GenericDataBundle`.
import_cam_result
    Import a :class:`~qphase_cam.result.CAMResult`.
import_bifurcation_result
    Import a :class:`~qphase_cam.bifurcation_result.CAMBifurcationResult`.
import_bifurcation_scan
    Import a :class:`~qphase_cam.bifurcation_result.CAMBifurcationScanResult`.
import_legacy_artifact
    Import a trusted pre-2.0 NPZ payload, dispatching on its keys.
compare_state_export
    Per-field comparison of a state product against its legacy result.
compare_candidates_export
    Per-field comparison of a candidate product against its legacy result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qphase.core.utils import canonical_json
from qphase.data import (
    AxisRole,
    AxisSchema,
    BundleDescriptor,
    CoordinateSchema,
    DataKind,
    Dataset,
    GenericDataBundle,
    ProductDeclaration,
    ProductSchema,
    StatisticsDataset,
    VariableSchema,
)

if TYPE_CHECKING:
    from .bifurcation_result import CAMBifurcationResult, CAMBifurcationScanResult
    from .result import CAMResult

__all__ = [
    "CAM_ANNOTATION_DECLARATION",
    "CAM_BUNDLE_TYPE_ID",
    "CAM_CANDIDATE_DECLARATION",
    "CAM_CANDIDATE_PRODUCT",
    "CAM_CONTINUATION_DECLARATION",
    "CAM_EVIDENCE_DECLARATION",
    "CAM_STATE_DECLARATION",
    "CAM_STATE_PRODUCT",
    "CAM_STATE_SCAN_DECLARATION",
    "CAM_STATE_SCAN_PRODUCT",
    "build_candidate_product",
    "build_state_product",
    "cam_bundle",
    "compare_candidates_export",
    "compare_state_export",
    "import_bifurcation_result",
    "import_bifurcation_scan",
    "import_cam_result",
    "import_legacy_artifact",
]

# --- planning declarations -------------------------------------------------

#: Solve results. The ``parameters`` family is carried as ``parameter.<name>``
#: coordinate variables; ``success``/``valid_mask``/``solution_count`` describe
#: a fixed-capacity solution axis, which is why no ragged offset field appears.
CAM_STATE_DECLARATION = ProductDeclaration(
    name="state",
    kind=DataKind.STATISTICS,
    quantity="cam_steady_state",
    fields=[
        "parameters",
        "states",
        "residuals",
        "success",
        "valid_mask",
        "solution_count",
    ],
    table="row_table",
)

#: The same records across a scan axis, so one scan consumes one product.
CAM_STATE_SCAN_DECLARATION = ProductDeclaration(
    name="state_scan",
    kind=DataKind.STATISTICS,
    quantity="cam_steady_state_scan",
    fields=[
        "parameters",
        "states",
        "residuals",
        "success",
        "valid_mask",
        "solution_count",
    ],
    table="row_table",
)

#: Candidates proposed by a search task: what axis they were found on, their
#: identity, the states/controls that witness them, the classification and the
#: audit rows behind the decision.
CAM_CANDIDATE_DECLARATION = ProductDeclaration(
    name="candidates",
    kind=DataKind.STATISTICS,
    quantity="cam_bifurcation_candidate",
    fields=[
        "case_axes",
        "case_params",
        "candidate_offsets",
        "candidate_identity",
        "states",
        "state_vectors",
        "control_values",
        "control_names",
        "full_residual_norm",
        "search_residual_norm",
        "success",
        "status",
        "method",
        "verification_digits",
        "verification_status",
        "classification",
        "base_audit",
        "case_metadata",
    ],
    table="row_table",
)

#: Search evidence. It must be able to express a *negative* result — a search
#: that covered the design space and accepted no candidate — so it carries the
#: search design, the coverage, the accounting of attempts and rejections, the
#: source artifacts it was derived from, the accepted candidates and the
#: rejection summary. No state or signature column is mandatory.
CAM_EVIDENCE_DECLARATION = ProductDeclaration(
    name="evidence",
    kind=DataKind.STATISTICS,
    quantity="cam_search_evidence",
    fields=[
        "search_design",
        "coverage",
        "accounting",
        "source_artifacts",
        "accepted_candidates",
        "rejection_summary",
    ],
    table="row_table",
)

#: Continuation output: which path a row belongs to, its position along the
#: continuation coordinate, the states and the events met on the way.
CAM_CONTINUATION_DECLARATION = ProductDeclaration(
    name="continuation",
    kind=DataKind.STATISTICS,
    quantity="cam_continuation_branch",
    fields=[
        "path_identity",
        "continuation_coordinate",
        "states",
        "events",
    ],
    table="row_table",
)

#: Postprocessor annotations: what they are attached to, the annotation fields
#: and the method metadata that produced them.
CAM_ANNOTATION_DECLARATION = ProductDeclaration(
    name="annotations",
    kind=DataKind.STATISTICS,
    quantity="cam_annotation",
    fields=[
        "target_identity",
        "fields",
        "method_metadata",
    ],
    table="row_table",
)


# --- schema templates ------------------------------------------------------

#: Bundle type of a CAM product collection. The core ``generic/1`` adapter is
#: used on purpose: a CAM-specific bundle type would need a registered adapter,
#: and Phase 10E owns the engine handover that could register one.
CAM_BUNDLE_TYPE_ID = "generic.dataset_bundle/1"
CAM_BUNDLE_ADAPTER_ID = "generic/1"

#: Semantic product role restored by the bundle descriptor, keyed by the
#: ``product`` attribute of a product.
CAM_PRODUCT_ROLES = {
    "state": "state",
    "state_scan": "state",
    "candidates": "candidates",
}

#: Why the solution axis is not a branch identity. Stated once, in the schema,
#: so a consumer reading only the artifact cannot misread the axis.
SOLUTION_AXIS_SEMANTICS = (
    "the solution axis is a fixed-capacity slot, not a branch identity: slots "
    "are ordered independently at every scan point and carry no cross-parameter "
    "continuity"
)

#: What an imported ``legacy:`` identity does and does not promise.
LEGACY_IDENTITY_SEMANTICS = (
    "identity is stable inside this imported artifact only; it is derived from "
    "the legacy case and candidate slots, not from a solver-owned branch id"
)


def _state_axes() -> list[AxisSchema]:
    return [
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="solution", role=AxisRole.INDEX),
        AxisSchema(name="state_row", role=AxisRole.INDEX),
        AxisSchema(name="state_col", role=AxisRole.INDEX),
    ]


def _state_variables() -> list[VariableSchema]:
    return [
        VariableSchema(
            name="states",
            dtype="complex128",
            value_domain="complex",
            dims=("scan", "solution", "state_row", "state_col"),
            quantity="cam_steady_state",
        ),
        VariableSchema(
            name="residuals",
            dtype="float64",
            value_domain="real",
            dims=("scan", "solution"),
            quantity="cam_steady_state_residual",
        ),
        VariableSchema(
            name="success",
            dtype="bool",
            value_domain="real",
            dims=("scan", "solution"),
        ),
        VariableSchema(
            name="valid_mask",
            dtype="bool",
            value_domain="real",
            dims=("scan", "solution"),
        ),
        VariableSchema(
            name="solution_count",
            dtype="int64",
            value_domain="real",
            dims=("scan",),
        ),
    ]


def _state_attributes(product: str, table: str) -> dict[str, Any]:
    return {
        "table": table,
        "product": product,
        "scan_axis": "scan",
        "solution_axis": "solution",
        "state_matrix_axes": ["state_row", "state_col"],
        "solution_semantics": SOLUTION_AXIS_SEMANTICS,
        "ragged": False,
    }


#: One parameter point. Axis sizes stay open here: the builder closes them.
CAM_STATE_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_state_axes(),
    variables=_state_variables(),
    attributes=_state_attributes("state", "cam_state"),
)

#: The same product across a scan axis. The two differ only in their declared
#: product name and table, so a consumer never has to probe ``scan.size``.
CAM_STATE_SCAN_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_state_axes(),
    variables=_state_variables(),
    attributes=_state_attributes("state_scan", "cam_state_scan"),
)


def _candidate_axes() -> list[AxisSchema]:
    return [
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="group", role=AxisRole.INDEX),
        AxisSchema(name="group_offset", role=AxisRole.INDEX),
        AxisSchema(name="candidate", role=AxisRole.INDEX),
        AxisSchema(name="state_row", role=AxisRole.INDEX),
        AxisSchema(name="state_col", role=AxisRole.INDEX),
        AxisSchema(name="state_vector", role=AxisRole.INDEX),
        AxisSchema(name="control", role=AxisRole.INDEX),
    ]


def _candidate_variables(
    widths: Mapping[str, int] | None = None,
) -> list[VariableSchema]:
    """Return the candidate columns, with string widths from ``widths``.

    The template and every builder share this function so the frozen column
    list cannot drift from what a builder materializes; only the string width
    of a ``U`` column is data-dependent.
    """
    width_of = dict(widths or {})

    def text(name: str, dims: tuple[str, ...], quantity: str = "") -> VariableSchema:
        width = max(int(width_of.get(name, 1)), 1)
        return VariableSchema(
            name=name,
            dtype=f"U{width}",
            value_domain="real",
            dims=dims,
            quantity=quantity,
        )

    return [
        VariableSchema(
            name="group_scan_index",
            dtype="int64",
            value_domain="real",
            dims=("group",),
        ),
        VariableSchema(
            name="candidate_offsets",
            dtype="int64",
            value_domain="real",
            dims=("group_offset",),
        ),
        text("candidate_identity", ("candidate",), "cam_candidate_identity"),
        VariableSchema(
            name="states",
            dtype="complex128",
            value_domain="complex",
            dims=("candidate", "state_row", "state_col"),
            quantity="cam_candidate_state",
        ),
        VariableSchema(
            name="state_vectors",
            dtype="float64",
            value_domain="real",
            dims=("candidate", "state_vector"),
            quantity="cam_candidate_state_vector",
        ),
        VariableSchema(
            name="control_values",
            dtype="float64",
            value_domain="real",
            dims=("candidate", "control"),
            quantity="cam_candidate_control",
        ),
        text("control_names", ("control",)),
        VariableSchema(
            name="full_residual_norm",
            dtype="float64",
            value_domain="real",
            dims=("candidate",),
            quantity="cam_candidate_full_residual",
        ),
        VariableSchema(
            name="search_residual_norm",
            dtype="float64",
            value_domain="real",
            dims=("candidate",),
            quantity="cam_candidate_search_residual",
        ),
        VariableSchema(
            name="success",
            dtype="bool",
            value_domain="real",
            dims=("candidate",),
        ),
        text("status", ("candidate",)),
        text("method", ("candidate",)),
        VariableSchema(
            name="verification_digits",
            dtype="int64",
            value_domain="real",
            dims=("candidate",),
        ),
        text("verification_status", ("candidate",)),
        text("classification", ("candidate",)),
        text("base_audit", ("candidate",)),
        text("case_metadata", ("scan",)),
    ]


def _candidate_attributes() -> dict[str, Any]:
    return {
        "table": "cam_candidates",
        "product": "candidates",
        "scan_axis": "scan",
        "group_axis": "group",
        "group_offset_axis": "group_offset",
        "candidate_axis": "candidate",
        "ragged_axis": "candidate",
        "offsets_variable": "candidate_offsets",
        "identity_variable": "candidate_identity",
        "state_matrix_axes": ["state_row", "state_col"],
        "json_columns": ["classification", "base_audit", "case_metadata"],
        "identity_semantics": LEGACY_IDENTITY_SEMANTICS,
        "ragged": True,
    }


#: Ragged bifurcation/search candidates. Axis sizes stay open here.
CAM_CANDIDATE_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=_candidate_axes(),
    variables=_candidate_variables(),
    attributes=_candidate_attributes(),
)


# --- value conversion helpers ----------------------------------------------


def _json_value(value: Any) -> Any:
    """Return a JSON-serializable equivalent of one legacy value.

    Non-finite floats become ``None`` (JSON has no NaN) and complex values
    become an explicit ``{"real", "imag"}`` pair, so a JSON text column is
    always losslessly parseable.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays have no JSON representation")
        if value.ndim == 0:
            return _json_value(value.item())
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (complex, np.complexfloating)):
        scalar = complex(value)
        real = scalar.real if np.isfinite(scalar.real) else None
        imag = scalar.imag if np.isfinite(scalar.imag) else None
        return {"real": real, "imag": imag}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable")


def _json_mapping(mapping: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Convert a mapping for JSON, returning it plus the keys that failed."""
    converted: dict[str, Any] = {}
    unmapped: list[str] = []
    for key, value in mapping.items():
        try:
            converted[str(key)] = _json_value(value)
        except (TypeError, ValueError):
            unmapped.append(str(key))
    return converted, sorted(unmapped)


def _canonical_json_text(value: Any) -> str:
    """Return canonical JSON text for one column entry."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON column text is not valid JSON: {exc}") from exc
        return canonical_json(parsed)
    return canonical_json(_json_value(value))


def _string_column(values: Sequence[str]) -> np.ndarray:
    """Return a ``U`` array exactly wide enough for ``values``."""
    items = [str(value) for value in values]
    width = max((len(item) for item in items), default=0)
    return np.asarray(items, dtype=f"U{max(width, 1)}")


def _json_text_column(values: Sequence[Any]) -> np.ndarray:
    return _string_column([_canonical_json_text(value) for value in values])


def _string_widths(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Return the declared ``U`` width of every string column."""
    return {
        name: array.dtype.itemsize // 4
        for name, array in arrays.items()
        if array.dtype.kind == "U"
    }


def _closed_product(
    template: ProductSchema,
    *,
    sizes: Mapping[str, int],
    variables: Sequence[VariableSchema],
    coordinates: Sequence[CoordinateSchema],
    attributes: Mapping[str, Any],
) -> ProductSchema:
    """Return the template with every axis closed and payloads attached."""
    unknown = [axis.name for axis in template.axes if axis.name not in sizes]
    if unknown:
        raise ValueError(f"missing axis sizes for {sorted(unknown)}")
    return ProductSchema(
        kind=template.kind,
        axes=[
            axis.model_copy(update={"size": int(sizes[axis.name])})
            for axis in template.axes
        ],
        sampling_bases=list(template.sampling_bases),
        variables=list(variables),
        uncertainties=list(template.uncertainties),
        coordinates=list(coordinates),
        attributes=dict(attributes),
    )


def _grid_column(
    value: Any, *, shape: tuple[int, ...], count: int
) -> np.ndarray | None:
    """Flatten one per-point parameter value, or return None when it fits not."""
    array = np.asarray(value)
    if array.ndim == 0:
        column = np.full(count, array)
    elif tuple(array.shape) == shape or (array.ndim == 1 and array.size == count):
        column = array.reshape(count)
    else:
        return None
    dtype = "complex128" if np.iscomplexobj(column) else "float64"
    try:
        return np.asarray(column, dtype=dtype)
    except (TypeError, ValueError):
        return None


def _coordinate_columns(
    mapping: Mapping[str, Any] | None,
    *,
    prefix: str,
    shape: tuple[int, ...],
    count: int,
    reserved: set[str] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    list[VariableSchema],
    list[CoordinateSchema],
    list[str],
]:
    """Turn one parameter mapping into typed columns plus coordinates.

    Values that are not scalar or grid-shaped, values that are not numeric and
    names whose coordinate already exists are returned as unmapped names
    instead of being dropped.
    """
    arrays: dict[str, np.ndarray] = {}
    variables: list[VariableSchema] = []
    coordinates: list[CoordinateSchema] = []
    unmapped: list[str] = []
    taken = reserved if reserved is not None else set()
    for raw_name, value in (mapping or {}).items():
        name = str(raw_name)
        if name in taken:
            unmapped.append(name)
            continue
        column = _grid_column(value, shape=shape, count=count)
        if column is None:
            unmapped.append(name)
            continue
        variable_name = f"{prefix}{name}"
        complex_column = column.dtype.kind == "c"
        arrays[variable_name] = column
        variables.append(
            VariableSchema(
                name=variable_name,
                dtype="complex128" if complex_column else "float64",
                value_domain="complex" if complex_column else "real",
                dims=("scan",),
                quantity="scan_parameter",
            )
        )
        coordinates.append(
            CoordinateSchema(
                name=name,
                variable=variable_name,
                dims=("scan",),
                role="parameter",
            )
        )
        taken.add(name)
    return arrays, variables, coordinates, sorted(unmapped)


# --- builders --------------------------------------------------------------


def build_state_product(
    *,
    states: Any,
    residuals: Any,
    success: Any,
    valid_mask: Any,
    solution_count: Any,
    parameters: Mapping[str, Any] | None = None,
    axes: Mapping[str, Any] | None = None,
    grid_shape: Sequence[int] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build the state product of one parameter point or one scan.

    ``states`` spans ``grid_shape + (solution, state_row, state_col)``,
    ``residuals``/``success``/``valid_mask`` span ``grid_shape + (solution,)``
    and ``solution_count`` spans ``grid_shape``. An empty ``grid_shape`` is the
    single-point ``state`` product; anything else is ``state_scan``. The
    leading grid axes are flattened into the ``scan`` axis and ``grid_shape``
    is recorded as an attribute.

    A multi-dimensional grid is flattened rather than kept, because a product
    has exactly one parameter axis: the case grid is a *layout*, not data.

    Raises
    ------
    ValueError
        If a shape is inconsistent, a count leaves ``[0, capacity]``, or
        ``valid_mask`` does not flag exactly ``solution_count`` slots.

    """
    counts = np.asarray(solution_count, dtype=np.int64)
    shape = tuple(
        int(value) for value in (counts.shape if grid_shape is None else grid_shape)
    )
    if tuple(int(value) for value in counts.shape) != shape:
        raise ValueError(
            f"solution_count shape {tuple(counts.shape)} does not match the "
            f"grid shape {shape}"
        )
    scan_size = int(np.prod(shape, dtype=np.int64))
    residual_array = np.asarray(residuals, dtype=np.float64)
    if residual_array.ndim != len(shape) + 1:
        raise ValueError(
            f"residuals must span grid_shape + (solution,); got shape "
            f"{residual_array.shape} for grid shape {shape}"
        )
    capacity = int(residual_array.shape[-1])
    state_array = np.asarray(states, dtype=np.complex128)
    if state_array.ndim != len(shape) + 3:
        raise ValueError(
            f"states must span grid_shape + (solution, state_row, state_col); "
            f"got shape {state_array.shape} for grid shape {shape}"
        )
    matrix_shape = tuple(int(value) for value in state_array.shape[-2:])
    if state_array.shape != shape + (capacity, *matrix_shape):
        raise ValueError(
            f"states shape {state_array.shape} does not match grid shape "
            f"{shape}, solution capacity {capacity} and matrix shape "
            f"{matrix_shape}"
        )
    boolean_columns: dict[str, np.ndarray] = {}
    for name, value in (("success", success), ("valid_mask", valid_mask)):
        column = np.asarray(value, dtype=bool)
        if column.shape != shape + (capacity,):
            raise ValueError(
                f"{name} must span grid_shape + (solution,); got shape {column.shape}"
            )
        boolean_columns[name] = column
    counts_flat = counts.reshape(scan_size)
    valid_flat = boolean_columns["valid_mask"].reshape(scan_size, capacity)
    if counts_flat.size and (counts_flat.min() < 0 or counts_flat.max() > capacity):
        raise ValueError(f"solution_count must lie in [0, {capacity}]")
    if np.any(valid_flat.sum(axis=-1) != counts_flat):
        raise ValueError(
            "valid_mask must flag exactly solution_count slots at every scan point"
        )

    reserved: set[str] = set()
    parameter_arrays, parameter_variables, coordinates, unmapped_parameters = (
        _coordinate_columns(
            parameters,
            prefix="parameter.",
            shape=shape,
            count=scan_size,
            reserved=reserved,
        )
    )
    axis_arrays, axis_variables, axis_coordinates, unmapped_axes = _coordinate_columns(
        axes,
        prefix="axis.",
        shape=shape,
        count=scan_size,
        reserved=reserved,
    )
    coordinates = [*coordinates, *axis_coordinates]

    arrays: dict[str, np.ndarray] = {
        "states": state_array.reshape(scan_size, capacity, *matrix_shape),
        "residuals": residual_array.reshape(scan_size, capacity),
        "success": boolean_columns["success"].reshape(scan_size, capacity),
        "valid_mask": valid_flat,
        "solution_count": counts_flat,
        **parameter_arrays,
        **axis_arrays,
    }
    template = CAM_STATE_PRODUCT if not shape else CAM_STATE_SCAN_PRODUCT
    attributes = {
        **template.attributes,
        "grid_shape": [int(value) for value in shape],
        "scan_size": scan_size,
        "solution_capacity": capacity,
        "state_shape": [int(value) for value in matrix_shape],
    }
    schema = _closed_product(
        template,
        sizes={
            "scan": scan_size,
            "solution": capacity,
            "state_row": matrix_shape[0],
            "state_col": matrix_shape[1],
        },
        variables=[*_state_variables(), *parameter_variables, *axis_variables],
        coordinates=coordinates,
        attributes=attributes,
    )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.cam",
        provenance={
            **dict(provenance or {}),
            "unmapped_parameter_names": unmapped_parameters,
            "unmapped_axis_names": unmapped_axes,
        },
    )


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
    column = _string_column(list(values))
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
    seen: dict[str, int] = {}
    for value in identity:
        seen[value] = seen.get(value, 0) + 1
    duplicates = sorted(value for value, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            "candidate_identity must be unique across the product; duplicated: "
            f"{duplicates}"
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
    axis_arrays, axis_variables, coordinates, unmapped_case_axes = _coordinate_columns(
        case_axes,
        prefix="parameter.",
        shape=tuple(case_shape_list),
        count=n_case,
        reserved=reserved,
    )
    # ``case_parameter.*`` columns describe a case, not a parameter axis, so
    # they stay out of ``coordinates`` and out of the coordinate namespace.
    parameter_arrays, parameter_variables, _, unmapped_case_params = (
        _coordinate_columns(
            case_params,
            prefix="case_parameter.",
            shape=(n_case,),
            count=n_case,
            reserved=set(),
        )
    )

    arrays: dict[str, np.ndarray] = {
        "group_scan_index": group_index,
        "candidate_offsets": offsets,
        "candidate_identity": _string_column(identity),
        "states": state_array,
        "state_vectors": vector_array,
        "control_values": control_array,
        "control_names": _string_column(names),
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
        "classification": _json_text_column(
            _json_rows(classification, n_candidate, "classification")
        ),
        "base_audit": _json_text_column(
            _json_rows(base_audit, n_candidate, "base_audit")
        ),
        "case_metadata": _json_text_column(
            _json_rows(case_metadata, n_case, "case_metadata")
        ),
        **axis_arrays,
        **parameter_arrays,
    }
    variables = [
        *_candidate_variables(_string_widths(arrays)),
        *axis_variables,
        *parameter_variables,
    ]
    attributes = {
        **CAM_CANDIDATE_PRODUCT.attributes,
        "case_shape": case_shape_list,
        "case_count": n_case,
        "candidate_count": n_candidate,
    }
    schema = _closed_product(
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


# --- bundle ----------------------------------------------------------------


def cam_bundle(
    products: Mapping[str, Dataset],
    *,
    provenance: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GenericDataBundle:
    """Group CAM products into the core generic bundle.

    No CAM-specific bundle class is introduced: a second result type would
    duplicate what ``GenericDataBundle`` already restores, and 10E is the phase
    that owns any engine-level bundle. The semantic role of each product comes
    from its own ``product`` attribute, so the bundle stays resource-generic
    while ``descriptor.product_roles`` still answers "which product is the
    state?".
    """
    roles: dict[str, str] = {}
    for name, product in products.items():
        product_kind = str(product.attributes.get("product", name))
        roles[CAM_PRODUCT_ROLES.get(product_kind, name)] = name
    descriptor = BundleDescriptor(
        type_id=CAM_BUNDLE_TYPE_ID,
        adapter_id=CAM_BUNDLE_ADAPTER_ID,
        descriptor_schema=CAM_BUNDLE_TYPE_ID,
        descriptor={},
        product_roles=roles,
    )
    return GenericDataBundle(
        products,
        descriptor,
        provenance=provenance,
        metadata=metadata,
    )


# --- legacy import ---------------------------------------------------------


def _legacy_token(result: Any) -> str:
    """Return a deterministic, delimiter-free source token for identities."""
    meta = getattr(result, "meta", None)
    label = meta.get("label") if isinstance(meta, Mapping) else None
    base = label if isinstance(label, str) and label.strip() else None
    if base is None:
        base = str(getattr(result, "result_kind", "legacy"))
    token = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in base.strip()
    )
    return token or "legacy"


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


def _is_classification_key(name: str) -> bool:
    lowered = name.lower()
    return "signature" in lowered or "classif" in lowered


def _split_diagnostics(
    diagnostics: Mapping[str, Any] | None, n_candidate: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Split per-candidate diagnostics into classification and audit rows.

    Keys whose name mentions a scaling signature or a classification decision
    go to ``classification``; every other per-candidate key goes to
    ``base_audit``. A key that cannot be expressed per candidate (object array,
    wrong leading length, unmappable nesting) is reported by name instead of
    being dropped.
    """
    classification: list[dict[str, Any]] = [dict() for _ in range(n_candidate)]
    base_audit: list[dict[str, Any]] = [dict() for _ in range(n_candidate)]
    unmapped: list[str] = []
    for raw_name, value in (diagnostics or {}).items():
        name = str(raw_name)
        array = np.asarray(value)
        if array.dtype.hasobject or array.ndim == 0 or array.shape[0] != n_candidate:
            unmapped.append(name)
            continue
        rows = classification if _is_classification_key(name) else base_audit
        try:
            for position in range(n_candidate):
                rows[position][name] = _json_value(array[position])
        except (TypeError, ValueError):
            unmapped.append(name)
    return classification, base_audit, sorted(unmapped)


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
    branches = getattr(result, "branches", None)
    if branches is None:
        return []
    return sorted(str(name) for name in branches.__dataclass_fields__)


def _trusted_npz(path: str | Path) -> Any:
    """Open one trusted pre-2.0 NPZ payload.

    ``allow_pickle`` is deliberate and bounded: a pre-2.0 artifact stores its
    ``params``/``meta``/``diagnostics`` mappings as object arrays, and this
    importer exists to read exactly those historical files. Every value it
    keeps afterwards is typed; only the trusted-input step unpickles.
    """
    return np.load(_legacy_path(path), allow_pickle=True)


def _legacy_path(path: str | Path) -> Path:
    """Resolve the legacy ``.npz`` payload of an unsuffixed result path."""
    target = Path(path)
    if not target.exists() and target.suffix != ".npz":
        return target.with_suffix(".npz")
    return target


def import_cam_result(
    source: CAMResult | str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> GenericDataBundle:
    """Import one :class:`~qphase_cam.result.CAMResult`.

    The legacy ``postprocess`` mapping is not mapped: it is reported by key
    name in ``unmapped_legacy_postprocess``. ``params`` and ``axes`` entries
    that are neither scalar nor grid-shaped are reported in
    ``unmapped_legacy_parameters``/``unmapped_legacy_axes``.
    """
    from .result import CAMResult

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
    meta, unmapped_meta = _json_mapping(result.meta or {})
    name = str(dataset.attributes["product"])
    return cam_bundle(
        {name: dataset},
        provenance={
            **dict(provenance or {}),
            "legacy_source": "qphase_cam.result:CAMResult",
            "legacy_result_kind": result.result_kind,
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
) -> GenericDataBundle:
    """Import one :class:`~qphase_cam.bifurcation_result.CAMBifurcationResult`.

    A single case is normalized to a scan of size one, so both importers
    produce the same schema. Candidate identities are
    ``legacy:<source>:0:<candidate>``, with ``<source>`` the result label or
    the result kind.
    """
    from .bifurcation_result import CAMBifurcationResult

    result = (
        CAMBifurcationResult.load(_legacy_path(source))
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(result, CAMBifurcationResult):
        raise TypeError(f"expected a CAMBifurcationResult, got {type(result).__name__}")
    token = _legacy_token(result)
    n_candidate = _legacy_candidate_count(result)
    arrays = _legacy_candidate_arrays(result)
    classification, base_audit, unmapped_diagnostics = _split_diagnostics(
        result.diagnostics, n_candidate
    )
    dataset = build_candidate_product(
        candidate_identity=[
            f"legacy:{token}:0:{position}" for position in range(n_candidate)
        ],
        classification=classification,
        base_audit=base_audit,
        provenance={
            "legacy_source": ("qphase_cam.bifurcation_result:CAMBifurcationResult")
        },
        **arrays,
    )
    meta, unmapped_meta = _json_mapping(result.meta or {})
    return cam_bundle(
        {"candidates": dataset},
        provenance={
            **dict(provenance or {}),
            "legacy_source": ("qphase_cam.bifurcation_result:CAMBifurcationResult"),
            "legacy_result_kind": result.result_kind,
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
) -> GenericDataBundle:
    """Import one ragged scan of bifurcation candidates.

    The case grid is flattened into the ``scan`` axis and the original shape is
    kept as the ``case_shape`` attribute. ``group_scan_index`` is the flattened
    case index, so a consumer can rebuild a case's candidates from
    ``candidate_offsets`` alone.
    """
    from .bifurcation_result import CAMBifurcationScanResult

    result = (
        CAMBifurcationScanResult.load(_legacy_path(source))
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(result, CAMBifurcationScanResult):
        raise TypeError(
            f"expected a CAMBifurcationScanResult, got {type(result).__name__}"
        )
    token = _legacy_token(result)
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
    classification, base_audit, unmapped_diagnostics = _split_diagnostics(
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
            "legacy_source": ("qphase_cam.bifurcation_result:CAMBifurcationScanResult")
        },
        **arrays,
    )
    product_provenance = dict(dataset.provenance)
    scan_meta, unmapped_scan_meta = _json_mapping(result.meta or {})
    candidate_meta, unmapped_candidate_meta = _json_mapping(
        result.candidates.meta or {}
    )
    return cam_bundle(
        {"candidates": dataset},
        provenance={
            **dict(provenance or {}),
            "legacy_source": ("qphase_cam.bifurcation_result:CAMBifurcationScanResult"),
            "legacy_result_kind": result.result_kind,
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


def import_legacy_artifact(
    path: str | Path,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> GenericDataBundle:
    """Import a trusted pre-2.0 NPZ payload, dispatching on its keys.

    A legacy artifact records its own result type only in the file layout, so
    the dispatcher reads the key set: ``valid_mask`` is a fixed-point result,
    ``case_axes`` a candidate scan, ``control_names`` a single candidate set.
    """
    with _trusted_npz(path) as data:
        keys = set(data.files)
    target = _legacy_path(path)
    if "valid_mask" in keys:
        return import_cam_result(target, provenance=provenance)
    if "case_axes" in keys:
        return import_bifurcation_scan(target, provenance=provenance)
    if "control_names" in keys:
        return import_bifurcation_result(target, provenance=provenance)
    raise ValueError(
        f"{target} is not a recognizable pre-2.0 CAM artifact; keys: {sorted(keys)}"
    )


# --- export comparison -----------------------------------------------------


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
    from .bifurcation_result import (
        CAMBifurcationResult,
        CAMBifurcationScanResult,
    )

    arrays = _dataset_arrays(product)
    report = _comparison_report("candidates")
    report["unmapped"] = []
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

    classification, base_audit, _unmapped = _split_diagnostics(
        candidates.diagnostics, n_candidate
    )
    for name, expected_rows in (
        ("classification", classification),
        ("base_audit", base_audit),
    ):
        actual_rows = [json.loads(str(value)) for value in arrays[name]]
        expected_text = [_canonical_json_text(row) for row in expected_rows]
        actual_text = [_canonical_json_text(row) for row in actual_rows]
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
        [_canonical_json_text(row) for row in actual_case_metadata]
        == [_canonical_json_text(row) for row in case_metadata],
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
