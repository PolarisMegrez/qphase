"""qphase_sde: validate a frozen prediction against a measured observation
---------------------------------------------------------
A prediction is a statement about a readout: what an approximation implies for a
frequency response, an Allan intensity or an information rate. This module answers
the next question — does the measured data agree with that statement, within a
tolerance the caller states up front?

The service is deliberately small and ordinary:

* both inputs are *restored artifacts*. The prediction is a typed CAM prediction
  table (``product="prediction"``) together with the manifest it was published
  under, and the observation is a restored :class:`~qphase_sde.result.SDEDataBundle`.
  The manifest is passed explicitly, so the prediction's identity and parent chain
  are evidence this call reads, never something it re-derives from a path;
* every comparison names the prediction row, the prediction metric, the observed
  product/field and the exact placement inside the observed array. A multi-valued
  axis is never truncated to its first element and never averaged: an unresolved
  axis is an error, not a silent choice;
* the verdict is either ``valid`` or ``threshold_rejected``. A number outside the
  tolerance is a formal negative result — not an exception, and never a rewrite of
  the prediction it tested;
* the prediction product is only read. Its schema, its values and its
  ``prediction_id`` are unchanged by the call and by persisting the result.

Tolerance convention (recorded in the product attributes): with
``scale = max(|expected|, |observed|)``, ``absolute_deviation = |observed - expected|``
and ``effective_tolerance = max(absolute_tolerance, relative_tolerance * scale)``, a row
is valid exactly when ``absolute_deviation <= effective_tolerance``. The comparison is
in the observable's own units on both sides; ``relative_deviation = absolute_deviation /
scale`` (0 when ``scale == 0``) is reported as a dimensionless diagnostic only and is
never compared with a tolerance. No division by zero and no NaN can enter the table.

The returned bundle carries the validation table as its only product, keeps the
observation's :class:`~qphase_sde.contracts.bundle.SDEProvenance` unchanged, states
no artifact id of its own (it is a new derived result) and records both inputs in its
parent chain — the prediction manifest's parents followed by the prediction artifact,
then the observation's parents followed by the observation artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from qphase.core.utils import canonical_json
from qphase.data import (
    ArtifactManifest,
    AxisRole,
    AxisSchema,
    DataKind,
    Dataset,
    ProductSchema,
    StatisticsDataset,
    VariableSchema,
)

from .contracts.bundle import SDEProvenance
from .result import SDEDataBundle

__all__ = [
    "PREDICTION_VALIDATION_METRICS",
    "PREDICTION_VALIDATION_SCHEMA",
    "PredictionComparison",
    "PredictionMetric",
    "PredictionValidationStatus",
    "validate_prediction",
]

#: Schema marker of the validation product this module emits.
PREDICTION_VALIDATION_SCHEMA = "qphase_sde.prediction-validation/1"

#: Prediction metrics this service compares. The set is closed: a comparison names
#: one of these columns or fails. Every entry is a numeric CAM prediction column.
PredictionMetric = Literal[
    "expected_frequency_response_full",
    "allan_intensity_reconstructed",
    "full_information_rate",
]

#: The metrics as a runtime tuple (closed vocabulary, no free-form field names).
PREDICTION_VALIDATION_METRICS: tuple[str, ...] = (
    "expected_frequency_response_full",
    "allan_intensity_reconstructed",
    "full_information_rate",
)

#: The only two verdicts one validation row can carry.
PredictionValidationStatus = Literal["valid", "threshold_rejected"]

#: The scan axis name SDE products carry before they are sliced to one point.
_SCAN_AXIS = "scan"

#: Deviation and tolerance semantics, recorded on the product.
_TOLERANCE_SEMANTICS = (
    "scale = max(|expected|, |observed|); absolute_deviation = |observed - expected|; "
    "effective_tolerance = max(absolute_tolerance, relative_tolerance * scale); a row "
    "is valid when absolute_deviation <= effective_tolerance and threshold_rejected "
    "otherwise; relative_deviation = absolute_deviation / scale (0 when scale is 0) is "
    "a dimensionless diagnostic and is never compared with a tolerance"
)

_SELECTION_SEMANTICS = (
    "every observed value is resolved by an explicit index per remaining axis: the "
    "scan axis from scan_index, the other axes from selection, and a single-valued "
    "axis at its only index; a multi-valued axis without an explicit index is an "
    "error, never a first-element or averaged read"
)


class PredictionComparison(BaseModel):
    """One explicit prediction-versus-observation comparison.

    ``scan_index`` follows the observation grid, exactly like
    :meth:`~qphase_sde.result.SDEDataBundle.point_view`: ``None`` for a single-point
    observation, an ``int`` or one-element tuple for a one-dimensional scan, and a
    tuple of the grid rank for a multi-dimensional scan. ``selection`` names the
    value to read along every other axis of the observed field (``channel``, ``tau``,
    ...); an axis that is neither the scan axis nor selected must be single-valued.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str = Field(min_length=1)
    metric: PredictionMetric
    observed_product: str = Field(min_length=1)
    observed_field: str = Field(min_length=1)
    absolute_tolerance: float = Field(ge=0.0, allow_inf_nan=False)
    relative_tolerance: float = Field(ge=0.0, allow_inf_nan=False)
    scan_index: StrictInt | tuple[StrictInt, ...] | None = None
    selection: Mapping[str, StrictInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_selection(self) -> PredictionComparison:
        for name, value in self.selection.items():
            if not str(name).strip():
                raise ValueError("selection axis names must not be empty")
            if int(value) < 0:
                raise ValueError(
                    f"selection index for {name!r} must be non-negative; got {value}"
                )
        index = self.scan_index
        if isinstance(index, int) and index < 0:
            raise ValueError(f"scan_index must be non-negative; got {index}")
        if isinstance(index, tuple):
            for value in index:
                if value < 0:
                    raise ValueError(
                        f"scan_index entries must be non-negative; got {index}"
                    )
        return self

    def normalized_scan_index(self) -> tuple[int, ...]:
        """Return the scan index as a tuple (empty when it was not given)."""
        index = self.scan_index
        if index is None:
            return ()
        if isinstance(index, int):
            return (int(index),)
        return tuple(int(value) for value in index)

    def selection_items(self) -> tuple[tuple[str, int], ...]:
        """Return the selection as an ordered tuple of ``(axis, index)`` pairs."""
        return tuple(
            (str(name), int(value)) for name, value in sorted(self.selection.items())
        )


def validate_prediction(
    prediction: Dataset,
    manifest: ArtifactManifest,
    observation: SDEDataBundle,
    comparisons: Sequence[PredictionComparison],
    *,
    prediction_label: str = "prediction",
    product_label: str = "prediction_validation",
) -> SDEDataBundle:
    """Validate the selected prediction rows against one restored observation.

    ``prediction`` is the restored CAM prediction table, ``manifest`` the artifact
    manifest that table was published under, ``observation`` the restored SDE
    observation and ``comparisons`` the explicit comparisons to run.
    ``prediction_label`` is the product name the prediction artifact publishes (the
    manifest must hold that entry with exactly this schema). The result is a bundle
    holding one ``prediction_validation`` product; it states no artifact id of its own
    and names both inputs in its parent chain.
    """
    if not isinstance(observation, SDEDataBundle):
        raise TypeError(
            "the observation must be a restored SDEDataBundle; got "
            f"{type(observation).__name__}"
        )
    rows = _prediction_rows(prediction)
    prediction_artifact_id = _prediction_artifact_id(
        prediction, manifest, label=prediction_label
    )
    observation_artifact_id = _observation_artifact_id(observation)
    provenance = observation.provenance
    if not isinstance(provenance, SDEProvenance):
        raise ValueError(
            "the observation is not an SDEDataBundle: its provenance is not an "
            f"SDEProvenance ({type(provenance).__name__})"
        )
    requests = _comparison_requests(comparisons)
    entries = [
        _compare(
            request,
            rows=rows,
            prediction=prediction,
            observation=observation,
            prediction_artifact_id=prediction_artifact_id,
            observation_artifact_id=observation_artifact_id,
        )
        for request in requests
    ]
    parents = _parent_chain(
        prediction_parents=manifest.parents,
        prediction_artifact_id=prediction_artifact_id,
        observation_parents=observation.parents,
        observation_artifact_id=observation_artifact_id,
    )
    product = _validation_product(
        entries,
        prediction_artifact_id=prediction_artifact_id,
        observation_artifact_id=observation_artifact_id,
    )
    rejected = sum(1 for entry in entries if entry["status"] != "valid")
    meta = {
        "mode": "prediction_validation",
        "product": product_label,
        "method": {
            "service": "qphase_sde.prediction.validate_prediction",
            "comparison": "explicit prediction row against one observed scalar",
            "tolerance": _TOLERANCE_SEMANTICS,
            "selection": _SELECTION_SEMANTICS,
            "scope": (
                "a formal tolerance verdict on a frozen prediction; it never edits "
                "or re-runs the prediction"
            ),
        },
        "prediction_artifact_id": prediction_artifact_id,
        "observation_artifact_id": observation_artifact_id,
        "comparison_count": len(entries),
        "valid_count": len(entries) - rejected,
        "threshold_rejected_count": rejected,
        "prediction_ids": sorted({entry["prediction_id"] for entry in entries}),
    }
    return SDEDataBundle(
        {product_label: product},
        provenance,
        meta=meta,
        parents=parents,
    )


# -- prediction table ---------------------------------------------------------


def _prediction_rows(prediction: Dataset) -> dict[str, int]:
    """Return the ``prediction_id -> row`` index of one prediction product."""
    attributes = prediction.attributes
    if (
        attributes.get("product") != "prediction"
        or attributes.get("table") != "cam_prediction"
    ):
        raise ValueError(
            "validate_prediction takes a CAM prediction product "
            "(product='prediction', table='cam_prediction'); got "
            f"product={attributes.get('product')!r}, "
            f"table={attributes.get('table')!r}"
        )
    names = {variable.name for variable in prediction.variables}
    missing = {
        "prediction_id",
        "target_identity",
        "completeness",
        "status",
        *PREDICTION_VALIDATION_METRICS,
    } - names
    if missing:
        raise ValueError(
            f"the prediction product is missing columns: {sorted(missing)}"
        )
    identifiers = [
        str(value)
        for value in np.asarray(prediction.handle("prediction_id").materialize())
    ]
    duplicates = sorted(
        {value for value in identifiers if identifiers.count(value) > 1}
    )
    if duplicates:
        raise ValueError(
            f"the prediction product repeats prediction ids: {duplicates}"
        )
    return {identifier: row for row, identifier in enumerate(identifiers)}


def _prediction_artifact_id(
    prediction: Dataset, manifest: ArtifactManifest, *, label: str
) -> str:
    """Return the prediction artifact id, proving the manifest published it.

    A manifest is evidence only if it lists this product under the requested name
    with exactly this schema; an id match alone would accept a foreign or forged
    manifest.
    """
    published = str(manifest.artifact_id or "").strip()
    if not published:
        raise ValueError("the prediction manifest carries no artifact id")
    if not manifest.products:
        raise ValueError(
            f"the manifest {published!r} lists no products; it cannot have published "
            "the prediction table"
        )
    available = {entry.name: entry for entry in manifest.products}
    entry = available.get(label)
    if entry is None:
        raise ValueError(
            f"the manifest {published!r} publishes no product {label!r}; available: "
            f"{sorted(available)}"
        )
    if entry.product_schema != prediction.schema:
        raise ValueError(
            f"the manifest product {label!r} does not match the restored prediction "
            "schema; the dataset and the manifest describe different products"
        )
    stated = prediction.provenance.get("product")
    if stated is not None and str(stated) != label:
        raise ValueError(
            f"the dataset provenance names product {str(stated)!r} but the manifest "
            f"entry read is {label!r}"
        )
    declared = prediction.provenance.get("artifact_id")
    if declared is not None and str(declared) != published:
        raise ValueError(
            f"the prediction product states artifact id {str(declared)!r} but the "
            f"given manifest publishes {published!r}"
        )
    return published


def _observation_artifact_id(observation: SDEDataBundle) -> str:
    """Return the observation artifact id; a loose bundle cannot validate."""
    declared = getattr(observation, "declared_artifact_id", None)
    identity = declared() if callable(declared) else None
    if not identity:
        raise ValueError(
            "the observation bundle states no artifact id; restore it from its "
            "persisted artifact before validating a prediction"
        )
    return str(identity)


def _parent_chain(
    *,
    prediction_parents: Sequence[str],
    prediction_artifact_id: str,
    observation_parents: Sequence[str],
    observation_artifact_id: str,
) -> tuple[str, ...]:
    """Return the ordered, deduplicated chain of both inputs."""
    chain: list[str] = []
    for value in (
        *prediction_parents,
        prediction_artifact_id,
        *observation_parents,
        observation_artifact_id,
    ):
        text = str(value)
        if text and text not in chain:
            chain.append(text)
    return tuple(chain)


# -- comparisons --------------------------------------------------------------


def _comparison_requests(
    comparisons: Sequence[PredictionComparison],
) -> tuple[PredictionComparison, ...]:
    requests = tuple(comparisons)
    if not requests:
        raise ValueError("validate_prediction needs at least one comparison")
    for request in requests:
        if not isinstance(request, PredictionComparison):
            raise TypeError(
                "validate_prediction takes PredictionComparison objects; got "
                f"{type(request).__name__}"
            )
    return requests


def _compare(
    request: PredictionComparison,
    *,
    rows: Mapping[str, int],
    prediction: Dataset,
    observation: SDEDataBundle,
    prediction_artifact_id: str,
    observation_artifact_id: str,
) -> dict[str, Any]:
    """Return one validation row for one comparison."""
    row = rows.get(request.prediction_id)
    if row is None:
        raise ValueError(
            f"unknown prediction id {request.prediction_id!r}; the product holds "
            f"{len(rows)} rows"
        )
    completeness = str(
        np.asarray(prediction.handle("completeness").materialize())[row]
    )
    if completeness != "complete":
        raise ValueError(
            f"prediction {request.prediction_id!r} is {completeness!r}; only a "
            "complete prediction can be validated against an observation"
        )
    status = str(np.asarray(prediction.handle("status").materialize())[row])
    if status != "valid":
        raise ValueError(
            f"prediction {request.prediction_id!r} has status {status!r}; only a "
            "valid prediction can be validated against an observation"
        )
    expected = float(
        np.asarray(prediction.handle(request.metric).materialize())[row]
    )
    if not np.isfinite(expected):
        raise ValueError(
            f"prediction {request.prediction_id!r} has no finite "
            f"{request.metric} to compare"
        )
    target_identity = str(
        np.asarray(prediction.handle("target_identity").materialize())[row]
    )
    observed, selection = _observed_scalar(
        observation, request, scan_index=request.normalized_scan_index()
    )
    scale = max(abs(expected), abs(observed))
    absolute_deviation = abs(observed - expected)
    relative_deviation = absolute_deviation / scale if scale > 0.0 else 0.0
    effective_tolerance = max(
        request.absolute_tolerance, request.relative_tolerance * scale
    )
    verdict: PredictionValidationStatus = (
        "valid"
        if absolute_deviation <= effective_tolerance
        else "threshold_rejected"
    )
    return {
        "prediction_id": request.prediction_id,
        "target_identity": target_identity,
        "prediction_artifact_id": prediction_artifact_id,
        "observation_artifact_id": observation_artifact_id,
        "metric": request.metric,
        "observed_product": request.observed_product,
        "observed_field": request.observed_field,
        "selection": selection,
        "expected": expected,
        "observed": observed,
        "absolute_deviation": absolute_deviation,
        "relative_deviation": relative_deviation,
        "effective_tolerance": effective_tolerance,
        "absolute_tolerance": request.absolute_tolerance,
        "relative_tolerance": request.relative_tolerance,
        "status": verdict,
    }


def _observed_scalar(
    observation: SDEDataBundle,
    request: PredictionComparison,
    *,
    scan_index: tuple[int, ...],
) -> tuple[float, dict[str, Any]]:
    """Return the observed scalar and the selection that produced it."""
    products = observation.products
    product = products.get(request.observed_product)
    if product is None:
        raise ValueError(
            f"the observation has no product {request.observed_product!r}; "
            f"available: {sorted(products)}"
        )
    variables = {variable.name: variable for variable in product.variables}
    variable = variables.get(request.observed_field)
    if variable is None:
        raise ValueError(
            f"product {request.observed_product!r} has no field "
            f"{request.observed_field!r}; available: {sorted(variables)}"
        )
    if np.dtype(variable.dtype).kind not in "iuf":
        raise ValueError(
            f"observed field {request.observed_field!r} is not numeric "
            f"(dtype {variable.dtype})"
        )
    # The declared dims and axis sizes are enough to place the read: nothing is
    # loaded until the point selection below asks the storage layer for it.
    extents: dict[str, int] = {}
    for dim in variable.dims:
        size = product.axis(dim).size
        if size is None:
            raise ValueError(
                f"observed field {request.observed_field!r} spans axis {dim!r} whose "
                "size is not declared; the product cannot be point-selected"
            )
        extents[dim] = int(size)
    scan_shape = tuple(int(value) for value in observation.shape)
    if scan_index:
        if not scan_shape:
            allowed = (0,)
            if scan_index not in (allowed, ()):
                raise ValueError(
                    "the observation has no scan axes; scan_index must be unset or "
                    f"(0,) for a single-point observation; got {scan_index}"
                )
        elif len(scan_index) == len(scan_shape):
            for axis, (value, extent) in enumerate(
                zip(scan_index, scan_shape, strict=True)
            ):
                if value >= extent:
                    raise ValueError(
                        f"scan_index[{axis}]={value} is out of bounds for the scan "
                        f"axis of extent {extent}"
                    )
        elif len(scan_shape) == 1 and len(scan_index) == 1:
            if scan_index[0] >= scan_shape[0]:
                raise ValueError(
                    f"scan_index={scan_index[0]} is out of bounds for the scan axis "
                    f"of extent {scan_shape[0]}"
                )
        else:
            raise ValueError(
                f"scan_index {scan_index} does not match the observation scan shape "
                f"{scan_shape}; give one index per scan axis"
            )
        flat_scan = (
            int(np.ravel_multi_index(scan_index, scan_shape))
            if len(scan_index) == len(scan_shape) and scan_shape
            else int(scan_index[0])
        )
    else:
        flat_scan = 0
    selection = dict(request.selection_items())
    chosen: dict[str, int] = {}
    coordinates: dict[str, Any] = {}
    for dim in variable.dims:
        extent = extents[dim]
        if dim == _SCAN_AXIS:
            if not scan_index and extent > 1:
                raise ValueError(
                    f"observed field {request.observed_field!r} has {extent} points "
                    "along the scan axis; select one with scan_index"
                )
            if not scan_shape and flat_scan != 0:
                raise ValueError(
                    "the observed field has no scan dimension for the given "
                    "scan_index"
                )
            if scan_shape and flat_scan >= extent:
                raise ValueError(
                    f"scan_index {scan_index} is out of bounds for the observed "
                    f"field scan dimension of extent {extent}"
                )
            chosen[dim] = flat_scan
            # The scan placement is already carried by scan_index and the scan
            # coordinates; the selection column records only the axes chosen here.
            continue
        if dim in selection:
            index = selection.pop(dim)
            if index >= extent:
                raise ValueError(
                    f"selection index for {dim!r} is {index}, out of bounds for "
                    f"extent {extent}"
                )
            chosen[dim] = index
            coordinates[dim] = index
            continue
        if extent == 1:
            chosen[dim] = 0
            coordinates[dim] = 0
            continue
        raise ValueError(
            f"observed field {request.observed_field!r} has {extent} values along "
            f"axis {dim!r}; select it explicitly with selection[{dim!r}]"
        )
    if selection:
        raise ValueError(
            f"selection names axes the observed field does not have: "
            f"{sorted(selection)}; its axes are {list(variable.dims)}"
        )
    # Point-select through the existing dataset view: the point selection asks the
    # storage handles for the touched chunks only (`materialize_selection`), so a
    # point of a large spectral product never loads the whole field.
    point = product.materialize().point_view(**chosen)
    scalar_variable = point.variable(request.observed_field)
    if scalar_variable.dims:
        raise ValueError(
            f"observed field {request.observed_field!r} still spans "
            f"{list(scalar_variable.dims)} after the point selection; its declared "
            "dims do not match the stored payload"
        )
    selected = np.asarray(point.handle(request.observed_field).materialize())
    if selected.ndim != 0:
        raise ValueError(
            f"observed field {request.observed_field!r} does not resolve to a "
            "scalar at the requested indices (materialized shape "
            f"{tuple(selected.shape)})"
        )
    number = float(selected)
    if not np.isfinite(number):
        raise ValueError(
            f"observed field {request.observed_field!r} is not finite at the "
            f"requested indices ({number!r})"
        )
    payload = {
        "scan_index": list(scan_index),
        "scan_coordinates": _scan_coordinates(observation, scan_index),
        "selection": dict(sorted(coordinates.items())),
    }
    return number, payload


def _scan_coordinates(
    observation: SDEDataBundle, scan_index: tuple[int, ...]
) -> dict[str, float]:
    """Return the observed scan coordinates of the selected point.

    A scan axis held as one coordinate per point is read at the flat index; a grid
    axis held as its axis values is read at the grid position of that axis.
    """
    if not scan_index:
        return {}
    shape = tuple(int(value) for value in observation.shape)
    if not shape:
        return {}
    size = int(np.prod(shape)) if shape else 1
    flat = (
        int(np.ravel_multi_index(scan_index, shape))
        if len(scan_index) == len(shape)
        else int(scan_index[0])
    )
    grid_index = (
        np.unravel_index(flat, shape) if len(scan_index) == len(shape) else (flat,)
    )
    coordinates: dict[str, float] = {}
    for position, (name, values) in enumerate(observation.axes.items()):
        array = np.asarray(values).reshape(-1)
        if array.size == size and flat < array.size:
            coordinates[str(name)] = float(array[flat])
        elif position < len(grid_index) and array.size == shape[position]:
            coordinates[str(name)] = float(array[grid_index[position]])
    return coordinates


# -- product ------------------------------------------------------------------


def _validation_product(
    entries: list[dict[str, Any]],
    *,
    prediction_artifact_id: str,
    observation_artifact_id: str,
) -> Dataset:
    """Build the typed validation table of one call."""
    count = len(entries)
    rejected = sum(1 for entry in entries if entry["status"] != "valid")
    text_columns = (
        "prediction_id",
        "target_identity",
        "prediction_artifact_id",
        "observation_artifact_id",
        "metric",
        "observed_product",
        "observed_field",
        "selection",
        "status",
    )
    numeric_columns = (
        "expected",
        "observed",
        "absolute_deviation",
        "relative_deviation",
        "effective_tolerance",
        "absolute_tolerance",
        "relative_tolerance",
    )
    arrays: dict[str, Any] = {}
    variables: list[VariableSchema] = []
    for name in text_columns:
        values = [
            canonical_json(entry[name])
            if name == "selection"
            else str(entry[name])
            for entry in entries
        ]
        width = max((len(value) for value in values), default=1)
        arrays[name] = np.asarray(values, dtype=f"U{max(width, 1)}")
        variables.append(
            VariableSchema(
                name=name,
                dtype=np.dtype(f"U{max(width, 1)}").str,
                value_domain="real",
                dims=("validation",),
            )
        )
    for name in numeric_columns:
        arrays[name] = np.asarray(
            [float(entry[name]) for entry in entries], dtype=float
        )
        variables.append(
            VariableSchema(
                name=name,
                dtype="float64",
                value_domain="real",
                dims=("validation",),
            )
        )
    attributes = {
        "schema": PREDICTION_VALIDATION_SCHEMA,
        "product": "prediction_validation",
        "table": "prediction_validation",
        "validation_axis": "validation",
        "validation_count": count,
        "valid_count": count - rejected,
        "threshold_rejected_count": rejected,
        "metrics": list(PREDICTION_VALIDATION_METRICS),
        "status_vocabulary": ["threshold_rejected", "valid"],
        "status_semantics": (
            "valid means the observed value agrees with the frozen prediction within "
            "the effective tolerance; threshold_rejected is a formal negative result "
            "and never rewrites the prediction"
        ),
        "tolerance_semantics": _TOLERANCE_SEMANTICS,
        "selection_semantics": _SELECTION_SEMANTICS,
        "json_columns": ["selection"],
        "prediction_artifact_id": prediction_artifact_id,
        "observation_artifact_id": observation_artifact_id,
        "inputs": {
            "prediction_artifact_id": prediction_artifact_id,
            "observation_artifact_id": observation_artifact_id,
        },
    }
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="validation", role=AxisRole.INDEX, size=count)],
        variables=variables,
        attributes=attributes,
    )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.sde",
        provenance={
            "product": "prediction_validation",
            "prediction_artifact_id": prediction_artifact_id,
            "observation_artifact_id": observation_artifact_id,
            "comparison_count": count,
        },
    )
