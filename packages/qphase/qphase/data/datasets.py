"""qphase: Typed Dataset Containers
---------------------------------------------------------
Concrete :class:`~qphase.data.product.DataProduct` implementations for the
three public data kinds: :class:`TimeSeriesDataset`, :class:`SpectralDataset`
and :class:`StatisticsDataset`.

A dataset couples a frozen :class:`~qphase.data.schema.ProductSchema` with a
runtime backing (one read-only handle per variable) or a durable
:class:`~qphase.data.artifact.ArtifactRef`. Datasets never coerce to NumPy
arrays implicitly: payload access is explicit through handles, views and
``materialize(target_device, copy_policy)`` so device-to-host copies are
always deliberate. All metadata (schema, attributes, provenance) is plain
JSON — pickle and object arrays are forbidden by the schema layer.

Artifact-backed datasets resolve their payload lazily through the registered
storage adapter named in the artifact reference (a trusted registry id, never
a Python code path) and an explicit ``ArtifactResolver`` supplied by the
caller; the concrete artifact store and NPZ adapters ship with the current
artifact manifest.

Public API
----------
Dataset
    Base container shared by the three kinds.
TimeSeriesDataset
    Container of ``time_series`` products.
SpectralDataset
    Container of ``spectral`` products (mandatory spectral attributes).
StatisticsDataset
    Container of ``statistics`` products: typed columns over a row axis.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, NoReturn

import numpy as np

from ..core.utils import canonical_json
from .artifact import ArtifactRef
from .errors import ArtifactNotFoundError
from .handles import CopyPolicy, DataHandleProtocol
from .kinds import DataKind
from .product import RuntimeProductBacking, validate_backing
from .resolver import ArtifactResolverProtocol
from .runtime import (
    BackendArrayHandle,
    DictProductBacking,
    HostArrayHandle,
    ReadOnlyArrayView,
)
from .schema import (
    AxisRole,
    AxisSchema,
    CoordinateSchema,
    ProductSchema,
    SpectralAttributes,
    VariableSchema,
)

__all__ = [
    "Dataset",
    "SpectralDataset",
    "StatisticsDataset",
    "TimeSeriesDataset",
]


def _is_int_index(value: Any) -> bool:
    """Return True for integer indexers (bools excluded)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _wrap_array(
    array: Any,
    variable: VariableSchema,
    *,
    owner: str,
    device: str,
) -> Any:
    """Wrap a payload array in a read-only handle for its device."""
    if device == "cpu":
        return HostArrayHandle(array, variable, owner=owner, read_only=True)
    return BackendArrayHandle(
        array, variable, owner=owner, device=device, read_only=True
    )


def _load_artifact_backing(
    ref: ArtifactRef, resolver: ArtifactResolverProtocol | None
) -> RuntimeProductBacking:
    """Resolve a storage adapter through an explicit artifact resolver."""
    from .store import _resolve_adapter  # local import: store imports datasets

    adapter = _resolve_adapter(ref.storage_adapter)
    if resolver is None:
        raise ArtifactNotFoundError(
            "artifact materialization requires an explicit ArtifactResolver"
        )
    backing = adapter.open_ref(ref, resolver=resolver)
    if not isinstance(backing, RuntimeProductBacking):
        raise TypeError(
            f"storage adapter {ref.storage_adapter!r} returned "
            f"{type(backing).__name__}, expected a RuntimeProductBacking"
        )
    return backing


class Dataset:
    """Typed data product over a runtime backing or an artifact reference.

    Runtime handles are wrapped in read-only views at construction, so
    consumers never receive a writable buffer through a dataset.
    ``point_view`` and ``slice_view`` build derived datasets that share
    memory with the source buffers; artifact-backed datasets must be
    ``materialize()``-d before their payload can be accessed.
    """

    _EXPECTED_KIND: ClassVar[DataKind | None] = None

    def __init__(
        self,
        schema: ProductSchema,
        backing: RuntimeProductBacking | ArtifactRef,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        expected = self._EXPECTED_KIND
        if expected is not None and schema.kind is not expected:
            raise ValueError(
                f"{type(self).__name__} requires kind {expected.value!r}, "
                f"got {schema.kind.value!r}"
            )
        prov = dict(provenance or {})
        try:
            canonical_json(prov)
        except (TypeError, ValueError) as exc:
            raise ValueError("provenance must be JSON-serializable") from exc

        if isinstance(backing, ArtifactRef):
            if backing.product_schema != schema:
                raise ValueError(
                    "artifact ref product schema does not match the dataset schema"
                )
            self._backing: RuntimeProductBacking | ArtifactRef = backing
        elif isinstance(backing, RuntimeProductBacking):
            views = {
                name: handle if handle.read_only else ReadOnlyArrayView(handle)
                for name, handle in backing.variables.items()
            }
            runtime_backing = DictProductBacking(views)
            validate_backing(schema, runtime_backing)
            self._backing = runtime_backing
        else:
            raise TypeError(
                "backing must be a RuntimeProductBacking or an ArtifactRef, "
                f"got {type(backing).__name__}"
            )
        self._schema = schema
        self._provenance = prov

    # -- DataProduct protocol ---------------------------------------------

    @property
    def schema(self) -> ProductSchema:
        """The product's machine-readable schema."""
        return self._schema

    @property
    def provenance(self) -> Mapping[str, Any]:
        """JSON-serializable provenance mapping."""
        return dict(self._provenance)

    @property
    def backing(self) -> RuntimeProductBacking | ArtifactRef:
        """Runtime backing or artifact reference backing this product."""
        return self._backing

    # -- schema conveniences ------------------------------------------------

    @property
    def kind(self) -> DataKind:
        """The product's data kind."""
        return self._schema.kind

    @property
    def attributes(self) -> dict[str, Any]:
        """A copy of the product's JSON attributes."""
        return dict(self._schema.attributes)

    @property
    def axes(self) -> tuple[AxisSchema, ...]:
        """The product's named axes."""
        return tuple(self._schema.axes)

    def axis(self, name: str) -> AxisSchema:
        """Return the axis with the given name."""
        return self._schema.axis(name)

    @property
    def variables(self) -> tuple[VariableSchema, ...]:
        """The product's named variables."""
        return tuple(self._schema.variables)

    def variable(self, name: str) -> VariableSchema:
        """Return the variable with the given name."""
        return self._schema.variable(name)

    def coordinates(self) -> tuple[CoordinateSchema, ...]:
        """Return the product's declared coordinate labels."""
        return tuple(self._schema.coordinates)

    def coordinate(self, name: str) -> np.ndarray:
        """Materialize the values of one declared coordinate.

        Raises ``KeyError`` when no coordinate with that name is declared.
        """
        for coordinate in self._schema.coordinates:
            if coordinate.name == name:
                return np.asarray(self.handle(coordinate.variable).materialize())
        raise KeyError(f"unknown coordinate {name!r}")

    # -- backing introspection ----------------------------------------------

    @property
    def is_runtime_backed(self) -> bool:
        """Whether the payload is held by runtime handles."""
        return isinstance(self._backing, RuntimeProductBacking)

    @property
    def is_artifact_backed(self) -> bool:
        """Whether the payload is referenced by a durable artifact."""
        return isinstance(self._backing, ArtifactRef)

    @property
    def devices(self) -> tuple[str, ...]:
        """Devices holding the payload; empty for artifact backings."""
        if not self.is_runtime_backed:
            return ()
        return tuple(
            sorted({handle.device for handle in self._runtime_handles().values()})
        )

    @property
    def shape(self) -> dict[str, tuple[int | None, ...]]:
        """Per-variable payload shape.

        Runtime backings report concrete handle shapes; artifact backings
        derive shapes from the schema and mark open axes with None.
        """
        if self.is_runtime_backed:
            return {
                name: tuple(handle.shape)
                for name, handle in self._runtime_handles().items()
            }
        return {
            variable.name: tuple(self._schema.axis(dim).size for dim in variable.dims)
            for variable in self._schema.variables
        }

    @property
    def nbytes(self) -> int | None:
        """Total payload bytes; None while an artifact's layout is unknown.

        Derived from handles (runtime backings) or from closed schema axes
        (artifact backings); never triggers device synchronization.
        """
        if self.is_runtime_backed:
            return sum(handle.nbytes for handle in self._runtime_handles().values())
        total = 0
        for variable in self._schema.variables:
            size = 1
            for dim in variable.dims:
                axis_size = self._schema.axis(dim).size
                if axis_size is None:
                    return None
                size *= axis_size
            total += size * np.dtype(variable.dtype).itemsize
        return total

    def summary(self) -> dict[str, Any]:
        """JSON-serializable summary for service/GUI listings."""
        result: dict[str, Any] = {
            "kind": self._schema.kind.value,
            "schema_version": self._schema.schema_version,
            "fingerprint": self._schema.fingerprint(),
            "backing": "artifact" if self.is_artifact_backed else "runtime",
            "devices": list(self.devices),
            "nbytes": self.nbytes,
            "axes": [
                {
                    "name": axis.name,
                    "role": axis.role.value,
                    "size": axis.size,
                    "coordinate": axis.coordinate,
                    "start": axis.start,
                    "step": axis.step,
                    "units": axis.units,
                    "monotonic": axis.monotonic,
                }
                for axis in self._schema.axes
            ],
            "variables": [
                {
                    "name": variable.name,
                    "dtype": variable.dtype,
                    "value_domain": variable.value_domain,
                    "dims": list(variable.dims),
                    "quantity": variable.quantity,
                    "units": variable.units,
                }
                for variable in self._schema.variables
            ],
            "coordinates": [
                {
                    "name": coordinate.name,
                    "variable": coordinate.variable,
                    "dims": list(coordinate.dims),
                    "role": coordinate.role,
                    "units": coordinate.units,
                    "monotonic": coordinate.monotonic,
                }
                for coordinate in self._schema.coordinates
            ],
        }
        backing = self._backing
        if isinstance(backing, ArtifactRef):
            result["artifact_id"] = backing.artifact_id
        return result

    # -- payload access -------------------------------------------------------

    def handle(self, variable: str) -> DataHandleProtocol:
        """Read-only runtime handle of one variable.

        Artifact-backed datasets raise; call ``materialize()`` first.
        """
        handles = self._runtime_handles()
        try:
            return handles[variable]
        except KeyError:
            raise KeyError(f"unknown variable {variable!r}") from None

    def materialize(
        self,
        target_device: str | None = None,
        copy_policy: CopyPolicy = "allow",
        *,
        resolver: ArtifactResolverProtocol | None = None,
    ) -> Dataset:
        """Return a runtime-backed dataset on ``target_device``.

        ``target_device=None`` keeps the payload where it is: runtime-backed
        datasets are returned unchanged, artifact-backed datasets are loaded
        onto the host through the storage adapter named in the reference and
        ``resolver``. Explicit
        device placement goes through the per-variable handles with the given
        copy policy; host handles never perform host-to-device transfers, so
        moving a host payload to a device raises. No implicit device-to-host
        copy is ever performed.
        """
        backing = self._backing
        if isinstance(backing, ArtifactRef):
            host = type(self)(
                self._schema,
                _load_artifact_backing(backing, resolver),
                provenance=self._provenance,
            )
            if target_device in (None, "cpu"):
                return host
            return host.materialize(target_device, copy_policy, resolver=resolver)
        handles = dict(backing.variables)
        if target_device is None or (
            target_device == "cpu"
            and all(handle.device == "cpu" for handle in handles.values())
        ):
            return self
        moved = {
            name: _wrap_array(
                handle.materialize(target_device, copy_policy),
                handle.variable_schema,
                owner=handle.owner,
                device=target_device,
            )
            for name, handle in handles.items()
        }
        return type(self)(
            self._schema,
            DictProductBacking(moved),
            provenance=self._provenance,
        )

    def point_view(self, **indices: int) -> Dataset:
        """Point selection along named axes; every indexer must be an int."""
        for name, index in indices.items():
            if not _is_int_index(index):
                raise TypeError(
                    f"point_view indexer for axis {name!r} must be an int, "
                    f"got {type(index).__name__}"
                )
        return self.slice_view(**indices)

    def slice_view(self, **selection: Any) -> Dataset:
        """Axis-named int/slice selection returning a derived dataset.

        Integer selection drops the axis from the schema (and from the dims
        of every variable spanning it); slices keep the axis with an updated
        size. Views share memory with the source buffers and stay on their
        device. Handles implementing ``materialize_selection(indexers)``
        (e.g. lazy storage handles) are asked for a partial read instead of
        a full materialization. Sampling bases whose source axis is dropped
        are dropped as well; uncertainties referencing a dropped basis raise
        instead of being silently discarded. Artifact-backed datasets must
        be materialized first.
        """
        handles = self._runtime_handles()
        axes_by_name = {axis.name: axis for axis in self._schema.axes}
        for name, selector in selection.items():
            if name not in axes_by_name:
                raise ValueError(f"unknown axis {name!r}")
            if not (_is_int_index(selector) or isinstance(selector, slice)):
                raise TypeError(
                    f"selector for axis {name!r} must be an int or slice, "
                    f"got {type(selector).__name__}"
                )
        spanned = {dim for variable in self._schema.variables for dim in variable.dims}
        unspanned = set(selection) - spanned
        if unspanned:
            raise ValueError(
                f"selection axes are not spanned by any variable: {sorted(unspanned)}"
            )

        dropped = {
            name for name, selector in selection.items() if _is_int_index(selector)
        }
        new_variables: list[VariableSchema] = []
        new_handles: dict[str, Any] = {}
        axis_sizes: dict[str, int] = {}
        for variable in self._schema.variables:
            handle = handles[variable.name]
            if not any(dim in selection for dim in variable.dims):
                new_variables.append(variable)
                new_handles[variable.name] = handle
                continue
            indexers = tuple(
                selection[dim] if dim in selection else slice(None)
                for dim in variable.dims
            )
            selection_reader = getattr(handle, "materialize_selection", None)
            if callable(selection_reader):
                # Lazy handles (e.g. storage-backed) read only the chunks the
                # selection touches instead of the full payload.
                sub = selection_reader(indexers)
            else:
                sub = handle.materialize()[indexers]
            if isinstance(sub, np.generic):
                # All-int selection on a host array yields a NumPy scalar;
                # normalize to a 0-d array so handles can expose views.
                sub = np.asarray(sub)
            new_dims = tuple(dim for dim in variable.dims if dim not in dropped)
            new_variable = variable.model_copy(update={"dims": new_dims})
            new_handles[variable.name] = _wrap_array(
                sub, new_variable, owner=handle.owner, device=handle.device
            )
            for dim, size in zip(new_dims, sub.shape, strict=True):
                axis_sizes[dim] = size
            new_variables.append(new_variable)

        new_axes: list[AxisSchema] = []
        for axis in self._schema.axes:
            if axis.name in dropped:
                continue
            if axis.name in selection:
                selector = selection[axis.name]
                if axis.size is not None:
                    size = len(range(*selector.indices(axis.size)))
                else:
                    size = axis_sizes.get(axis.name)
                updates: dict[str, Any] = {"size": size}
                # Regular coordinates shift/stretches with the slice:
                # start advances to the first retained sample and the step
                # picks up the slice stride (negative for reversed views).
                if (
                    axis.coordinate == "regular"
                    and isinstance(selector, slice)
                    and axis.size is not None
                    and axis.start is not None
                    and axis.step is not None
                ):
                    start_index, _stop_index, slice_step = selector.indices(axis.size)
                    updates["start"] = axis.start + start_index * axis.step
                    updates["step"] = axis.step * slice_step
                axis = axis.model_copy(update=updates)
            new_axes.append(axis)

        dropped_basis_names = {
            basis.name
            for basis in self._schema.sampling_bases
            if basis.source_axis is not None and basis.source_axis in dropped
        }
        orphaned = sorted(
            uncertainty.target
            for uncertainty in self._schema.uncertainties
            if uncertainty.sampling_basis in dropped_basis_names
        )
        if orphaned:
            raise ValueError(
                "point selection drops the sampling basis of uncertainties on "
                f"{orphaned}; slice the axis or drop the uncertainties "
                "explicitly instead"
            )
        # Coordinates follow the same indexers as data variables; integer
        # selection drops their dims, demoting sub-/super-dimensional
        # coordinates to scalar auxiliary labels.
        new_coordinates = []
        for coordinate in self._schema.coordinates:
            new_coordinate_dims = tuple(
                dim for dim in coordinate.dims if dim not in dropped
            )
            role = coordinate.role
            if role == "dimension" and len(new_coordinate_dims) != 1:
                role = "auxiliary"
            new_coordinates.append(
                coordinate.model_copy(
                    update={"dims": new_coordinate_dims, "role": role}
                )
            )
        new_schema = ProductSchema(
            kind=self._schema.kind,
            axes=new_axes,
            sampling_bases=[
                basis
                for basis in self._schema.sampling_bases
                if basis.name not in dropped_basis_names
            ],
            variables=new_variables,
            uncertainties=self._schema.uncertainties,
            coordinates=new_coordinates,
            attributes=self._schema.attributes,
        )
        return type(self)(
            new_schema,
            DictProductBacking(new_handles),
            provenance=self._provenance,
        )

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        schema: ProductSchema,
        arrays: Mapping[str, Any],
        *,
        owner: str,
        provenance: Mapping[str, Any] | None = None,
        device: str = "cpu",
    ) -> Dataset:
        """Build a dataset from a variable→array mapping.

        With ``device="cpu"`` array-likes are converted with ``np.asarray``
        and wrapped in host handles; any other device wraps NumPy-API arrays
        in backend handles without copying. Every schema variable must be
        covered exactly once.
        """
        expected = {variable.name for variable in schema.variables}
        missing = expected - set(arrays)
        extra = set(arrays) - expected
        if missing:
            raise ValueError(f"missing arrays for variables: {sorted(missing)}")
        if extra:
            raise ValueError(f"arrays without schema variables: {sorted(extra)}")
        handles = {}
        for variable in schema.variables:
            array = arrays[variable.name]
            if device == "cpu":
                handle: Any = HostArrayHandle(np.asarray(array), variable, owner=owner)
            else:
                handle = BackendArrayHandle(array, variable, owner=owner, device=device)
            handles[variable.name] = handle
        return cls(schema, DictProductBacking(handles), provenance=provenance)

    # -- guards ---------------------------------------------------------------

    def __array__(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Forbid implicit NumPy coercion (and implicit device-to-host copies)."""
        raise TypeError(
            f"{type(self).__name__} never coerces to a NumPy array "
            "implicitly; use materialize()/slice_view() with an explicit "
            "device and copy policy"
        )

    def __repr__(self) -> str:
        """Compact debug representation."""
        names = [variable.name for variable in self._schema.variables]
        backing = "artifact" if self.is_artifact_backed else "runtime"
        return (
            f"{type(self).__name__}(kind={self._schema.kind.value!r}, "
            f"variables={names!r}, backing={backing!r})"
        )

    # -- internals --------------------------------------------------------------

    def _runtime_handles(self) -> dict[str, Any]:
        backing = self._backing
        if not isinstance(backing, RuntimeProductBacking):
            raise RuntimeError(
                "dataset is artifact-backed; call materialize() to obtain a "
                "runtime-backed dataset first"
            )
        return dict(backing.variables)


class TimeSeriesDataset(Dataset):
    """Dataset of ``time_series`` products."""

    _EXPECTED_KIND: ClassVar[DataKind | None] = DataKind.TIME_SERIES


class SpectralDataset(Dataset):
    """Dataset of ``spectral`` products with mandatory spectral attributes."""

    _EXPECTED_KIND: ClassVar[DataKind | None] = DataKind.SPECTRAL

    @property
    def spectral_attributes(self) -> SpectralAttributes:
        """Validated spectral attribute set of this product."""
        return SpectralAttributes.model_validate(self._schema.attributes)


class StatisticsDataset(Dataset):
    """Dataset of ``statistics`` products: typed columns over a row axis.

    Tables are stored as one typed variable per column plus an index-role
    row axis — never as ``list[dict]`` object payloads.
    """

    _EXPECTED_KIND: ClassVar[DataKind | None] = DataKind.STATISTICS

    @property
    def row_axis(self) -> AxisSchema | None:
        """The first index-role axis, i.e. the table's row axis."""
        for axis in self._schema.axes:
            if axis.role is AxisRole.INDEX:
                return axis
        return None

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names of the table (all variables)."""
        return tuple(variable.name for variable in self._schema.variables)

    def column(self, name: str) -> VariableSchema:
        """Return the schema of one table column."""
        return self._schema.variable(name)
