"""qphase_sde: Simulation Result
---------------------------------------------------------
Container for SDE simulation results.

The 2.x engine returns :class:`SDEDataBundle`, a catalog of typed data
products plus provenance that satisfies the core
``ResultProtocol``/``DatasetResultProtocol`` and the frozen
:class:`~qphase_sde.contracts.bundle.SDEDataBundleProtocol`.

Public API
----------
``SDEDataBundle`` : 2.0 bundle of typed data products plus provenance.
``recorded_distribution_versions`` : Real installed distribution versions
recorded in artifact provenance.
"""

import importlib.metadata
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from qphase.core.dataset import DatasetSaveReport
from qphase.data import (
    AxisRole,
    AxisSchema,
    BundleDescriptor,
    DataKind,
    Dataset,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
    save_products,
)
from qphase.data.errors import ArtifactCorruptError, ArtifactUnsupportedError
from qphase.data.store import ARTIFACT_SCHEMA_VERSION, register_bundle_adapter

from qphase_sde.contracts.bundle import (
    SDE_BUNDLE_ADAPTER_ID,
    SDE_BUNDLE_TYPE_ID,
    TRAJECTORY_PRODUCT,
    SDEProvenance,
)
from qphase_sde.contracts.quantities import SDEQuantity
from qphase_sde.products import (
    json_safe_meta,
)


def recorded_distribution_versions() -> dict[str, str]:
    """Real installed distribution versions for artifact provenance.

    Versions come from the installed distribution metadata — never a
    hand-written constant; when a package is not installed as a
    distribution (plain source checkout) the module ``__version__`` is the
    fallback so provenance stays populated.
    """
    versions: dict[str, str] = {}
    distributions = (("qphase", "qphase"), ("qphase-sde", "qphase_sde"))
    for distribution, module in distributions:
        try:
            versions[module] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[module] = importlib.import_module(module).__version__
    return versions




# ---------------------------------------------------------------------------
# 2.0 typed data bundle
# ---------------------------------------------------------------------------

_SCAN_AXIS = "scan"

# ``_json_safe_meta`` lives in ``qphase_sde.products`` (shared with the
# analyser-owned product builders); keep the historical private alias.
_json_safe_meta = json_safe_meta


class SDEDataBundle:
    """Logical 2.0 result of an SDE job: named typed products plus provenance.

    The bundle is a catalog — it never copies arrays into itself; products
    hold their own runtime or artifact backings. It satisfies three contracts
    at once:

    - :class:`~qphase_sde.contracts.bundle.SDEDataBundleProtocol`
      (``products``/``provenance``/``require``);
    - the core ``ResultProtocol`` (``data``/``metadata``/``label``/``save``),
      so the unmodified scheduler accepts it;
    - the core ``DatasetResultProtocol``
      (``axes``/``shape``/``point_view``/``save_dataset``), so scan jobs stay
      persistable and downstream ``input.mode=map`` jobs keep working.

    Product names are job-local labels; dependency selection goes through
    :meth:`require` by kind/quantity/fields, never by label alone.
    """

    def __init__(
        self,
        products: dict[str, Dataset],
        provenance: SDEProvenance,
        *,
        scan_axes: dict[str, Any] | None = None,
        scan_shape: tuple[int, ...] = (),
        meta: dict[str, Any] | None = None,
        n_traj_per_point: int | None = None,
        product_roles: Mapping[str, str] | None = None,
    ) -> None:
        self._products = dict(products)
        self._provenance = provenance
        self._scan_axes = dict(scan_axes or {})
        self._scan_shape = tuple(scan_shape)
        self._meta = dict(meta or {})
        self._n_traj_per_point = n_traj_per_point
        if product_roles is None:
            roles: dict[str, str] = {}
            if "trajectories" in self._products:
                roles["trajectories"] = "trajectories"
            spectra = [
                name
                for name, product in self._products.items()
                if product.kind is DataKind.SPECTRAL
            ]
            if len(spectra) == 1:
                roles["primary_spectrum"] = spectra[0]
            self._product_roles = roles
        else:
            self._product_roles = dict(product_roles)

    # -- SDEDataBundleProtocol --------------------------------------------

    @property
    def products(self) -> dict[str, Dataset]:
        """Named data products of the bundle (job-local labels)."""
        return dict(self._products)

    @property
    def provenance(self) -> SDEProvenance:
        """Job-level provenance shared by all products."""
        return self._provenance

    def require(
        self,
        *,
        kind: DataKind | None = None,
        quantity: str | None = None,
        fields: tuple[str, ...] = (),
    ) -> dict[str, Dataset]:
        """Select products by kind/quantity/fields, never by label alone."""
        selected: dict[str, Dataset] = {}
        for name, product in self._products.items():
            if kind is not None and product.kind is not kind:
                continue
            if quantity is not None and not any(
                variable.quantity == quantity for variable in product.variables
            ):
                continue
            if fields and not set(fields) <= {
                variable.name for variable in product.variables
            }:
                continue
            selected[name] = product
        return selected

    # -- ResultProtocol ------------------------------------------------------

    @property
    def data(self) -> dict[str, Dataset]:
        """The product mapping (ResultProtocol surface)."""
        return dict(self._products)

    @property
    def metadata(self) -> dict[str, Any]:
        """Job metadata plus the JSON provenance record."""
        return {
            **self._meta,
            "provenance": self._provenance.model_dump(mode="json"),
        }

    @property
    def label(self) -> Any:
        """Get the label (e.g. parameter value) from metadata."""
        return self._meta.get("label")

    @label.setter
    def label(self, value: Any) -> None:
        """Set the label in metadata."""
        self._meta["label"] = value

    # -- DatasetResultProtocol ----------------------------------------------

    @property
    def axes(self) -> dict[str, Any]:
        """Scan axis coordinates (empty for single-point jobs)."""
        return dict(self._scan_axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Scan grid shape (empty for single-point jobs)."""
        return self._scan_shape

    @property
    def scan_size(self) -> int:
        """Number of flat scan points (1 for single-point jobs)."""
        size = 1
        for extent in self._scan_shape:
            size *= extent
        return size

    @property
    def n_traj_per_point(self) -> int | None:
        """Independent realizations per scan point (None when unknown).

        Falls back to the trajectory axis of the ``trajectories`` product
        when no explicit value was recorded (e.g. single-point jobs).
        """
        if self._n_traj_per_point is not None:
            return self._n_traj_per_point
        trajectories = self._products.get("trajectories")
        if trajectories is not None and "trajectory" in {
            axis.name for axis in trajectories.axes
        }:
            return trajectories.axis("trajectory").size
        return None

    @property
    def nbytes(self) -> int:
        """Total payload bytes across products (0 where unknown)."""
        return sum(product.nbytes or 0 for product in self._products.values())

    def point_view(self, index: tuple[int, ...]) -> "SDEDataBundle":
        """Per-scan-point view: products lose their scan axis.

        Products without a scan axis pass through unchanged. The per-point
        metadata records the flat scan index and the axis coordinates.
        """
        if not self._scan_shape:
            if tuple(index) == ():
                return self
            raise IndexError("bundle has no scan axes")
        flat = int(np.ravel_multi_index(tuple(index), self._scan_shape))
        meta = dict(self._meta)
        meta["scan_index"] = flat
        observer_payloads = meta.get("observer_payloads")
        if isinstance(observer_payloads, dict):
            meta["observer_payloads"] = {
                name: payloads[flat]
                if isinstance(payloads, list) and len(payloads) == self.scan_size
                else payloads
                for name, payloads in observer_payloads.items()
            }
        parameter_values = meta.get("scan_parameter_values")
        if isinstance(parameter_values, dict):
            meta["scan_point"] = {
                name: np.asarray(values).reshape(-1)[flat]
                for name, values in parameter_values.items()
            }
        else:
            meta["scan_point"] = {
                name: values[position]
                for (name, values), position in zip(
                    self._scan_axes.items(), index, strict=True
                )
            }
        meta["params"] = dict(meta["scan_point"])
        params = meta.get("params")
        if isinstance(params, dict):
            # The per-point view reports this point's swept parameter values,
            # not the fused whole-scan arrays.
            params = dict(params)
            params.update(meta["scan_point"])
            meta["params"] = params
        products: dict[str, Dataset] = {}
        for name, product in self._products.items():
            axis_names = {axis.name for axis in product.axes}
            if _SCAN_AXIS in axis_names:
                products[name] = product.point_view(**{_SCAN_AXIS: flat})
            else:
                products[name] = product
        return SDEDataBundle(
            products,
            self._provenance,
            meta=meta,
            n_traj_per_point=self._n_traj_per_point,
            product_roles=self._product_roles,
        )

    # -- persistence ---------------------------------------------------------

    @property
    def bundle_descriptor(self) -> BundleDescriptor:
        """Bundle descriptor restoring this bundle as an SDEDataBundle.

        The descriptor records the scan grid (shape, named dimension order,
        coordinate values, combine flag) and the trajectories-per-point
        count so a clean process can rebuild the scan semantics without
        any in-process state.
        """
        scan_axes, dropped = _json_safe_meta(self._scan_axes)
        if dropped:
            raise ValueError(
                f"scan axes {dropped} are not JSON-serializable; the bundle "
                f"cannot be described as {SDE_BUNDLE_TYPE_ID!r}"
            )
        scan: dict[str, Any] = {
            "shape": list(self._scan_shape),
            "dimension_order": list(self._scan_axes),
            "axes": scan_axes,
            "n_traj_per_point": self.n_traj_per_point,
        }
        if "scan_combine" in self._meta:
            combine, dropped_combine = _json_safe_meta(
                {"combine": self._meta["scan_combine"]}
            )
            if not dropped_combine:
                scan["combine"] = combine["combine"]
        return BundleDescriptor(
            type_id=SDE_BUNDLE_TYPE_ID,
            adapter_id=SDE_BUNDLE_ADAPTER_ID,
            descriptor_schema=SDE_BUNDLE_TYPE_ID,
            descriptor={"scan": scan},
            product_roles=self._product_roles,
        )

    @property
    def manifest_provenance(self) -> dict[str, Any]:
        """JSON provenance record persisted into the current artifact manifest."""
        return self._manifest_provenance()

    def save(self, path: str | Path) -> None:
        """Persist all products as an artifact directory at ``path``."""
        save_products(
            Path(path),
            self._products,
            provenance=self._manifest_provenance(),
            bundle=self.bundle_descriptor,
        )

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        """Persist through the current manifest pipeline (DatasetResultProtocol).

        ``layout="single"`` truly disables sharding (one payload file per
        product); ``"sharded"`` and the legacy ``"per_point"`` both map to
        byte-targeted chunk sharding.
        """
        resolved = "single" if layout == "single" else "sharded"
        manifest = save_products(
            Path(path),
            self._products,
            provenance=self._manifest_provenance(),
            shard_target_bytes=shard_target_bytes,
            layout=resolved,
            bundle=self.bundle_descriptor,
        )
        files = tuple(sorted(item for item in Path(path).rglob("*") if item.is_file()))
        return DatasetSaveReport(
            resolved,
            files,
            loader="+".join(
                sorted({entry.storage.adapter for entry in manifest.products})
            ),
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )

    def _manifest_provenance(self) -> dict[str, Any]:
        safe_meta, dropped = _json_safe_meta(self._meta)
        provenance: dict[str, Any] = {
            "engine": "sde",
            "sde": self._provenance.model_dump(mode="json"),
            "meta": safe_meta,
            "versions": recorded_distribution_versions(),
        }
        if dropped:
            provenance["meta_dropped"] = dropped
        return provenance

    def __repr__(self) -> str:
        """Compact debug representation."""
        return (
            f"SDEDataBundle(products={sorted(self._products)!r}, "
            f"scan_shape={self._scan_shape!r})"
        )


def _fingerprint_text(instance: Any) -> str:
    """Return a readable plugin identity for numerical provenance."""
    if instance is None:
        return ""
    from qphase.core.execution import plugin_fingerprint

    payload = plugin_fingerprint(instance)
    version = payload["distribution_version"] or "workspace"
    return f"{payload['class']}@{version}"


def _trajectory_product(
    trajectory: Any,
    *,
    scan_size: int | None,
    n_traj_per_point: int | None,
) -> TimeSeriesDataset:
    """Build the trajectory time-series product from a TrajectorySet.

    The product always carries a leading scan axis: exactly ``scan_size``
    for a genuine scan job (a 1-point scan keeps a size-1 axis) and a
    single fused point for jobs without a parameter grid
    (``scan_size is None``).

    Device-resident payloads (e.g. CuPy arrays from a GPU run) are wrapped
    without copying: they stay on their device until an explicit copy policy
    allows a move (persistence, host analysis).
    """
    data = getattr(trajectory, "data", trajectory)
    t0 = float(getattr(trajectory, "t0", 0.0))
    dt = float(getattr(trajectory, "dt", 1.0))
    # Do NOT np.asarray here: device arrays must survive the boundary.
    array = data
    if array.ndim != 3:
        raise ValueError(
            f"trajectory product expects (trajectory, time, channel) data, "
            f"got shape {array.shape}"
        )
    fused, n_time, n_channel = array.shape
    scan_size = 1 if scan_size is None else scan_size
    if scan_size > 1:
        if n_traj_per_point is None:
            if fused % scan_size:
                raise ValueError(
                    f"fused trajectory count {fused} is not divisible by "
                    f"scan size {scan_size}"
                )
            n_traj_per_point = fused // scan_size
        if fused != scan_size * n_traj_per_point:
            raise ValueError(
                f"fused trajectory count {fused} does not match scan size "
                f"{scan_size} x {n_traj_per_point} trajectories per point"
            )
        n_traj = n_traj_per_point
    else:
        n_traj = fused
    alpha = array.reshape(scan_size, n_traj, n_time, n_channel)

    valid = np.full((scan_size, n_traj), n_time, dtype=np.int64)
    attributes = dict(TRAJECTORY_PRODUCT.attributes)
    trajectory_meta = getattr(trajectory, "meta", None)
    if isinstance(trajectory_meta, dict):
        recorded = trajectory_meta.get("valid_length")
        if recorded is None:
            recorded = trajectory_meta.get("valid_lengths")
        if recorded is not None:
            recorded_array = np.asarray(recorded, dtype=np.int64)
            if recorded_array.shape == (fused,):
                valid = recorded_array.reshape(scan_size, n_traj)
        safe_meta, dropped_meta = _json_safe_meta(
            {
                key: value
                for key, value in trajectory_meta.items()
                if key not in ("valid_length", "valid_lengths")
            }
        )
        attributes.update(safe_meta)
        if dropped_meta:
            attributes["dropped_meta_keys"] = dropped_meta

    dtype = np.dtype(array.dtype)
    domain = "complex" if dtype.kind == "c" else "real"
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=scan_size),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=n_traj),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=n_time,
                coordinate="regular",
                start=t0,
                step=dt,
                units="inverse_rate",
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=n_channel),
        ],
        variables=[
            VariableSchema(
                name="alpha",
                dtype=dtype.str,
                value_domain=domain,
                dims=("scan", "trajectory", "time", "channel"),
                quantity=SDEQuantity.FIELD_AMPLITUDE.value,
            ),
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("scan", "trajectory"),
                quantity="valid_length",
            ),
        ],
        attributes=attributes,
    )
    if isinstance(alpha, np.ndarray):
        return cast(
            TimeSeriesDataset,
            TimeSeriesDataset.from_arrays(
                schema,
                {"alpha": alpha, "valid_length": valid},
                owner="engine.sde",
            ),
        )
    from qphase.data.runtime import (
        BackendArrayHandle,
        DictProductBacking,
        HostArrayHandle,
    )

    raw_device = getattr(alpha, "device", None)
    device_id = getattr(raw_device, "id", None)
    device = (
        f"cuda:{device_id}" if device_id is not None else str(raw_device or "unknown")
    )
    handles = {
        "alpha": BackendArrayHandle(
            alpha, schema.variable("alpha"), owner="engine.sde", device=device
        ),
        "valid_length": HostArrayHandle(
            valid, schema.variable("valid_length"), owner="engine.sde"
        ),
    }
    return TimeSeriesDataset(schema, DictProductBacking(handles))


# ---------------------------------------------------------------------------
# current artifact restore
# ---------------------------------------------------------------------------


def restore_sde_bundle(manifest: Any, products: dict[str, Dataset]) -> SDEDataBundle:
    """Rebuild an SDEDataBundle from the current artifact manifest.

    Registered as the ``sde/1`` bundle adapter: the manifest's bundle
    descriptor supplies the scan grid and its provenance record supplies
    the SDE provenance and job metadata, so a clean process restores the
    concrete bundle without any in-process registry state.
    """
    descriptor = manifest.bundle.descriptor
    scan = descriptor.get("scan")
    if not isinstance(scan, dict):
        raise ValueError(
            f"{SDE_BUNDLE_TYPE_ID!r} artifact is missing its scan descriptor"
        )
    shape = tuple(int(extent) for extent in scan.get("shape", ()))
    axes = {str(name): values for name, values in scan.get("axes", {}).items()}
    order = scan.get("dimension_order")
    if order is not None:
        axes = {str(name): axes[str(name)] for name in order}
    if len(axes) != len(shape):
        raise ValueError(
            f"{SDE_BUNDLE_TYPE_ID!r} scan descriptor is inconsistent: "
            f"{len(axes)} axes for shape {shape}"
        )
    raw_provenance = dict(manifest.provenance)
    sde_record = raw_provenance.get("sde")
    if not isinstance(sde_record, dict):
        # Tolerate the flattened layout written before the manifest
        # provenance contract (SDEProvenance fields at the top level).
        sde_record = {
            key: value
            for key, value in raw_provenance.items()
            if key in SDEProvenance.model_fields
        }
    provenance = SDEProvenance.model_validate(sde_record)
    raw_meta = raw_provenance.get("meta")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    if "scan_combine" not in meta and "combine" in scan:
        meta["scan_combine"] = scan["combine"]
    job_name = raw_provenance.get("job_name")
    if job_name is not None:
        meta.setdefault("job_name", job_name)
    n_traj = scan.get("n_traj_per_point")
    return SDEDataBundle(
        products,
        provenance,
        scan_axes=axes,
        scan_shape=shape,
        meta=meta,
        n_traj_per_point=int(n_traj) if n_traj is not None else None,
        product_roles=manifest.bundle.product_roles,
    )


class _SDEBundleAdapter:
    adapter_id = SDE_BUNDLE_ADAPTER_ID
    descriptor_schema = SDE_BUNDLE_TYPE_ID

    def validate_descriptor(self, descriptor: BundleDescriptor) -> None:
        if descriptor.type_id != SDE_BUNDLE_TYPE_ID:
            raise ArtifactCorruptError(
                f"SDE bundle adapter cannot restore type {descriptor.type_id!r}"
            )
        if descriptor.descriptor_schema != self.descriptor_schema:
            raise ArtifactUnsupportedError(
                f"SDE bundle descriptor schema {descriptor.descriptor_schema!r} "
                f"is unsupported; expected {self.descriptor_schema!r}"
            )
        if set(descriptor.descriptor) != {"scan"}:
            raise ArtifactCorruptError(
                "SDE bundle descriptor must contain exactly the scan field"
            )
        scan = descriptor.descriptor.get("scan")
        if not isinstance(scan, dict):
            raise ArtifactCorruptError("SDE bundle descriptor misses scan mapping")
        allowed = {
            "shape",
            "dimension_order",
            "axes",
            "n_traj_per_point",
            "combine",
        }
        extra = sorted(set(scan) - allowed)
        missing = sorted({"shape", "dimension_order", "axes"} - set(scan))
        if extra or missing:
            raise ArtifactCorruptError(
                f"SDE scan descriptor fields are invalid (missing: {missing}, "
                f"extra: {extra})"
            )
        shape_raw = scan["shape"]
        axes = scan["axes"]
        order = scan["dimension_order"]
        if not isinstance(shape_raw, list) or any(
            type(extent) is not int or extent <= 0 for extent in shape_raw
        ):
            raise ArtifactCorruptError(
                "SDE scan shape extents must be positive integers"
            )
        shape = tuple(shape_raw)
        if not isinstance(axes, dict) or not isinstance(order, list):
            raise ArtifactCorruptError(
                "SDE scan axes must be a mapping and dimension_order a list"
            )
        if any(not isinstance(name, str) or not name for name in order):
            raise ArtifactCorruptError(
                "SDE scan dimension_order entries must be non-empty strings"
            )
        if any(not isinstance(name, str) or not name for name in axes):
            raise ArtifactCorruptError("SDE scan axis names must be non-empty strings")
        if len(set(order)) != len(order) or set(order) != set(axes):
            raise ArtifactCorruptError(
                "SDE scan shape, dimension_order and axes are inconsistent"
            )
        combine = scan.get("combine", "cartesian")
        if combine not in {"cartesian", "zipped"}:
            raise ArtifactCorruptError(
                f"SDE scan combine mode {combine!r} is unsupported"
            )
        if combine == "cartesian" and len(shape) != len(order):
            raise ArtifactCorruptError(
                "cartesian SDE scan needs one shape extent per axis"
            )
        if combine == "zipped" and (len(shape) != 1 or not order):
            raise ArtifactCorruptError(
                "zipped SDE scan needs one point extent and at least one axis"
            )
        expected_sizes = (
            dict(zip(order, shape, strict=True))
            if combine == "cartesian"
            else {name: shape[0] for name in order}
        )
        for name in order:
            values = axes[name]
            if not isinstance(values, list):
                raise ArtifactCorruptError(
                    f"SDE scan axis {name!r} is not a coordinate sequence"
                )
            size = len(values)
            if size != expected_sizes[name]:
                raise ArtifactCorruptError(
                    f"SDE scan axis {name!r} has {size} values, expected "
                    f"{expected_sizes[name]}"
                )
        n_traj = scan.get("n_traj_per_point")
        if n_traj is not None and (
            not isinstance(n_traj, int) or isinstance(n_traj, bool) or n_traj <= 0
        ):
            raise ArtifactCorruptError(
                "SDE n_traj_per_point must be a positive integer or null"
            )

    def validate_manifest(self, manifest: Any) -> None:
        self.validate_descriptor(manifest.bundle)
        shape = tuple(manifest.bundle.descriptor["scan"]["shape"])
        scan_size = math.prod(shape) if shape else 1
        entries = {entry.name: entry for entry in manifest.products}
        for entry in manifest.products:
            axis_names = {axis.name for axis in entry.product_schema.axes}
            if "scan" not in axis_names:
                continue
            axis = entry.product_schema.axis("scan")
            if axis.size != scan_size:
                raise ArtifactCorruptError(
                    f"product {entry.name!r} scan axis size {axis.size} does "
                    f"not match SDE bundle scan size {scan_size}"
                )

        allowed_roles = {"trajectories", "primary_spectrum"}
        unknown_roles = sorted(set(manifest.bundle.product_roles) - allowed_roles)
        if unknown_roles:
            raise ArtifactCorruptError(
                f"SDE bundle contains unknown stable roles {unknown_roles}"
            )
        trajectories = manifest.bundle.product_roles.get("trajectories")
        if trajectories is not None:
            schema = entries[trajectories].product_schema
            fields = {variable.name for variable in schema.variables}
            if (
                schema.kind is not DataKind.TIME_SERIES
                or not {
                    "alpha",
                    "valid_length",
                }
                <= fields
            ):
                raise ArtifactCorruptError(
                    "SDE trajectories role must reference a trajectory "
                    "time-series product"
                )
        spectrum = manifest.bundle.product_roles.get("primary_spectrum")
        if spectrum is not None:
            schema = entries[spectrum].product_schema
            if schema.kind is not DataKind.SPECTRAL:
                raise ArtifactCorruptError(
                    "SDE primary_spectrum role must reference a spectral product"
                )

    def build(self, manifest: Any, products: dict[str, Dataset]) -> SDEDataBundle:
        try:
            return restore_sde_bundle(manifest, products)
        except (ArtifactCorruptError, ArtifactUnsupportedError):
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactCorruptError(
                f"failed to restore SDE bundle descriptor: {exc}"
            ) from exc


_SDE_BUNDLE_ADAPTER = _SDEBundleAdapter()
register_bundle_adapter(_SDE_BUNDLE_ADAPTER)
