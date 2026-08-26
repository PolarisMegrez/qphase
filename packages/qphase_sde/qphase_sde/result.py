"""qphase_sde: Simulation Result
---------------------------------------------------------
Container for SDE simulation results, supporting serialization and deserialization.

``SDEResult`` is the legacy 1.x container, kept for 1.x reproduction and the
one-way migration tool; the 2.x engine returns :class:`SDEDataBundle`, a
catalog of typed data products plus provenance that satisfies the core
``ResultProtocol``/``DatasetResultProtocol`` and the frozen
:class:`~qphase_sde.contracts.bundle.SDEDataBundleProtocol`.

Public API
----------
``SDEResult`` : Legacy container for SDE simulation results.
``SDEDataBundle`` : 2.0 bundle of typed data products plus provenance.
``bundle_from_result`` : Boundary adapter from legacy results to bundles.
``recorded_distribution_versions`` : Real installed distribution versions
recorded in artifact provenance.
"""

import importlib.metadata
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from qphase.backend.xputil import convert_to_numpy
from qphase.core.dataset import DatasetSaveReport
from qphase.core.errors import QPhaseError
from qphase.core.utils import canonical_json
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
    add_scan_parameter_coordinates,
    assemble_typed_product,
    json_safe_meta,
    stack_payload_leaves,
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


@dataclass
class SDEResult:
    """Container for SDE simulation results.

    Attributes
    ----------
    trajectory : Any
        The trajectory data (e.g., numpy array or TrajectorySet).
    meta : dict[str, Any]
        Metadata about the simulation (config, runtime info, etc.).

    """

    trajectory: Any = None
    analysis: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Any:
        """Alias for trajectory to satisfy ResultProtocol.

        If trajectory was dropped (e.g. after analysis), return analysis results.
        """
        if self.trajectory is not None:
            return self.trajectory
        if self.analysis:
            return self.analysis
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Alias for meta to satisfy ResultProtocol."""
        return self.meta

    @property
    def label(self) -> Any:
        """Get the label (e.g. parameter value) from metadata."""
        return self.meta.get("label")

    @label.setter
    def label(self, value: Any) -> None:
        """Set the label in metadata."""
        self.meta["label"] = value

    @property
    def index(self) -> Any:
        """Get the index (time/parameter) from the trajectory if available."""
        if hasattr(self.trajectory, "index"):
            return self.trajectory.index
        if hasattr(self.trajectory, "times"):
            return self.trajectory.times
        return None

    def save(self, path: str | Path) -> None:
        """Save the result to a file.

        Parameters
        ----------
        path : str | Path
            Path to save the result to.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert trajectory to numpy if possible for storage
        data_to_save = None
        if self.trajectory is not None:
            if hasattr(self.trajectory, "data"):
                data_to_save = convert_to_numpy(self.trajectory.data)
            else:
                data_to_save = convert_to_numpy(self.trajectory)

        # Extract time info if available
        t0 = getattr(self.trajectory, "t0", self.meta.get("t0", 0.0))
        dt = getattr(self.trajectory, "dt", self.meta.get("dt", 1.0))

        try:
            # Wrap meta in object array to allow saving dict in npz
            # np.savez expects arrays, so we wrap the dict
            meta_arr = np.array(self.meta, dtype=object)
            analysis_arr = np.array(self.analysis, dtype=object)

            save_kwargs = {
                "t0": t0,
                "dt": dt,
                "meta": meta_arr,
                "analysis": analysis_arr,
                "trajectory_meta": np.array(
                    getattr(self.trajectory, "meta", {}), dtype=object
                ),
            }

            if data_to_save is not None:
                save_kwargs["data"] = data_to_save

            np.savez_compressed(path, **save_kwargs)
        except Exception as e:
            raise QPhaseError(f"Failed to save SDEResult to {path}: {e}") from e

    @classmethod
    def load(cls, path: str | Path) -> "SDEResult":
        """Load a result from a file.

        Parameters
        ----------
        path : str | Path
            Path to load the result from.

        Returns
        -------
        SDEResult
            Loaded result object.

        """
        path = Path(path)
        if not path.exists():
            raise QPhaseError(f"File not found: {path}")
        try:
            with np.load(path, allow_pickle=True) as npz:
                data = npz["data"] if "data" in npz else None
                t0 = float(npz["t0"]) if "t0" in npz else 0.0
                dt = float(npz["dt"]) if "dt" in npz else 1.0
                meta = npz["meta"].item() if "meta" in npz else {}
                analysis = npz["analysis"].item() if "analysis" in npz else {}
                trajectory_meta = (
                    npz["trajectory_meta"].item() if "trajectory_meta" in npz else {}
                )

                traj = None
                if data is not None:
                    # Construct a minimal object that mimics TrajectorySet
                    class MinimalTrajectory:
                        def __init__(self, data, t0, dt, meta):
                            self.data = data
                            self.t0 = t0
                            self.dt = dt
                            self.meta = meta

                    traj = MinimalTrajectory(data, t0, dt, trajectory_meta)

                return cls(trajectory=traj, meta=meta, analysis=analysis)

        except Exception as e:
            raise QPhaseError(f"Failed to load SDEResult from {path}: {e}") from e


# Alias for backward compatibility if needed
SimulationResult = SDEResult


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
        return sum(
            product.nbytes or 0 for product in self._products.values()
        )

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
        meta["scan_point"] = {
            name: values[position]
            for (name, values), position in zip(
                self._scan_axes.items(), index, strict=True
            )
        }
        params = meta.get("params")
        if isinstance(params, dict):
            # Mirror the 1.x SDEScanResult.point_view contract: the per-point
            # view reports this point's swept parameter values, not the fused
            # whole-scan arrays.
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
        """v3 bundle descriptor restoring this bundle as an SDEDataBundle.

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
        """JSON provenance record persisted into the v3 artifact manifest."""
        return self._manifest_provenance()

    def save(self, path: str | Path) -> None:
        """Persist all products as a v3 artifact directory at ``path``."""
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
        """Persist through the v3 manifest pipeline (DatasetResultProtocol).

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
        files = tuple(
            sorted(item for item in Path(path).rglob("*") if item.is_file())
        )
        return DatasetSaveReport(
            resolved,
            files,
            loader="+".join(
                sorted({entry.storage.adapter for entry in manifest.products})
            ),
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )

    def legacy_result(self) -> "SDEResult":
        """Legacy SDEResult view of a single-point bundle (Phase 1 bridge).

        Cross-job analysers written against the 1.x containers consume this
        view through ``load_sde_results``; scan bundles must be mapped per
        point via :meth:`point_view` first.
        """
        if self._scan_shape:
            raise ValueError(
                "scan bundle has no single legacy view; use point_view first"
            )
        return legacy_view_from_products(self._products, meta=dict(self._meta))

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
    """Short stable fingerprint string of a plugin instance."""
    if instance is None:
        return ""
    import hashlib

    from qphase.core.execution import plugin_fingerprint

    try:
        payload = plugin_fingerprint(instance)
    except Exception:  # noqa: BLE001 - fingerprinting must not break runs
        return ""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _trajectory_product(
    trajectory: Any,
    *,
    scan_size: int,
    n_traj_per_point: int | None,
) -> TimeSeriesDataset:
    """Build the trajectory time-series product from a TrajectorySet.

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
            {key: value for key, value in trajectory_meta.items()
             if key not in ("valid_length", "valid_lengths")}
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
            AxisSchema(
                name="trajectory", role=AxisRole.REALIZATION, size=n_traj
            ),
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
    device = f"cuda:{device_id}" if device_id is not None else str(
        raw_device or "unknown"
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


def _analysis_product(
    name: str,
    payload: Any,
    *,
    scan_size: int,
) -> Dataset | None:
    """Bridge one legacy analyser payload into a statistics product.

    Migration-only path (``bridge="legacy_analysis/1"``,
    ``graph_ready=False``): numeric payload leaves become variables over a
    parameter scan axis plus open positional index axes. Analysers with a
    typed product builder never take this path in new 2.x runs; the bridge
    remains for 1.x artifact migration and as a one-shot diagnostic for
    analysers not yet migrated. Missing per-point payloads and inconsistent
    per-point keys are rejected; nothing is ever pickled.
    """
    leaves = stack_payload_leaves(name, payload, scan_size=scan_size)
    if leaves is None:
        return None
    return assemble_typed_product(
        name,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        attributes={
            "bridge": "legacy_analysis/1",
            "graph_ready": False,
        },
    )


def _assign_nested(payload: dict[Any, Any], dotted_key: str, value: Any) -> None:
    """Assign ``value`` into ``payload`` following a dotted key path.

    Digit-only path segments become integer keys: legacy analyser payloads
    index per-mode tables by mode number.
    """
    parts = dotted_key.split(".")
    target = payload
    for part in parts[:-1]:
        key: Any = int(part) if part.isdigit() else part
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    leaf = parts[-1]
    target[int(leaf) if leaf.isdigit() else leaf] = value


def legacy_view_from_products(
    products: Mapping[str, Dataset],
    *,
    meta: dict[str, Any] | None = None,
) -> SDEResult:
    """Rebuild a legacy SDEResult view from typed products (Phase 1 bridge).

    Inverse of the bundle bridges: the ``trajectories`` product becomes a
    TrajectorySet, and each ``legacy_analysis/1`` statistics product becomes
    its original analyser payload (variables plus ``payload_meta`` leaves).
    Single-point semantics: callers must map scan products per point first
    (e.g. through :meth:`SDEDataBundle.point_view`).
    """
    from qphase_sde.state import TrajectorySet

    trajectory = None
    analysis: dict[str, Any] = {}
    deferred_bridges: list[tuple[str, str, dict[Any, Any]]] = []
    for name, product in products.items():
        axis_names = {axis.name for axis in product.axes}
        if _SCAN_AXIS in axis_names:
            scan_axis = product.axis(_SCAN_AXIS)
            if scan_axis.size not in (None, 1):
                raise ValueError(
                    f"product {name!r} still has a scan axis of size "
                    f"{scan_axis.size}; map per scan point first"
                )
        if name == "trajectories":
            alpha = product.handle("alpha").materialize()
            if alpha.ndim == 4:
                alpha = alpha[0]
            time_axis = product.axis("time") if "time" in axis_names else None
            t0 = (
                float(time_axis.start)
                if time_axis is not None and time_axis.start is not None
                else 0.0
            )
            dt = (
                float(time_axis.step)
                if time_axis is not None and time_axis.step is not None
                else 1.0
            )
            traj_meta = {
                key: value
                for key, value in product.attributes.items()
                if key not in TRAJECTORY_PRODUCT.attributes
            }
            trajectory = TrajectorySet(data=alpha, t0=t0, dt=dt, meta=traj_meta)
            continue
        payload: dict[Any, Any] = {}
        for variable in product.variables:
            _assign_nested(
                payload, variable.name, product.handle(variable.name).materialize()
            )
        per_point = set(product.attributes.get("per_point_meta", ()))
        scan_index = (meta or {}).get("scan_index")
        for meta_key, meta_value in product.attributes.get(
            "payload_meta", {}
        ).items():
            if meta_key in per_point and scan_index is not None:
                meta_value = list(meta_value)[int(scan_index)]
            _assign_nested(payload, meta_key, meta_value)
        if product.attributes.get("bridge") == "legacy_peaks/1":
            source = product.attributes.get("source_product")
            field = product.attributes.get("payload_field")
            if not isinstance(source, str) or not isinstance(field, str):
                raise ValueError(
                    f"legacy peak product {name!r} lacks an explicit source route"
                )
            deferred_bridges.append((source, field, payload))
            continue
        analysis[name] = payload
    for source, field, payload in deferred_bridges:
        if source not in analysis:
            raise ValueError(
                f"legacy peak bridge references missing source product {source!r}"
            )
        if field not in payload:
            raise ValueError(
                f"legacy peak bridge for {source!r} misses payload field {field!r}"
            )
        if field in analysis[source]:
            raise ValueError(
                f"legacy peak bridge would overwrite {source!r}.{field}"
            )
        analysis[source][field] = payload[field]
    return SDEResult(trajectory=trajectory, analysis=analysis, meta=dict(meta or {}))


def bundle_from_result(
    result: Any,
    *,
    provenance: SDEProvenance,
    n_traj_per_point: int | None = None,
    analysers: Mapping[str, Any] | None = None,
) -> SDEDataBundle:
    """Adapt a legacy SDEResult/SDEScanResult into an SDEDataBundle.

    Transitional Phase 1 adapter applied at the engine's public boundary:
    private execution paths still assemble legacy results (they migrate fully
    in Phase 2), while the engine's return value is always a bundle.

    ``analysers`` maps job-local analyser labels to the analyser instances
    that produced the payloads. An analyser implementing
    :class:`~qphase_sde.contracts.analyser.AnalyserProductBuilderProtocol`
    converts its own payload into graph-ready typed products; payloads of
    analysers without the hook fall back to the migration-only
    ``legacy_analysis/1`` bridge (``graph_ready=False``).
    """
    grid = getattr(result, "grid", None)
    combined = getattr(result, "combined", result)
    if not isinstance(combined, SDEResult):
        raise TypeError(
            f"cannot build an SDEDataBundle from {type(result).__name__}"
        )
    if grid is not None:
        scan_size = int(grid.size)
        scan_shape = tuple(grid.shape)
        scan_axes = dict(grid.axes)
        scan_coordinates = grid.parameter_arrays(flatten=True)
        if n_traj_per_point is None:
            n_traj_per_point = getattr(result, "n_traj_per_point", None)
    else:
        scan_size, scan_shape, scan_axes = 1, (), {}
        scan_coordinates = {}

    products: dict[str, Dataset] = {}
    if combined.trajectory is not None:
        products["trajectories"] = add_scan_parameter_coordinates(
            _trajectory_product(
                combined.trajectory,
                scan_size=scan_size,
                n_traj_per_point=n_traj_per_point,
            ),
            scan_coordinates,
        )
    dropped_products: list[str] = []
    for name, payload in combined.analysis.items():
        builder = None
        if analysers is not None:
            builder = getattr(analysers.get(str(name)), "build_products", None)
        if builder is not None:
            # The analyser's own builder is authoritative for its payload:
            # an empty result means "no products", never a bridge fallback.
            built = (
                builder(payload, scan_size=scan_size, label=str(name))
                if payload is not None
                else None
            )
            if built:
                analyser = analysers.get(str(name)) if analysers is not None else None
                declaration_factory = getattr(analyser, "output_spec", None)
                declaration = (
                    declaration_factory() if callable(declaration_factory) else None
                )
                for product_name, product in built.items():
                    if product_name in products:
                        raise ValueError(
                            f"analyser product {product_name!r} collides with "
                            "an existing engine or analyser product"
                        )
                    if not isinstance(product, Dataset):
                        raise TypeError(
                            f"analyser product {product_name!r} must be a Dataset"
                        )
                    graph_ready = product.attributes.get("graph_ready") is True
                    migration_bridge = product.attributes.get("bridge") in {
                        "legacy_analysis/1",
                        "legacy_peaks/1",
                    }
                    if not graph_ready and not migration_bridge:
                        raise TypeError(
                            f"analyser product {product_name!r} is not graph-ready"
                        )
                    if declaration is not None and graph_ready:
                        if product.kind is not declaration.kind:
                            raise TypeError(
                                f"analyser product {product_name!r} has kind "
                                f"{product.kind.value!r}, declared "
                                f"{declaration.kind.value!r}"
                            )
                        fields = {variable.name for variable in product.variables}
                        missing = sorted(set(declaration.fields) - fields)
                        if missing:
                            raise TypeError(
                                f"analyser product {product_name!r} misses "
                                f"declared fields {missing}"
                            )
                        if declaration.quantity and not any(
                            variable.quantity == declaration.quantity
                            for variable in product.variables
                        ):
                            raise TypeError(
                                f"analyser product {product_name!r} misses "
                                f"declared quantity {declaration.quantity!r}"
                            )
                    products[product_name] = (
                        add_scan_parameter_coordinates(product, scan_coordinates)
                        if graph_ready
                        else product
                    )
            elif payload is not None:
                dropped_products.append(str(name))
            continue
        product = _analysis_product(str(name), payload, scan_size=scan_size)
        if product is not None:
            products[str(name)] = product
        elif payload is not None:
            dropped_products.append(str(name))

    meta = dict(combined.meta)
    extra_meta = getattr(result, "meta", None)
    if isinstance(extra_meta, dict):
        meta.update(extra_meta)
    if dropped_products:
        meta["dropped_products"] = dropped_products
    return SDEDataBundle(
        products,
        provenance,
        scan_axes=scan_axes,
        scan_shape=scan_shape,
        meta=meta,
        n_traj_per_point=n_traj_per_point,
    )


# ---------------------------------------------------------------------------
# v3 artifact restore
# ---------------------------------------------------------------------------


def restore_sde_bundle(manifest: Any, products: dict[str, Dataset]) -> SDEDataBundle:
    """Rebuild an SDEDataBundle from a v3 artifact manifest.

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
            raise ArtifactCorruptError(
                "SDE scan axis names must be non-empty strings"
            )
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
        unknown_roles = sorted(
            set(manifest.bundle.product_roles) - allowed_roles
        )
        if unknown_roles:
            raise ArtifactCorruptError(
                f"SDE bundle contains unknown stable roles {unknown_roles}"
            )
        trajectories = manifest.bundle.product_roles.get("trajectories")
        if trajectories is not None:
            schema = entries[trajectories].product_schema
            fields = {variable.name for variable in schema.variables}
            if schema.kind is not DataKind.TIME_SERIES or not {
                "alpha",
                "valid_length",
            } <= fields:
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
