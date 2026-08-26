"""qphase_sde: One-Way 1.x Result Migration (2.0)
---------------------------------------------------------
Offline converter from frozen 1.x SDE result files to 2.0 typed data
products plus an artifact manifest v4.

Three legacy layouts are recognized:

- ``sde_result/1``: a single ``SDEResult``/``AnalysisResult`` npz (keys
  ``data``/``t0``/``dt`` plus the pickled ``meta``/``analysis``/
  ``trajectory_meta`` scalars);
- ``trajectory_set/1``: a raw trajectory npz (``trajectories``,
  ``valid_length``, ``t0``, ``dt``);
- ``sde_scan/2``: an artifact-manifest v2 with ``layout="per_point"`` whose
  shards are per-point ``sde_result/1`` files.

Temporary transition guarantees (this module is removed after Global Phase 4):

- conversion is one-way and never overwrites or modifies the source files;
- unknown object payloads are rejected unless an explicit adapter is given;
- every source file's SHA-256 is recorded in the output provenance;
- frequency orientation, independent counts and uncertainty methods survive
  through the ``legacy_analysis/1`` bridge (``payload_meta`` / variables);
- unrecoverable semantics produce structured warnings, never guesses;
- scan conversion streams per point: peak memory stays within one legacy
  shard plus one output chunk (shards are read twice, never concatenated
  as a whole).

Public API
----------
migrate_legacy_result
    Convert one ``sde_result/1`` or ``trajectory_set/1`` file.
migrate_scan_artifact
    Convert one ``sde_scan/2`` per-point artifact.
MigrationReport
    Structured record of one conversion (sources, hashes, warnings).
MigrationWarning
    One structured, non-fatal conversion finding.
LegacyFormatError
    Raised for unrecognized or unsupported legacy payloads.

This module is a workspace migration tool, not a stable qphase_sde 2.x API.
QPhase does not retain old-major compatibility code after the project migration
has been verified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from qphase.core.errors import QPhaseError
from qphase.core.utils import canonical_json
from qphase.data import (
    ArtifactManifest,
    AxisRole,
    AxisSchema,
    DataKind,
    Dataset,
    ProductEntry,
    ProductSchema,
    VariableSchema,
    save_products,
)
from qphase.data.npz import (
    NpzChunkRecord,
    NpzVariableDescriptor,
    build_product_storage,
)
from qphase.data.store import (
    GENERIC_BUNDLE_ADAPTER_ID,
    GENERIC_BUNDLE_TYPE_ID,
    BundleDescriptor,
)

from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.products import split_payload_leaves as _split_payload_leaves
from qphase_sde.result import (
    SDEResult,
    _trajectory_product,
    bundle_from_result,
    recorded_distribution_versions,
)

__all__ = [
    "LegacyFormatError",
    "MigrationReport",
    "MigrationWarning",
    "migrate_legacy_result",
    "migrate_scan_artifact",
]

_LEGACY_RESULT_KEYS = {"data", "t0", "dt", "meta", "analysis", "trajectory_meta"}
_PICKLED_SCALARS = {"meta", "analysis", "trajectory_meta"}
_RAW_TRAJECTORY_KEYS = {"trajectories", "valid_length", "t0", "dt"}
_CHUNK_KEY = "data"

#: Adapter for unknown payloads: receives (key, value) and returns a plain
#: mapping the ``legacy_analysis/1`` bridge can split, or raises.
PayloadAdapter = Callable[[str, Any], Mapping[str, Any]]


class LegacyFormatError(QPhaseError):
    """Raised for unrecognized or unsupported legacy result payloads."""


@dataclass(frozen=True)
class MigrationWarning:
    """One structured, non-fatal conversion finding."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {"code": self.code, "message": self.message, **self.context}


@dataclass
class MigrationReport:
    """Structured record of one conversion run."""

    output: Path
    artifact_id: str
    products: list[str]
    sources: dict[str, str]
    warnings: list[MigrationWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "output": str(self.output),
            "artifact_id": self.artifact_id,
            "products": list(self.products),
            "sources": dict(self.sources),
            "warnings": [warning.to_dict() for warning in self.warnings],
        }

    def write(self, path: Path | str) -> None:
        """Write the report JSON next to (never inside) the sources."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_output(output_dir: Path, *sources: Path) -> None:
    resolved = output_dir.resolve()
    for source in sources:
        source = source.resolve()
        if (
            source == resolved
            or resolved in source.parents
            or source in resolved.parents
        ):
            raise LegacyFormatError(
                f"output directory {output_dir} overlaps the legacy source "
                f"{source}; choose a separate directory"
            )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LegacyFormatError(
            f"output directory {output_dir} is not empty; migration never "
            "overwrites existing artifacts"
        )


def _audit_npz(source: Path) -> set[str]:
    """Audit npz keys at the npy header level; reject unknown objects.

    Headers only, never payloads: pickled object scalars are inspected
    without unpickling them, and no object array is ever materialized.
    """
    import ast
    import zipfile

    keys: set[str] = set()
    object_keys: set[str] = set()
    try:
        with zipfile.ZipFile(source) as archive:
            for member_name in archive.namelist():
                with archive.open(member_name) as member:
                    if member.read(6) != b"\x93NUMPY":
                        raise LegacyFormatError(
                            f"{source}: member {member_name} is not an npy file"
                        )
                    major = member.read(1)[0]
                    member.read(1)  # minor version
                    length_bytes = member.read(2 if major == 1 else 4)
                    header_len = int.from_bytes(length_bytes, "little")
                    header = ast.literal_eval(
                        member.read(header_len).decode("latin1")
                    )
                key = member_name.removesuffix(".npy")
                keys.add(key)
                if header["descr"] == "|O":
                    object_keys.add(key)
    except LegacyFormatError:
        raise
    except Exception as exc:
        raise LegacyFormatError(
            f"cannot read legacy result {source}: {exc}"
        ) from exc
    unknown_objects = object_keys - _PICKLED_SCALARS
    if unknown_objects:
        raise LegacyFormatError(
            f"{source}: unknown object payloads {sorted(unknown_objects)}; "
            "pass an explicit adapter to convert them"
        )
    return keys


def _legacy_provenance(
    result: SDEResult, *, scan_grid: dict[str, list[float]] | None = None
) -> SDEProvenance:
    """Build provenance from a legacy result's meta (best effort, no guesses)."""
    meta = result.meta if isinstance(result.meta, dict) else {}
    trajectory = result.trajectory
    return SDEProvenance(
        t0=float(getattr(trajectory, "t0", meta.get("t0", 0.0)) or 0.0),
        dt=float(getattr(trajectory, "dt", meta.get("dt", 1.0)) or 1.0),
        master_seed=(
            int(meta["seed"])
            if isinstance(meta.get("seed"), int | np.integer)
            else None
        ),
        scan_grid=scan_grid or {},
    )


def _product_warnings(product_name: str, product: Dataset) -> list[MigrationWarning]:
    """Surface bridge-recorded losses as structured warnings."""
    warnings: list[MigrationWarning] = []
    dropped = product.attributes.get("dropped_keys") or []
    if dropped:
        warnings.append(
            MigrationWarning(
                code="payload-leaves-dropped",
                message=(
                    f"product {product_name!r}: non-JSON-safe payload leaves "
                    "were not migrated"
                ),
                context={"product": product_name, "keys": list(dropped)},
            )
        )
    return warnings


def _migration_provenance(
    sources: dict[str, str],
    legacy_format: str,
    warnings: list[MigrationWarning],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Manifest provenance shared by all migration outputs."""
    return {
        "migration": {
            "tool": "qphase_sde.runtime.migrate",
            "legacy_format": legacy_format,
            "sources": dict(sources),
            "warnings": [warning.to_dict() for warning in warnings],
            "versions": recorded_distribution_versions(),
            **dict(extra or {}),
        }
    }


def _raw_trajectory_product(source: Path) -> Dataset:
    """Convert a ``trajectory_set/1`` file to the trajectory product."""
    with np.load(source, allow_pickle=False) as npz:
        data = npz["trajectories"]
        valid_length = np.asarray(npz["valid_length"], dtype=np.int64)
        t0 = float(npz["t0"])
        dt = float(npz["dt"])
    if data.ndim != 3:
        raise LegacyFormatError(
            f"{source}: 'trajectories' must be (trajectory, time, channel), "
            f"got shape {data.shape}"
        )
    trajectory = type(
        "_RawTrajectory",
        (),
        {"data": data, "t0": t0, "dt": dt, "meta": {"valid_length": valid_length}},
    )()
    return _trajectory_product(trajectory, scan_size=1, n_traj_per_point=None)


def migrate_legacy_result(
    source: Path | str,
    output_dir: Path | str,
    *,
    adapter: PayloadAdapter | None = None,
    shard_target_bytes: int | None = None,
) -> MigrationReport:
    """Convert one ``sde_result/1`` or ``trajectory_set/1`` file to v4.

    The source file is hashed, never modified, and the output directory must
    be empty and disjoint from it. Unknown object payloads are rejected
    unless ``adapter`` maps them to plain bridge-compatible mappings.
    """
    source = Path(source)
    output_dir = Path(output_dir)
    _check_output(output_dir, source)
    keys = _audit_npz(source)
    sources = {source.name: _sha256_file(source)}
    warnings: list[MigrationWarning] = []

    if keys >= _RAW_TRAJECTORY_KEYS and "data" not in keys:
        products = {"trajectories": _raw_trajectory_product(source)}
        legacy_format = "trajectory_set/1"
        provenance_extra: dict[str, Any] = {}
    else:
        unknown = keys - _LEGACY_RESULT_KEYS
        if unknown:
            raise LegacyFormatError(
                f"{source}: unknown keys {sorted(unknown)}; not an SDE 1.x "
                "result file"
            )
        result = SDEResult.load(source)
        if adapter is not None:
            analysis: dict[str, Any] = {}
            for name, payload in result.analysis.items():
                if isinstance(payload, Mapping) or payload is None:
                    analysis[name] = payload
                else:
                    analysis[name] = adapter(str(name), payload)
            result.analysis = analysis
        bundle = bundle_from_result(
            result, provenance=_legacy_provenance(result)
        )
        products = bundle.products
        legacy_format = "sde_result/1"
        provenance_extra = {
            "sde": bundle.provenance.model_dump(mode="json"),
        }
        for dropped in bundle.metadata.get("dropped_products", []):
            warnings.append(
                MigrationWarning(
                    code="analysis-product-dropped",
                    message=(
                        f"analyser payload {dropped!r} has no numeric leaves "
                        "and cannot form a data product"
                    ),
                    context={"product": dropped},
                )
            )
    for name, product in products.items():
        warnings.extend(_product_warnings(name, product))

    save_kwargs: dict[str, Any] = {}
    if shard_target_bytes is not None:
        save_kwargs["shard_target_bytes"] = shard_target_bytes
    manifest = save_products(
        output_dir,
        products,
        provenance=_migration_provenance(
            sources, legacy_format, warnings, provenance_extra
        ),
        parents=[sources[source.name]],
        **save_kwargs,
    )
    return MigrationReport(
        output=output_dir,
        artifact_id=manifest.artifact_id,
        products=sorted(products),
        sources=sources,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Scan (artifact-manifest v2, per-point layout) streaming conversion
# ---------------------------------------------------------------------------


def _scan_seed(point_metas: list[dict[str, Any]]) -> int | None:
    """Take the shared legacy seed when all points agree (else None)."""
    seeds = {
        int(meta["seed"])
        for meta in point_metas
        if isinstance(meta.get("seed"), int | np.integer)
    }
    return seeds.pop() if len(seeds) == 1 else None


def _read_v2_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[Path]]:
    """Read and validate an artifact-manifest v2 (per-point layout only)."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LegacyFormatError(
            f"cannot read artifact manifest v2 {manifest_path}: {exc}"
        ) from exc
    if manifest.get("schema_version") != "2.0":
        raise LegacyFormatError(
            f"{manifest_path}: expected schema_version '2.0', got "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("layout") != "per_point":
        raise LegacyFormatError(
            f"{manifest_path}: only layout 'per_point' is supported, got "
            f"{manifest.get('layout')!r}"
        )
    files = manifest.get("files")
    shape = manifest.get("shape")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(shape, list)
        or len(shape) != 1
        or int(shape[0]) != len(files)
    ):
        raise LegacyFormatError(
            f"{manifest_path}: expected shape [<n>] matching {len(files or [])} "
            "per-point shard files"
        )
    shards = [manifest_path.parent / str(name) for name in files]
    for shard in shards:
        if not shard.is_file():
            raise LegacyFormatError(f"missing per-point shard {shard}")
    return manifest, shards


def _write_chunk(
    directory: Path,
    stem: str,
    variable: str,
    index: int,
    array: np.ndarray,
) -> NpzChunkRecord:
    """Write one point chunk file following the ``npz/3`` conventions."""
    chunk = np.ascontiguousarray(array)
    filename = f"{stem}__{variable}__{index:04d}.npz"
    np.savez(directory / filename, data=chunk)
    return NpzChunkRecord(
        file=filename,
        key=_CHUNK_KEY,
        logical_range=(index, index + 1),
        shape=tuple(chunk.shape),
        dtype=np.dtype(chunk.dtype).str,
    )


def _fused_trajectory_schema(
    point_schema: ProductSchema, n_points: int
) -> ProductSchema:
    """Rebuild the point trajectory schema with a full-size scan axis."""
    axes = [
        AxisSchema(name="scan", role=AxisRole.PARAMETER, size=n_points)
        if axis.name == "scan"
        else axis
        for axis in point_schema.axes
    ]
    return ProductSchema(
        kind=point_schema.kind,
        axes=axes,
        variables=list(point_schema.variables),
        attributes=dict(point_schema.attributes),
    )


def _fused_analysis_schema(
    name: str,
    point_splits: list[tuple[dict[str, np.ndarray], dict[str, Any], list[str]]],
    n_points: int,
    warnings: list[MigrationWarning],
) -> tuple[ProductSchema | None, list[str]]:
    """Build the fused scan schema for one analyser's per-point splits.

    Returns the schema (or None when no variable survives) plus the list of
    keys demoted to per-point metadata because their shapes differ across
    the scan.
    """
    key_sets = [set(split[0]) for split in point_splits]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        warnings.append(
            MigrationWarning(
                code="scan-payload-keys-differ",
                message=(
                    f"analyser {name!r}: per-point payload keys differ across "
                    "the scan; product skipped"
                ),
                context={"product": name, "key_sets": [sorted(k) for k in key_sets]},
            )
        )
        return None, []
    arrays: dict[str, np.ndarray] = {}
    demoted: list[str] = []
    for key in sorted(key_sets[0]):
        shapes = {split[0][key].shape for split in point_splits}
        dtypes = {np.dtype(split[0][key].dtype).str for split in point_splits}
        if len(shapes) == 1 and len(dtypes) == 1:
            arrays[key] = point_splits[0][0][key]
        else:
            demoted.append(key)
    positional: dict[str, AxisSchema] = {}
    variables: list[VariableSchema] = []
    for key, array in arrays.items():
        dims = ["scan"]
        for position, extent in enumerate(array.shape):
            axis_name = f"{key}.dim{position}"
            if axis_name not in positional:
                positional[axis_name] = AxisSchema(
                    name=axis_name,
                    role=AxisRole.INDEX,
                    size=int(extent),
                )
            elif positional[axis_name].size != int(extent):
                raise LegacyFormatError(
                    f"analyser {name!r}: positional axis {axis_name!r} has "
                    f"conflicting sizes {positional[axis_name].size} and "
                    f"{int(extent)}"
                )
            dims.append(axis_name)
        dtype = np.dtype(array.dtype)
        variables.append(
            VariableSchema(
                name=key,
                dtype=dtype.str,
                value_domain="complex" if dtype.kind == "c" else "real",
                dims=tuple(dims),
            )
        )
    if not variables:
        warnings.append(
            MigrationWarning(
                code="analysis-product-dropped",
                message=(
                    f"analyser {name!r}: no numeric payload leaves survive "
                    "the scan; product skipped"
                ),
                context={"product": name},
            )
        )
        return None, demoted
    dropped = sorted({key for split in point_splits for key in split[2]})
    payload_meta: dict[str, Any] = {}
    meta_keys = {key for split in point_splits for key in split[1]}
    for key in sorted(meta_keys):
        payload_meta[key] = [split[1].get(key) for split in point_splits]
    for key in demoted:
        values = [split[0][key].tolist() for split in point_splits]
        try:
            canonical_json(values)
        except (TypeError, ValueError):
            warnings.append(
                MigrationWarning(
                    code="payload-leaves-dropped",
                    message=(
                        f"analyser {name!r}: ragged leaf {key!r} is not "
                        "JSON-safe and was not migrated"
                    ),
                    context={"product": name, "keys": [key]},
                )
            )
            continue
        payload_meta[key] = values
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=n_points),
            *positional.values(),
        ],
        variables=variables,
        attributes={
            "bridge": "legacy_analysis/1",
            "source_analyser": str(name),
            "dropped_keys": dropped,
            "payload_meta": payload_meta,
            "per_point_meta": sorted([*meta_keys, *demoted]),
        },
    )
    return schema, demoted


def migrate_scan_artifact(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    adapter: PayloadAdapter | None = None,
) -> MigrationReport:
    """Convert an ``sde_scan/2`` per-point artifact to v4, streaming per point.

    Each legacy shard is read twice (structure pass, chunk pass) and never
    concatenated as a whole: every point contributes one chunk per variable
    along the scan axis, so peak memory stays within one shard plus one
    output chunk.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    v2, shards = _read_v2_manifest(manifest_path)
    _check_output(output_dir, manifest_path, *shards)
    n_points = len(shards)
    sources = {manifest_path.name: _sha256_file(manifest_path)}
    warnings: list[MigrationWarning] = []

    # -- pass 1: structure only; arrays are released immediately -------------
    point_metas: list[dict[str, Any]] = []
    trajectory_info: tuple[tuple[int, ...], str, float, float] | None = None
    SplitTriple = tuple[dict[str, np.ndarray], dict[str, Any], list[str]]
    analysers: dict[str, list[SplitTriple]] = {}
    for shard in shards:
        sources[shard.name] = _sha256_file(shard)
        _audit_npz(shard)
        result = SDEResult.load(shard)
        if adapter is not None:
            for name, payload in result.analysis.items():
                if not isinstance(payload, Mapping) and payload is not None:
                    result.analysis[name] = adapter(str(name), payload)
        point_metas.append(dict(result.meta))
        if result.trajectory is not None:
            data = np.asarray(
                getattr(result.trajectory, "data", result.trajectory)
            )
            info = (
                tuple(data.shape),
                np.dtype(data.dtype).str,
                float(getattr(result.trajectory, "t0", 0.0)),
                float(getattr(result.trajectory, "dt", 1.0)),
            )
            if trajectory_info is None:
                trajectory_info = info
            elif trajectory_info != info:
                raise LegacyFormatError(
                    f"{shard}: trajectory shape/dtype/grid {info} differs from "
                    f"the first point {trajectory_info}; fused conversion "
                    "requires uniform points"
                )
        for name, payload in result.analysis.items():
            splits = analysers.setdefault(str(name), [])
            splits.append(_split_payload_leaves(payload))
    for name, splits in analysers.items():
        if len(splits) != n_points:
            raise LegacyFormatError(
                f"analyser {name!r} appears in only {len(splits)} of "
                f"{n_points} points"
            )

    # -- fused schemas --------------------------------------------------------
    scan_axes_meta = v2.get("axes", {})
    scan_grid = {
        str(name): [float(value) for value in values]
        for name, values in scan_axes_meta.items()
    }
    trajectory_schema: ProductSchema | None = None
    if trajectory_info is not None:
        first = SDEResult.load(shards[0]).trajectory
        point_product = _trajectory_product(
            first, scan_size=1, n_traj_per_point=None
        )
        trajectory_schema = _fused_trajectory_schema(
            point_product.schema, n_points
        )
    analysis_schemas: dict[str, ProductSchema] = {}
    demoted_keys: dict[str, list[str]] = {}
    for name, splits in analysers.items():
        schema, demoted = _fused_analysis_schema(
            name, splits, n_points, warnings
        )
        if schema is not None:
            analysis_schemas[name] = schema
            demoted_keys[name] = demoted

    # -- pass 2: stream one chunk per point per variable ----------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ProductEntry] = []
    product_names = (
        ["trajectories"] if trajectory_schema is not None else []
    ) + sorted(analysis_schemas)
    for product_index, product_name in enumerate(product_names):
        stem = f"{product_index:02d}_{product_name}"
        records: dict[str, list[NpzChunkRecord]] = {}
        for point_index, shard in enumerate(shards):
            result = SDEResult.load(shard)
            if adapter is not None:
                for name, payload in result.analysis.items():
                    if not isinstance(payload, Mapping) and payload is not None:
                        result.analysis[name] = adapter(str(name), payload)
            if product_name == "trajectories":
                assert trajectory_schema is not None
                product = _trajectory_product(
                    result.trajectory, scan_size=1, n_traj_per_point=None
                )
                for variable in trajectory_schema.variables:
                    array = np.asarray(
                        product.handle(variable.name).materialize(
                            "cpu", copy_policy="allow"
                        )
                    )
                    records.setdefault(variable.name, []).append(
                        _write_chunk(
                            output_dir,
                            stem,
                            variable.name,
                            point_index,
                            array,
                        )
                    )
            else:
                schema = analysis_schemas[product_name]
                split = _split_payload_leaves(result.analysis[product_name])
                for variable in schema.variables:
                    array = np.asarray(split[0][variable.name])
                    records.setdefault(variable.name, []).append(
                        _write_chunk(
                            output_dir,
                            stem,
                            variable.name,
                            point_index,
                            array.reshape(1, *array.shape),
                        )
                    )
        schema = (
            trajectory_schema
            if product_name == "trajectories"
            else analysis_schemas[product_name]
        )
        assert schema is not None  # None-schema products are never listed
        n_points = len(shards)
        variables: dict[str, NpzVariableDescriptor] = {}
        for variable in schema.variables:
            chunks = records[variable.name]
            full_shape = (n_points, *chunks[0].shape[1:])
            variables[variable.name] = NpzVariableDescriptor(
                full_shape=full_shape,
                dtype=chunks[0].dtype,
                chunk_axis=variable.dims[0],
                chunks=chunks,
            )
        storage = build_product_storage(schema, variables)
        entries.append(
            ProductEntry(
                name=product_name,
                product_schema=schema,
                storage=storage,
            )
        )

    provenance = _migration_provenance(
        sources,
        "sde_scan/2",
        warnings,
        {
            "sde": SDEProvenance(
                t0=trajectory_info[2] if trajectory_info is not None else 0.0,
                dt=trajectory_info[3] if trajectory_info is not None else None,
                master_seed=_scan_seed(point_metas),
                scan_grid=scan_grid,
            ).model_dump(mode="json"),
            "scan": {
                "axes": scan_axes_meta,
                "shape": list(v2["shape"]),
                "n_traj_per_point": (
                    trajectory_info[0][0] if trajectory_info is not None else None
                ),
            },
        },
    )
    artifact_id = uuid4().hex
    bundle = BundleDescriptor(
        type_id=GENERIC_BUNDLE_TYPE_ID,
        adapter_id=GENERIC_BUNDLE_ADAPTER_ID,
        descriptor_schema=GENERIC_BUNDLE_TYPE_ID,
        descriptor={},
        product_roles=(
            {"trajectories": "trajectories"}
            if any(entry.name == "trajectories" for entry in entries)
            else {}
        ),
    )
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        created_at=datetime.now(UTC).isoformat(),
        bundle=bundle,
        products=entries,
        provenance=provenance,
        parents=[sources[manifest_path.name]],
    )
    manifest.write(output_dir)
    return MigrationReport(
        output=output_dir,
        artifact_id=artifact_id,
        products=product_names,
        sources=sources,
        warnings=warnings,
    )
