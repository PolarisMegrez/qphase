"""qphase: Artifact Manifest v3 and Storage Adapter Contract
---------------------------------------------------------
The v3 artifact manifest is the single public restore entry point for
persisted data products. It records, per artifact:

- artifact schema/version/id/content hash and creation time;
- product names with their full :class:`ProductSchema`;
- the storage adapter id and the per-variable chunk/shard mapping;
- JSON provenance (plugin/config/backend fingerprints, conventions);
- parent artifact ids.

Trust model:

- manifests and refs name *registered adapter ids* (``name/version``), never
  Python code — restoring an artifact cannot import arbitrary modules;
- all payload paths are validated as artifact-relative POSIX paths and
  re-resolved under the artifact root at open time (no ``..``, absolute,
  drive, UNC or symlink escapes);
- integrity is verified in three layers: the manifest/product layer is
  re-validated (cross-field and content hashes) at parse time, and each
  payload chunk is hash/dtype/shape-verified on first read. ``content_hash``
  is an integrity check against accidental corruption, not a digital
  signature.

Storage adapters implement :class:`StorageAdapterProtocol`; the NPZ 2.x
adapter (:mod:`qphase.data.npz`) is the reference implementation. Other
adapters (e.g. Zarr) can register through :func:`register_adapter` without
changing the manifest format; registration never silently overwrites an
existing adapter id.

All manifest data is plain JSON; payload files use native dtypes only —
restoring never requires ``allow_pickle``.

Public API
----------
ChunkRecord
    One persisted array chunk of a variable.
ProductStorage
    Adapter id plus the variable→chunk mapping of one product.
ProductEntry
    One named product inside an artifact manifest.
BundleDescriptor
    Persisted bundle type/adapter plus product roles.
ArtifactManifestV3
    The v3 artifact manifest.
StorageAdapterProtocol
    Persistence adapter contract.
save_products
    Persist typed datasets and write the v3 manifest.
load_products
    Reopen an artifact directory as lazily-backed datasets.
load_bundle
    Restore an artifact directory as a (generic or adapted) bundle object.
register_adapter
    Register a storage adapter implementation.
register_bundle_adapter
    Register a bundle adapter restoring concrete bundle types.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..core.utils import canonical_json
from .artifact import ArtifactRef
from .datasets import Dataset, SpectralDataset, StatisticsDataset, TimeSeriesDataset
from .errors import (
    ArtifactAdapterError,
    ArtifactCorruptError,
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactUnsupportedError,
)
from .kinds import DataKind
from .product import RuntimeProductBacking
from .resolver import ArtifactResolverProtocol
from .schema import ProductSchema, VariableSchema

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "GENERIC_BUNDLE_ADAPTER_ID",
    "GENERIC_BUNDLE_TYPE_ID",
    "MANIFEST_FILENAME",
    "ArtifactManifestV3",
    "BundleDescriptor",
    "ChunkRecord",
    "ProductEntry",
    "ProductStorage",
    "StorageAdapterProtocol",
    "artifact_content_hash",
    "chunk_content_hash",
    "load_bundle",
    "load_products",
    "product_content_hash",
    "register_adapter",
    "register_bundle_adapter",
    "resolve_artifact_path",
    "save_products",
    "validate_artifact_relative_path",
]

#: Manifest schema identifier for the qphase 2.x artifact format.
ARTIFACT_SCHEMA_VERSION: Literal["qphase.artifact/3"] = "qphase.artifact/3"

#: Manifest file name inside every artifact directory.
MANIFEST_FILENAME = "artifact_manifest.json"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:$")

#: Default shard target: 64 MiB of payload per chunk.
DEFAULT_SHARD_TARGET_BYTES = 64 * (1 << 20)


def _validate_sha256_digest(value: str) -> str:
    if not _SHA256_PATTERN.match(value):
        raise ValueError("expected a 64-character lowercase SHA-256 digest")
    return value


# -- path safety --------------------------------------------------------------


def validate_artifact_relative_path(value: str) -> str:
    """Validate one manifest-relative POSIX payload path.

    Payload paths must be non-empty relative POSIX paths: no absolute paths,
    drive letters, UNC prefixes, empty/``.``/``..`` parts, backslashes or
    NUL bytes. This is part of the loader's trusted path, not only of
    development-time validators.
    """
    if not value or "\x00" in value:
        raise ValueError("artifact paths must be non-empty and NUL-free")
    if "\\" in value:
        raise ValueError(
            f"artifact paths must use POSIX separators, got {value!r}"
        )
    if value.startswith("/"):
        raise ValueError(f"absolute artifact paths are not allowed: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"artifact paths must not contain empty/'.'/'..' parts: {value!r}"
        )
    if ":" in parts[0] or _DRIVE_PATTERN.match(parts[0]):
        raise ValueError(
            f"drive/UNC artifact paths are not allowed: {value!r}"
        )
    return value


def resolve_artifact_path(root: Path | str, relative: str) -> Path:
    """Resolve a manifest-relative path, refusing escapes.

    Re-validates the relative path and checks the fully resolved path stays
    under the resolved artifact root, so symlink/junction escapes are
    rejected at open time as well.
    """
    validate_artifact_relative_path(relative)
    root_resolved = Path(root).resolve()
    resolved = (root_resolved / relative).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ArtifactCorruptError(
            f"artifact path escapes the artifact root: {relative!r}"
        )
    return resolved


# -- layered content hashes -----------------------------------------------------


def chunk_content_hash(
    array: np.ndarray, logical_range: tuple[int, int] | None = None
) -> str:
    """Hash one chunk: canonical header + NUL + C-order payload bytes.

    The header pins dtype (including byte order), shape, memory order and
    the logical selection, so the same payload bytes reinterpreted as a
    different dtype or shape never match. NaN payloads hash like any other
    bit pattern; 0-d arrays are trivially C-contiguous.
    """
    contiguous = (
        np.ascontiguousarray(array) if array.ndim else np.asarray(array)
    )
    header = canonical_json(
        {
            "dtype": np.dtype(contiguous.dtype).str,
            "shape": list(contiguous.shape),
            "order": "C",
            "logical_range": list(logical_range)
            if logical_range is not None
            else None,
        }
    )
    return hashlib.sha256(
        header.encode("utf-8") + b"\x00" + contiguous.tobytes()
    ).hexdigest()


def product_content_hash(
    name: str, product_schema: ProductSchema, storage: ProductStorage
) -> str:
    """Hash a product over its canonical storage descriptor."""
    listing = {
        "adapter": storage.adapter,
        "name": name,
        "schema": product_schema.fingerprint(),
        "variables": {
            variable: [
                {
                    "axis0_range": list(chunk.axis0_range)
                    if chunk.axis0_range is not None
                    else None,
                    "dtype": chunk.dtype,
                    "file": chunk.file,
                    "key": chunk.key,
                    "sha256": chunk.sha256,
                    "shape": list(chunk.shape),
                }
                for chunk in chunks
            ]
            for variable, chunks in sorted(storage.variables.items())
        },
    }
    return hashlib.sha256(canonical_json(listing).encode("utf-8")).hexdigest()


def artifact_content_hash(
    bundle: Mapping[str, Any] | None,
    products: Sequence[ProductEntry],
    provenance: Mapping[str, Any],
    parents: Sequence[str],
) -> str:
    """Hash an artifact over bundle, products, provenance and parents."""
    listing = {
        "bundle": bundle,
        "parents": sorted(parents),
        "products": [
            {"name": entry.name, "sha256": entry.sha256}
            for entry in sorted(products, key=lambda entry: entry.name)
        ],
        "provenance": provenance,
    }
    return hashlib.sha256(canonical_json(listing).encode("utf-8")).hexdigest()


# -- manifest models ------------------------------------------------------------


class ChunkRecord(BaseModel):
    """One persisted array chunk of a variable.

    Chunks split a variable contiguously along its first dimension;
    ``axis0_range`` records the [start, stop) window the chunk covers in the
    full variable (None when the chunk holds the whole variable). ``file`` is
    relative to the artifact directory so artifacts can be relocated.
    """

    model_config = ConfigDict(extra="forbid")

    file: str
    key: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    axis0_range: tuple[int, int] | None = None

    @field_validator("file")
    @classmethod
    def _check_file(cls, value: str) -> str:
        return validate_artifact_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _validate_sha256_digest(value)


class ProductStorage(BaseModel):
    """Storage description of one product: adapter id and chunk mapping."""

    model_config = ConfigDict(extra="forbid")

    adapter: str = Field(min_length=1)
    variables: dict[str, list[ChunkRecord]]


class ProductEntry(BaseModel):
    """One named product inside an artifact manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    product_schema: ProductSchema
    storage: ProductStorage
    sha256: str = Field(
        description="Content hash over the canonical storage descriptor."
    )

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _validate_sha256_digest(value)


#: Bundle type of a plain product collection (no resource-specific semantics).
GENERIC_BUNDLE_TYPE_ID = "generic.dataset_bundle/1"

#: Adapter id of the built-in generic bundle builder.
GENERIC_BUNDLE_ADAPTER_ID = "generic/1"


class BundleDescriptor(BaseModel):
    """Descriptor restoring how a product collection forms one result.

    ``type_id`` names the bundle type (``generic.dataset_bundle/1``,
    ``sde.bundle/1``, ...); ``adapter_id`` selects the *registered* bundle
    adapter used to rebuild a concrete bundle (a trusted registry id, never
    a code path); ``descriptor`` holds adapter-validated JSON (scan layout,
    bundle metadata); ``product_roles`` maps stable semantic roles to
    job-local product names.
    """

    model_config = ConfigDict(extra="forbid")

    type_id: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    descriptor_schema: str = Field(min_length=1)
    descriptor: dict[str, Any] = Field(default_factory=dict)
    product_roles: dict[str, str] = Field(default_factory=dict)

    @field_validator("descriptor")
    @classmethod
    def _check_descriptor_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("bundle descriptor must be JSON-serializable") from exc
        return value


class ArtifactManifestV3(BaseModel):
    """The v3 artifact manifest: the public restore entry of an artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["qphase.artifact/3"] = ARTIFACT_SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    created_at: str
    bundle: BundleDescriptor
    products: list[ProductEntry] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    parents: list[str] = Field(default_factory=list)
    content_hash: str = Field(
        description="SHA-256 over the canonical bundle/product/provenance/"
        "parent listing. An integrity check, not a digital signature."
    )

    @field_validator("content_hash")
    @classmethod
    def _check_content_hash(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"created_at must be an ISO 8601 timestamp, got {value!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise ValueError("created_at must carry timezone information")
        return value

    @field_validator("products")
    @classmethod
    def _check_unique_products(cls, value: list[ProductEntry]) -> list[ProductEntry]:
        names = [entry.name for entry in value]
        if len(set(names)) != len(names):
            raise ValueError("artifact product names must be unique")
        return value

    @field_validator("parents")
    @classmethod
    def _check_unique_parents(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("artifact parents must be unique")
        return value

    @field_validator("provenance")
    @classmethod
    def _check_provenance_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest provenance must be JSON-serializable") from exc
        return value

    def product_entry(self, name: str) -> ProductEntry:
        """Return the entry of the product with the given name."""
        for entry in self.products:
            if entry.name == name:
                return entry
        raise KeyError(f"unknown product {name!r}")

    def product_ref(self, name: str) -> ArtifactRef:
        """Build the durable reference of one product."""
        entry = self.product_entry(name)
        return ArtifactRef(
            artifact_id=self.artifact_id,
            product_name=name,
            product_schema=entry.product_schema,
            storage_adapter=entry.storage.adapter,
            content_hash=entry.sha256,
        )

    def write(self, directory: Path | str) -> Path:
        """Write the manifest JSON into the artifact directory."""
        path = Path(directory) / MANIFEST_FILENAME
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def read(cls, directory: Path | str) -> ArtifactManifestV3:
        """Read and fully validate the v3 manifest of an artifact directory.

        Raises typed errors: :class:`ArtifactNotFoundError` when no manifest
        exists, :class:`ArtifactUnsupportedError` for other schema versions
        and :class:`ArtifactCorruptError` for parse, cross-field or hash
        failures.
        """
        path = Path(directory) / MANIFEST_FILENAME
        if not path.exists():
            raise ArtifactNotFoundError(f"no artifact manifest at {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactCorruptError(
                f"failed to parse artifact manifest {path}: {exc}"
            ) from exc
        version = raw.get("schema_version") if isinstance(raw, dict) else None
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactUnsupportedError(
                f"unsupported artifact schema version {version!r}; expected "
                f"{ARTIFACT_SCHEMA_VERSION!r}"
            )
        try:
            manifest = cls.model_validate(raw)
        except ValidationError as exc:
            raise ArtifactCorruptError(
                f"artifact manifest {path} is invalid: {exc}"
            ) from exc
        manifest._cross_validate(Path(directory))
        return manifest

    def _cross_validate(self, directory: Path) -> None:
        """Cross-field, path-safety and content-hash validation layer."""
        product_names = {entry.name for entry in self.products}
        unknown_roles = sorted(
            target
            for target in self.bundle.product_roles.values()
            if target not in product_names
        )
        if unknown_roles:
            raise ArtifactCorruptError(
                f"bundle product_roles reference unknown products {unknown_roles}"
            )
        files: dict[str, str] = {}
        for entry in self.products:
            _validate_product_entry(entry)
            for variable, chunks in entry.storage.variables.items():
                owner = f"{entry.name}.{variable}"
                for chunk in chunks:
                    previous = files.setdefault(chunk.file, owner)
                    if previous != owner:
                        raise ArtifactCorruptError(
                            f"chunk file {chunk.file!r} is referenced by both "
                            f"{previous} and {owner}"
                        )
                    resolve_artifact_path(directory, chunk.file)
            expected = product_content_hash(
                entry.name, entry.product_schema, entry.storage
            )
            if entry.sha256 != expected:
                raise ArtifactCorruptError(
                    f"product {entry.name!r} content hash mismatch: the "
                    "manifest was modified after writing"
                )
        expected_content = artifact_content_hash(
            self.bundle.model_dump(mode="json"),
            self.products,
            self.provenance,
            self.parents,
        )
        if self.content_hash != expected_content:
            raise ArtifactCorruptError(
                "artifact content hash mismatch: the manifest was modified "
                "after writing"
            )


# -- cross-field validation -----------------------------------------------------


def _validate_product_entry(entry: ProductEntry) -> None:
    """Validate storage/schema consistency of one product entry."""
    schema_vars = {
        variable.name: variable for variable in entry.product_schema.variables
    }
    storage_vars = entry.storage.variables
    missing = sorted(set(schema_vars) - set(storage_vars))
    extra = sorted(set(storage_vars) - set(schema_vars))
    if missing or extra:
        raise ArtifactCorruptError(
            f"storage variables of product {entry.name!r} do not match the "
            f"product schema (missing: {missing}, extra: {extra})"
        )
    for name, variable in schema_vars.items():
        _validate_variable_chunks(
            entry.name, variable, entry.product_schema, storage_vars[name]
        )


def _validate_variable_chunks(
    product: str,
    variable: VariableSchema,
    schema: ProductSchema,
    chunks: list[ChunkRecord],
) -> None:
    """Validate chunk dtype/shape/coverage against the variable schema."""
    label = f"variable {variable.name!r} of product {product!r}"
    if not chunks:
        raise ArtifactCorruptError(f"{label} has no storage chunks")
    expected_dtype = np.dtype(variable.dtype)
    axes = [schema.axis(dim) for dim in variable.dims]
    for chunk in chunks:
        if np.dtype(chunk.dtype) != expected_dtype:
            raise ArtifactCorruptError(
                f"{label} chunk {chunk.file!r} has dtype {chunk.dtype!r}, "
                f"expected {expected_dtype.str!r}"
            )
        if len(chunk.shape) != len(variable.dims):
            raise ArtifactCorruptError(
                f"{label} chunk {chunk.file!r} has rank {len(chunk.shape)}, "
                f"expected {len(variable.dims)}"
            )
    if len(chunks) == 1:
        chunk = chunks[0]
        if chunk.axis0_range is not None:
            raise ArtifactCorruptError(
                f"{label} is unsharded but declares an axis0_range"
            )
        for axis, size in zip(axes, chunk.shape, strict=True):
            if axis.size is not None and axis.size != size:
                raise ArtifactCorruptError(
                    f"{label} chunk shape {chunk.shape} does not match the "
                    f"closed axis {axis.name!r} of size {axis.size}"
                )
        return
    if not variable.dims:
        raise ArtifactCorruptError(f"{label} is scalar but has multiple chunks")
    expected_start = 0
    for chunk in chunks:
        if chunk.axis0_range is None:
            raise ArtifactCorruptError(
                f"{label} chunk {chunk.file!r} misses its axis0_range"
            )
        start, stop = chunk.axis0_range
        if not 0 <= start < stop:
            raise ArtifactCorruptError(
                f"{label} chunk {chunk.file!r} has out-of-bounds range "
                f"[{start}, {stop})"
            )
        if chunk.shape[0] != stop - start:
            raise ArtifactCorruptError(
                f"{label} chunk {chunk.file!r} shape {chunk.shape} does not "
                f"match its range [{start}, {stop})"
            )
        if start != expected_start:
            raise ArtifactCorruptError(
                f"{label} chunks overlap, gap or are out of order at "
                f"[{start}, {stop}); expected start {expected_start}"
            )
        expected_start = stop
    tail = chunks[0].shape[1:]
    for chunk in chunks[1:]:
        if chunk.shape[1:] != tail:
            raise ArtifactCorruptError(
                f"{label} chunks have inconsistent trailing dimensions"
            )
    for axis, size in zip(axes[1:], tail, strict=True):
        if axis.size is not None and axis.size != size:
            raise ArtifactCorruptError(
                f"{label} chunk shape does not match the closed axis "
                f"{axis.name!r} of size {axis.size}"
            )
    if axes[0].size is not None and expected_start != axes[0].size:
        raise ArtifactCorruptError(
            f"{label} chunks cover [0, {expected_start}) but the closed axis "
            f"{axes[0].name!r} has size {axes[0].size}"
        )


# -- storage adapter registry ---------------------------------------------------


@runtime_checkable
class StorageAdapterProtocol(Protocol):
    """Persistence adapter contract for typed data products.

    Adapters write a runtime-backed dataset as chunk files plus a storage
    record, reopen a storage record as a lazily-reading runtime backing and
    restore artifact refs. The NPZ 2.x adapter is the reference
    implementation; further adapters (e.g. Zarr) plug in without manifest
    changes.
    """

    @property
    def adapter_id(self) -> str:
        """Stable adapter identifier recorded in the manifest."""
        ...

    def write_product(
        self,
        name: str,
        dataset: Dataset,
        directory: Path,
        *,
        shard_target_bytes: int,
        file_stem: str,
    ) -> ProductStorage:
        """Persist one product and return its storage record."""
        ...

    def open_product(
        self, entry: ProductEntry, directory: Path
    ) -> RuntimeProductBacking:
        """Open a stored product as a lazily-reading runtime backing."""
        ...

    def open_ref(
        self,
        ref: ArtifactRef,
        *,
        resolver: ArtifactResolverProtocol | None = None,
    ) -> RuntimeProductBacking:
        """Open the product referenced by an artifact ref.

        ``resolver`` turns the ref's artifact id into an on-disk location;
        adapters fall back to the process-default resolver when omitted.
        """
        ...


_ADAPTERS: dict[str, StorageAdapterProtocol] = {}


def register_adapter(adapter: StorageAdapterProtocol) -> None:
    """Register a storage adapter under its adapter id.

    Registration never silently overwrites an existing id: upgrades must use
    a new adapter id (and descriptor schema), so persisted artifacts keep
    their original semantics.
    """
    existing = _ADAPTERS.get(adapter.adapter_id)
    if existing is not None and existing is not adapter:
        raise ArtifactAdapterError(
            f"storage adapter {adapter.adapter_id!r} is already registered; "
            "refusing to overwrite it"
        )
    _ADAPTERS[adapter.adapter_id] = adapter


def _resolve_adapter(adapter_id: str) -> StorageAdapterProtocol:
    if adapter_id not in _ADAPTERS:
        # Ensure the built-in NPZ adapter is registered (exactly once).
        from .npz import NpzStorageAdapter

        if NpzStorageAdapter.ADAPTER_ID not in _ADAPTERS:
            register_adapter(NpzStorageAdapter())
    try:
        return _ADAPTERS[adapter_id]
    except KeyError:
        raise ArtifactAdapterError(
            f"unknown storage adapter {adapter_id!r}; registered: {sorted(_ADAPTERS)}"
        ) from None


_DATASET_BY_KIND: dict[DataKind, type[Dataset]] = {
    DataKind.TIME_SERIES: TimeSeriesDataset,
    DataKind.SPECTRAL: SpectralDataset,
    DataKind.STATISTICS: StatisticsDataset,
}


def _sanitize(name: str) -> str:
    return _SAFE_STEM_PATTERN.sub("_", name) or "product"


# -- write/read entry points ----------------------------------------------------


def save_products(
    directory: Path | str,
    products: Mapping[str, Dataset],
    *,
    provenance: Mapping[str, Any] | None = None,
    parents: Sequence[str] = (),
    artifact_id: str | None = None,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    bundle: BundleDescriptor | None = None,
) -> ArtifactManifestV3:
    """Persist typed datasets and write the v3 artifact manifest.

    Artifact-backed datasets are fully materialized first (an explicit load,
    never an implicit one); device-resident payloads are copied to the host
    with an explicit ``copy_policy="allow"`` at save time. Every variable is
    stored as one or more chunks split along its first dimension. An empty
    mapping is allowed and produces a manifest with no products (e.g. a
    simulation that only yields scalar metadata). Without an explicit
    ``bundle`` descriptor a generic bundle is recorded, restoring the
    products as a :class:`~qphase.data.bundle.GenericDataBundle`.
    """
    if not isinstance(products, Mapping):
        raise TypeError(
            f"products must be a mapping of name to Dataset, got "
            f"{type(products).__name__}"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    from .npz import NpzStorageAdapter  # local: npz depends on this module
    from .resolver import register_artifact_location

    adapter = _resolve_adapter(NpzStorageAdapter.ADAPTER_ID)
    provenance_dict = dict(provenance or {})
    parent_list = [str(parent) for parent in parents]
    entries: list[ProductEntry] = []
    used_files: set[str] = set()
    for index, (name, dataset) in enumerate(products.items()):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"product {name!r} must be a Dataset, got {type(dataset).__name__}"
            )
        if dataset.is_artifact_backed:
            dataset = dataset.materialize()
        storage = adapter.write_product(
            name,
            dataset,
            directory,
            shard_target_bytes=shard_target_bytes,
            file_stem=f"{index:02d}_{_sanitize(name)}",
        )
        for chunks in storage.variables.values():
            for chunk in chunks:
                if chunk.file in used_files:
                    raise ArtifactError(
                        f"writer produced a duplicate chunk file name {chunk.file!r}"
                    )
                used_files.add(chunk.file)
        entries.append(
            ProductEntry(
                name=name,
                product_schema=dataset.schema,
                storage=storage,
                sha256=product_content_hash(name, dataset.schema, storage),
            )
        )

    if bundle is None:
        bundle = BundleDescriptor(
            type_id=GENERIC_BUNDLE_TYPE_ID,
            adapter_id=GENERIC_BUNDLE_ADAPTER_ID,
            descriptor_schema=GENERIC_BUNDLE_TYPE_ID,
            descriptor={},
            product_roles={name: name for name in products},
        )
    if artifact_id is None:
        artifact_id = hashlib.sha256(
            f"{datetime.now(UTC).isoformat()}-{sorted(products)}".encode()
        ).hexdigest()[:16]
    manifest = ArtifactManifestV3(
        artifact_id=artifact_id,
        created_at=datetime.now(UTC).isoformat(),
        bundle=bundle,
        products=entries,
        provenance=provenance_dict,
        parents=parent_list,
        content_hash=artifact_content_hash(
            bundle.model_dump(mode="json"), entries, provenance_dict, parent_list
        ),
    )
    manifest.write(directory)
    register_artifact_location(manifest.artifact_id, directory)
    return manifest


def load_products(directory: Path | str) -> dict[str, Dataset]:
    """Reopen an artifact directory as lazily-backed datasets.

    Payloads are not read: the returned datasets are backed by lazy handles
    that load chunks on demand (and only the chunks a selection touches).
    The manifest is fully validated first; unknown storage adapters raise
    :class:`ArtifactAdapterError`.
    """
    directory = Path(directory)
    manifest = ArtifactManifestV3.read(directory)
    return _load_products(manifest, directory)


def _load_products(manifest: ArtifactManifestV3, directory: Path) -> dict[str, Dataset]:
    """Open the products of an already-validated manifest."""
    from .resolver import register_artifact_location

    result: dict[str, Dataset] = {}
    for entry in manifest.products:
        adapter = _resolve_adapter(entry.storage.adapter)
        backing = adapter.open_product(entry, directory)
        dataset_class = _DATASET_BY_KIND[entry.product_schema.kind]
        result[entry.name] = dataset_class(
            entry.product_schema,
            backing,
            provenance={
                **manifest.provenance,
                "artifact_id": manifest.artifact_id,
                "product": entry.name,
            },
        )
    register_artifact_location(manifest.artifact_id, directory)
    return result


# -- bundle restore -------------------------------------------------------------


def _generic_bundle_builder(
    manifest: ArtifactManifestV3, products: dict[str, Dataset]
) -> Any:
    from .bundle import GenericDataBundle

    return GenericDataBundle(
        products,
        manifest.bundle,
        provenance=manifest.provenance,
        metadata={"artifact_id": manifest.artifact_id},
    )


_BUNDLE_ADAPTERS: dict[str, Any] = {GENERIC_BUNDLE_ADAPTER_ID: _generic_bundle_builder}


def register_bundle_adapter(adapter_id: str, builder: Any) -> None:
    """Register a bundle adapter restoring concrete bundles by adapter id.

    ``builder`` receives ``(manifest, products)`` and returns the restored
    bundle object. Registration never silently overwrites an existing id.
    """
    existing = _BUNDLE_ADAPTERS.get(adapter_id)
    if existing is not None and existing is not builder:
        raise ArtifactAdapterError(
            f"bundle adapter {adapter_id!r} is already registered; refusing "
            "to overwrite it"
        )
    _BUNDLE_ADAPTERS[adapter_id] = builder


def load_bundle(directory: Path | str) -> Any:
    """Restore a v3 artifact directory as a bundle object.

    The manifest's bundle adapter id selects the registered builder; without
    a registered builder a :class:`~qphase.data.bundle.GenericDataBundle` is
    returned, so core can always restore any artifact even when the owning
    resource package is not installed.
    """
    from .bundle import GenericDataBundle

    directory = Path(directory)
    manifest = ArtifactManifestV3.read(directory)
    products = _load_products(manifest, directory)
    builder = _BUNDLE_ADAPTERS.get(manifest.bundle.adapter_id)
    if builder is None:
        return GenericDataBundle(
            products,
            manifest.bundle,
            provenance=manifest.provenance,
            metadata={"artifact_id": manifest.artifact_id},
        )
    return builder(manifest, products)
