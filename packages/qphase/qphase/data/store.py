"""qphase: Artifact Manifest v3 and Storage Adapter Contract
---------------------------------------------------------
The v3 artifact manifest is the single public restore entry point for
persisted data products. It records, per artifact:

- artifact schema/version/id/content hash and creation time;
- product names with their full :class:`ProductSchema`;
- the storage adapter id and the per-variable chunk/shard mapping;
- JSON provenance (plugin/config/backend fingerprints, conventions);
- parent artifact ids and the public loader entry point
  (``module:attr`` syntax, see :class:`~qphase.data.artifact.ArtifactRef`).

Storage adapters implement :class:`StorageAdapterProtocol`; the NPZ 2.x
adapter (:mod:`qphase.data.npz`) is the reference implementation. Other
adapters (e.g. Zarr) can register through :func:`register_adapter` without
changing the manifest format.

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
ArtifactManifestV3
    The v3 artifact manifest.
StorageAdapterProtocol
    Persistence adapter contract.
save_products
    Persist typed datasets and write the v3 manifest.
load_products
    Reopen an artifact directory as lazily-backed datasets.
register_adapter
    Register a storage adapter implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.utils import canonical_json
from .artifact import ArtifactRef
from .datasets import Dataset, SpectralDataset, StatisticsDataset, TimeSeriesDataset
from .kinds import DataKind
from .product import RuntimeProductBacking
from .schema import ProductSchema

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "ArtifactManifestV3",
    "ChunkRecord",
    "ProductEntry",
    "ProductStorage",
    "StorageAdapterProtocol",
    "load_products",
    "register_adapter",
    "save_products",
]

#: Manifest schema identifier frozen for qphase 2.0 Phase 1.
ARTIFACT_SCHEMA_VERSION: Literal["qphase.artifact/3"] = "qphase.artifact/3"

#: Manifest file name inside every artifact directory.
MANIFEST_FILENAME = "artifact_manifest.json"

#: Default public restore entry point (NPZ 2.x adapter).
DEFAULT_LOADER = "qphase.data.npz:load_product_backing"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

#: Default shard target: 64 MiB of payload per chunk.
DEFAULT_SHARD_TARGET_BYTES = 64 * (1 << 20)


def _validate_sha256_digest(value: str) -> str:
    if not _SHA256_PATTERN.match(value):
        raise ValueError("expected a 64-character lowercase SHA-256 digest")
    return value


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
        description="Content hash over the product's chunk hashes."
    )

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _validate_sha256_digest(value)


class ArtifactManifestV3(BaseModel):
    """The v3 artifact manifest: the public restore entry of an artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["qphase.artifact/3"] = ARTIFACT_SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    created_at: str
    loader: str = Field(
        default=DEFAULT_LOADER,
        description="Public restore entry point in 'module:attr' syntax.",
    )
    products: list[ProductEntry] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    parents: list[str] = Field(default_factory=list)
    content_hash: str = Field(
        description="SHA-256 over the canonical product/parent listing."
    )

    @field_validator("content_hash")
    @classmethod
    def _check_content_hash(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("provenance")
    @classmethod
    def _check_provenance_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "manifest provenance must be JSON-serializable"
            ) from exc
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
            artifact_id=f"{self.artifact_id}:{name}",
            product_schema=entry.product_schema,
            loader=self.loader,
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
        """Read and validate the v3 manifest of an artifact directory."""
        path = Path(directory) / MANIFEST_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"no artifact manifest at {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("schema_version")
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported artifact schema version {version!r}; expected "
                f"{ARTIFACT_SCHEMA_VERSION!r}"
            )
        return cls.model_validate(raw)


@runtime_checkable
class StorageAdapterProtocol(Protocol):
    """Persistence adapter contract for typed data products.

    Adapters write a runtime-backed dataset as chunk files plus a storage
    record, and reopen a storage record as a lazily-reading runtime backing.
    The NPZ 2.x adapter is the reference implementation; further adapters
    (e.g. Zarr) plug in without manifest changes.
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


_ADAPTERS: dict[str, Any] = {}


def register_adapter(adapter: StorageAdapterProtocol) -> None:
    """Register a storage adapter under its adapter id."""
    _ADAPTERS[adapter.adapter_id] = adapter


def _resolve_adapter(adapter_id: str) -> StorageAdapterProtocol:
    if adapter_id not in _ADAPTERS:
        # Ensure the built-in NPZ adapter is registered.
        from .npz import NpzStorageAdapter

        register_adapter(NpzStorageAdapter())
    try:
        return _ADAPTERS[adapter_id]
    except KeyError:
        raise ValueError(
            f"unknown storage adapter {adapter_id!r}; registered: "
            f"{sorted(_ADAPTERS)}"
        ) from None


_DATASET_BY_KIND: dict[DataKind, type[Dataset]] = {
    DataKind.TIME_SERIES: TimeSeriesDataset,
    DataKind.SPECTRAL: SpectralDataset,
    DataKind.STATISTICS: StatisticsDataset,
}


def _sanitize(name: str) -> str:
    return _SAFE_STEM_PATTERN.sub("_", name) or "product"


def _content_hash(
    products: Sequence[ProductEntry], parents: Sequence[str]
) -> str:
    listing = {
        "parents": sorted(parents),
        "products": [
            {
                "name": entry.name,
                "schema": entry.product_schema.fingerprint(),
                "sha256": entry.sha256,
            }
            for entry in sorted(products, key=lambda entry: entry.name)
        ],
    }
    return hashlib.sha256(canonical_json(listing).encode("utf-8")).hexdigest()


def save_products(
    directory: Path | str,
    products: Mapping[str, Dataset],
    *,
    provenance: Mapping[str, Any] | None = None,
    parents: Sequence[str] = (),
    artifact_id: str | None = None,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
) -> ArtifactManifestV3:
    """Persist typed datasets and write the v3 artifact manifest.

    Artifact-backed datasets are fully materialized first (an explicit load,
    never an implicit one); device-resident payloads are copied to the host
    with an explicit ``copy_policy="allow"`` at save time. Every variable is
    stored as one or more chunks split along its first dimension. An empty
    mapping is allowed and produces a manifest with no products (e.g. a
    simulation that only yields scalar metadata).
    """
    if not isinstance(products, Mapping):
        raise TypeError(
            f"products must be a mapping of name to Dataset, got "
            f"{type(products).__name__}"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    from .npz import (  # local import: npz depends on this module's models
        NpzStorageAdapter,
        register_product_location,
    )

    adapter = _resolve_adapter(NpzStorageAdapter.ADAPTER_ID)
    entries: list[ProductEntry] = []
    for index, (name, dataset) in enumerate(products.items()):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"product {name!r} must be a Dataset, got "
                f"{type(dataset).__name__}"
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
        product_hash = hashlib.sha256(
            "\n".join(
                chunk.sha256
                for chunks in storage.variables.values()
                for chunk in chunks
            ).encode("utf-8")
        ).hexdigest()
        entries.append(
            ProductEntry(
                name=name,
                product_schema=dataset.schema,
                storage=storage,
                sha256=product_hash,
            )
        )

    if artifact_id is None:
        artifact_id = hashlib.sha256(
            f"{datetime.now(UTC).isoformat()}-"
            f"{sorted(products)}".encode()
        ).hexdigest()[:16]
    manifest = ArtifactManifestV3(
        artifact_id=artifact_id,
        created_at=datetime.now(UTC).isoformat(),
        products=entries,
        provenance=dict(provenance or {}),
        parents=[str(parent) for parent in parents],
        content_hash=_content_hash(entries, parents),
    )
    manifest.write(directory)
    for entry in entries:
        register_product_location(manifest.artifact_id, entry.name, directory)
    return manifest


def load_products(directory: Path | str) -> dict[str, Dataset]:
    """Reopen an artifact directory as lazily-backed datasets.

    Payloads are not read: the returned datasets are backed by lazy handles
    that load chunks on demand (and only the chunks a selection touches).
    """
    from .npz import register_product_location

    directory = Path(directory)
    manifest = ArtifactManifestV3.read(directory)
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
        register_product_location(manifest.artifact_id, entry.name, directory)
    return result
