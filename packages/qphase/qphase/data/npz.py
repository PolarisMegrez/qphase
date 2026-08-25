"""qphase: NPZ 2.x Storage Adapter
---------------------------------------------------------
Reference :class:`~qphase.data.store.StorageAdapterProtocol` implementation.
Layout rules:

- one chunk file per variable chunk, named
  ``{stem}__{variable}__{chunk:04d}.npz`` with a single ``"data"`` key;
  unsharded variables use ``{stem}__{variable}.npz``;
- arrays are stored with their native dtype (complex/tensor payloads
  included) — never object arrays, so restoring never needs
  ``allow_pickle``;
- metadata lives only in the manifest JSON, never inside the NPZ;
- every chunk carries a content hash over its dtype/shape/order/selection
  header plus its C-contiguous payload bytes, verified on each read together
  with the actual dtype, shape and key set;
- reopening a product is lazy: handles expose shape/dtype/nbytes from the
  validated manifest without reading, and selections read only the chunks
  they touch (no full concatenation for point/chunk access).

The module keeps a process-local registry mapping artifact ids to artifact
directories, so that :class:`~qphase.data.artifact.ArtifactRef`-backed
datasets can resolve their storage context. The registry is populated by
``save_products``/``load_products``; cross-process restores must open the
artifact directory once before dereferencing refs.

Public API
----------
NpzStorageAdapter
    The NPZ 2.x storage adapter.
NpzArrayHandle
    Lazy single-chunk array handle.
ShardedNpzArrayHandle
    Lazy sharded array handle with chunk-pruning selection reads.
load_product_backing
    Registry-backed restore entry point for NPZ artifact refs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .artifact import ArtifactRef
from .datasets import Dataset
from .errors import (
    ArtifactAdapterError,
    ArtifactChecksumError,
    ArtifactCorruptError,
    ArtifactNotFoundError,
)
from .handles import CopyPolicy
from .product import RuntimeProductBacking
from .resolver import ArtifactResolverProtocol, default_artifact_resolver
from .runtime import DictProductBacking, _ArrayHandleBase
from .schema import VariableSchema
from .store import (
    ChunkRecord,
    ProductEntry,
    ProductStorage,
    chunk_content_hash,
    resolve_artifact_path,
)

__all__ = [
    "NpzArrayHandle",
    "NpzStorageAdapter",
    "ShardedNpzArrayHandle",
    "load_product_backing",
]

_KEY = "data"


def _read_chunk(path: Path, record: ChunkRecord) -> np.ndarray:
    """Read one chunk file and verify key set, dtype, shape and hash.

    ``allow_pickle`` is never enabled: chunk files hold native-dtype arrays
    only, so a payload that would require pickle is rejected by NumPy. The
    content hash covers the dtype/shape/order/selection header plus the
    C-order payload bytes, so reinterpreted payloads never verify.
    """
    try:
        with np.load(path) as npz:
            keys = set(npz.files)
            if keys != {record.key}:
                raise ArtifactCorruptError(
                    f"artifact chunk {path} holds keys {sorted(keys)}, "
                    f"expected exactly {record.key!r}"
                )
            array = np.asarray(npz[record.key])
    except FileNotFoundError:
        raise ArtifactNotFoundError(
            f"artifact chunk file is missing: {path}"
        ) from None
    except (ArtifactCorruptError, ArtifactNotFoundError):
        raise
    except Exception as exc:
        raise ArtifactCorruptError(
            f"failed to read artifact chunk {path}: {exc}"
        ) from exc
    if np.dtype(array.dtype) != np.dtype(record.dtype):
        raise ArtifactChecksumError(
            f"artifact chunk {path} has dtype {array.dtype.str!r}, expected "
            f"{np.dtype(record.dtype).str!r}"
        )
    if tuple(array.shape) != tuple(record.shape):
        raise ArtifactChecksumError(
            f"artifact chunk {path} has shape {tuple(array.shape)}, expected "
            f"{tuple(record.shape)}"
        )
    actual = chunk_content_hash(array, record.axis0_range)
    if actual != record.sha256:
        raise ArtifactChecksumError(
            f"checksum mismatch for artifact chunk {path}: expected "
            f"{record.sha256}, got {actual}"
        )
    return array


class NpzArrayHandle(_ArrayHandleBase):
    """Lazy handle over one single-chunk variable stored in an NPZ file."""

    def __init__(
        self,
        path: Path,
        record: ChunkRecord,
        variable_schema: VariableSchema,
        *,
        owner: str,
    ) -> None:
        super().__init__(variable_schema, owner=owner, read_only=True)
        self._path = path
        self._record = record

    @property
    def device(self) -> str:
        """NPZ payloads are host-resident."""
        return "cpu"

    @property
    def dtype(self) -> str:
        """Element dtype name of the payload."""
        return np.dtype(self._variable_schema.dtype).str

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the payload (manifest metadata, never reads the file)."""
        return tuple(self._record.shape)

    @property
    def nbytes(self) -> int:
        """Payload size in bytes (manifest metadata, never reads the file)."""
        size = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        return size * np.dtype(self._variable_schema.dtype).itemsize

    def materialize(
        self,
        target_device: str | None = None,
        copy_policy: CopyPolicy = "allow",
    ) -> np.ndarray:
        """Read the chunk from disk; host-only, never needs a copy policy."""
        self._check_live()
        if target_device not in (None, "cpu"):
            raise RuntimeError(
                f"npz handle cannot materialize on {target_device!r}; device "
                "transfers are performed by backends creating backend handles"
            )
        return _read_chunk(self._path, self._record)

    def materialize_selection(self, indexers: tuple[Any, ...]) -> np.ndarray:
        """Read the chunk and apply the selection (single-chunk fast path)."""
        return self.materialize()[indexers]


class ShardedNpzArrayHandle(_ArrayHandleBase):
    """Lazy handle over a variable split into chunks along its first dim.

    ``materialize()`` concatenates all chunks (an explicit full read);
    ``materialize_selection(indexers)`` reads only the chunks the selection
    touches for contiguous slices and point indices along the shard axis,
    and falls back to a full read for strided selections.
    """

    def __init__(
        self,
        chunks: list[tuple[Path, ChunkRecord]],
        variable_schema: VariableSchema,
        *,
        shape: tuple[int, ...],
        owner: str,
    ) -> None:
        super().__init__(variable_schema, owner=owner, read_only=True)
        if not shape or len(shape) != len(variable_schema.dims):
            raise ValueError(
                f"sharded handle for variable {variable_schema.name!r} needs "
                "a full shape matching the variable dims"
            )
        expected = 0
        ranges: list[tuple[int, int]] = []
        for _path, record in chunks:
            if record.axis0_range is None:
                raise ValueError(
                    f"sharded handle chunk {record.file!r} misses its "
                    "axis0_range"
                )
            start, stop = record.axis0_range
            if start != expected:
                raise ValueError(
                    f"sharded handle chunks must be contiguous: expected start "
                    f"{expected}, got {start} ({record.file})"
                )
            ranges.append((start, stop))
            expected = stop
        if expected != shape[0]:
            raise ValueError(
                f"sharded chunks cover [0, {expected}) but the variable shape "
                f"declares {shape[0]} rows"
            )
        self._chunks = chunks
        self._ranges = ranges
        self._shape = tuple(shape)

    @property
    def device(self) -> str:
        """NPZ payloads are host-resident."""
        return "cpu"

    @property
    def dtype(self) -> str:
        """Element dtype name of the payload."""
        return np.dtype(self._variable_schema.dtype).str

    @property
    def shape(self) -> tuple[int, ...]:
        """Full shape of the payload (never reads chunk files)."""
        return self._shape

    @property
    def nbytes(self) -> int:
        """Payload size in bytes (never reads chunk files)."""
        size = int(np.prod(self._shape, dtype=np.int64))
        return size * np.dtype(self._variable_schema.dtype).itemsize

    @property
    def chunk_count(self) -> int:
        """Number of persisted chunks."""
        return len(self._chunks)

    def read_chunk(self, index: int) -> np.ndarray:
        """Read one chunk file (with full content verification)."""
        self._check_live()
        path, record = self._chunks[index]
        return _read_chunk(path, record)

    def materialize(
        self,
        target_device: str | None = None,
        copy_policy: CopyPolicy = "allow",
    ) -> np.ndarray:
        """Read and concatenate all chunks along the shard axis."""
        self._check_live()
        if target_device not in (None, "cpu"):
            raise RuntimeError(
                f"npz handle cannot materialize on {target_device!r}; device "
                "transfers are performed by backends creating backend handles"
            )
        parts = [self.read_chunk(i) for i in range(len(self._chunks))]
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    def materialize_selection(self, indexers: tuple[Any, ...]) -> np.ndarray:
        """Read only the chunks the selection touches.

        Fast path: point indices and contiguous (step-1) slices along the
        shard axis prune unread chunks. Strided selections fall back to an
        explicit full read.
        """
        self._check_live()
        if not indexers:
            return self.materialize()
        selector = indexers[0]
        rest = indexers[1:]
        if isinstance(selector, (int, np.integer)) and not isinstance(
            selector, bool
        ):
            row = int(selector)
            if row < 0:
                row += self._shape[0]
            for (path, record), (start, stop) in zip(
                self._chunks, self._ranges, strict=True
            ):
                if start <= row < stop:
                    return _read_chunk(path, record)[row - start, *rest]
            raise IndexError(
                f"index {selector} out of bounds for axis of size "
                f"{self._shape[0]}"
            )
        if isinstance(selector, slice):
            start, stop, step = selector.indices(self._shape[0])
            if step == 1:
                if start >= stop:
                    empty = np.empty(
                        (0, *self._shape[1:]), dtype=self._variable_schema.dtype
                    )
                    return empty[(slice(None), *rest)]
                base = 0
                parts: list[np.ndarray] = []
                for (path, record), (c0, c1) in zip(
                    self._chunks, self._ranges, strict=True
                ):
                    if c0 < stop and c1 > start:
                        if not parts:
                            base = c0
                        parts.append(_read_chunk(path, record))
                data = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
                return data[slice(start - base, stop - base), *rest]
        return self.materialize()[indexers]


class NpzStorageAdapter:
    """NPZ 2.x storage adapter: native dtypes, per-chunk content hashes."""

    ADAPTER_ID: ClassVar[str] = "npz/2"

    @property
    def adapter_id(self) -> str:
        """Stable adapter identifier recorded in manifests."""
        return self.ADAPTER_ID

    def write_product(
        self,
        name: str,
        dataset: Dataset,
        directory: Path,
        *,
        shard_target_bytes: int,
        file_stem: str,
    ) -> ProductStorage:
        """Persist one runtime-backed dataset as per-variable chunk files."""
        directory = Path(directory)
        records: dict[str, list[ChunkRecord]] = {}
        for variable in dataset.schema.variables:
            handle = dataset.handle(variable.name)
            # Persistence is a host operation: the copy policy is explicit.
            # np.ascontiguousarray promotes 0-d arrays to 1-d, so keep
            # scalars as-is (0-d arrays are trivially C-contiguous).
            array = handle.materialize("cpu", copy_policy="allow")
            if array.ndim:
                array = np.ascontiguousarray(array)
            else:
                array = np.asarray(array)
            records[variable.name] = self._write_variable(
                file_stem, variable.name, array, directory, shard_target_bytes
            )
        return ProductStorage(adapter=self.ADAPTER_ID, variables=records)

    def open_product(
        self, entry: ProductEntry, directory: Path
    ) -> RuntimeProductBacking:
        """Open a stored product as lazily-reading runtime handles.

        Every chunk path is re-resolved under the artifact root and checked
        for existence; payloads themselves are only read on demand.
        """
        directory = Path(directory)
        handles: dict[str, Any] = {}
        for variable in entry.product_schema.variables:
            try:
                chunks = entry.storage.variables[variable.name]
            except KeyError:
                raise ArtifactCorruptError(
                    f"storage of product {entry.name!r} misses variable "
                    f"{variable.name!r}"
                ) from None
            owner = f"npz:{entry.name}"
            paths = [resolve_artifact_path(directory, chunk.file) for chunk in chunks]
            for path in paths:
                if not path.is_file():
                    raise ArtifactNotFoundError(
                        f"artifact chunk file is missing: {path}"
                    )
            if len(chunks) == 1:
                handles[variable.name] = NpzArrayHandle(
                    paths[0], chunks[0], variable, owner=owner
                )
            else:
                full_shape = (
                    sum(chunk.shape[0] for chunk in chunks),
                    *chunks[0].shape[1:],
                )
                handles[variable.name] = ShardedNpzArrayHandle(
                    list(zip(paths, chunks, strict=True)),
                    variable,
                    shape=full_shape,
                    owner=owner,
                )
        return DictProductBacking(handles)

    def open_ref(
        self,
        ref: ArtifactRef,
        *,
        resolver: ArtifactResolverProtocol | None = None,
    ) -> RuntimeProductBacking:
        """Open the product referenced by an NPZ artifact ref."""
        if ref.storage_adapter != self.ADAPTER_ID:
            raise ArtifactAdapterError(
                f"artifact ref requires adapter {ref.storage_adapter!r}, "
                f"not {self.ADAPTER_ID!r}"
            )
        active = resolver if resolver is not None else default_artifact_resolver()
        directory = active.resolve(ref)
        from .store import ArtifactManifestV3  # local: avoid an import cycle

        manifest = ArtifactManifestV3.read(directory)
        try:
            entry = manifest.product_entry(ref.product_name)
        except KeyError:
            raise ArtifactNotFoundError(
                f"artifact {ref.artifact_id!r} has no product {ref.product_name!r}"
            ) from None
        if entry.product_schema != ref.product_schema:
            raise ArtifactChecksumError(
                f"artifact {ref.artifact_id!r} product schema does not match "
                "the reference"
            )
        if entry.sha256 != ref.content_hash:
            raise ArtifactChecksumError(
                f"artifact {ref.artifact_id!r} content hash does not match "
                "the reference"
            )
        return self.open_product(entry, directory)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _write_variable(
        stem: str,
        variable_name: str,
        array: np.ndarray,
        directory: Path,
        shard_target_bytes: int,
    ) -> list[ChunkRecord]:
        safe_variable = _sanitize_stem(variable_name)
        rows = array.shape[0] if array.ndim >= 1 else 0
        if array.ndim >= 1 and rows > 1 and array.nbytes > shard_target_bytes > 0:
            chunk_count = math.ceil(array.nbytes / shard_target_bytes)
            rows_per_chunk = math.ceil(rows / chunk_count)
            records = []
            for index, start in enumerate(range(0, rows, rows_per_chunk)):
                stop = min(start + rows_per_chunk, rows)
                chunk = np.ascontiguousarray(array[start:stop])
                filename = f"{stem}__{safe_variable}__{index:04d}.npz"
                np.savez(directory / filename, data=chunk)
                records.append(
                    ChunkRecord(
                        file=filename,
                        key=_KEY,
                        shape=tuple(chunk.shape),
                        dtype=np.dtype(chunk.dtype).str,
                        sha256=chunk_content_hash(chunk, (start, stop)),
                        axis0_range=(start, stop),
                    )
                )
            return records
        filename = f"{stem}__{safe_variable}.npz"
        np.savez(directory / filename, data=array)
        return [
            ChunkRecord(
                file=filename,
                key=_KEY,
                shape=tuple(array.shape),
                dtype=np.dtype(array.dtype).str,
                sha256=chunk_content_hash(array),
            )
        ]


def _sanitize_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)


def load_product_backing(ref: ArtifactRef) -> RuntimeProductBacking:
    """Restore entry point for NPZ-backed artifact refs.

    Resolves the artifact's on-disk location through the process-local
    registry populated by ``save_products``/``load_products`` and opens the
    product as a lazily-reading runtime backing. Datasets reach this through
    the storage adapter registry, never through a persisted code path.
    """
    return NpzStorageAdapter().open_ref(ref)
