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
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .artifact import ArtifactRef
from .datasets import Dataset
from .errors import (
    ArtifactAdapterError,
    ArtifactChecksumError,
    ArtifactCorruptError,
    ArtifactNotFoundError,
    ArtifactUnsupportedError,
)
from .handles import CopyPolicy
from .product import RuntimeProductBacking
from .resolver import ArtifactResolverProtocol, default_artifact_resolver
from .runtime import DictProductBacking, _ArrayHandleBase
from .schema import ProductSchema, VariableSchema
from .store import (
    ProductEntry,
    ProductStorage,
    StorageVariableSummary,
    chunk_content_hash,
    resolve_artifact_path,
    validate_artifact_relative_path,
)

__all__ = [
    "NPZ_DESCRIPTOR_SCHEMA",
    "NpzArrayHandle",
    "NpzChunkRecord",
    "NpzProductDescriptor",
    "NpzStorageAdapter",
    "NpzVariableDescriptor",
    "ShardedNpzArrayHandle",
    "build_product_storage",
    "load_product_backing",
]

_KEY = "data"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: Descriptor schema id of the NPZ 2.x per-product storage descriptor.
NPZ_DESCRIPTOR_SCHEMA = "npz.product/2"


class NpzChunkRecord(BaseModel):
    """One persisted array chunk of a variable (NPZ descriptor entry).

    ``logical_range`` records the [start, stop) window the chunk covers
    along the variable's ``chunk_axis`` (None when the chunk holds the whole
    variable). ``file`` is relative to the artifact directory so artifacts
    can be relocated.
    """

    model_config = ConfigDict(extra="forbid")

    file: str
    key: str
    logical_range: tuple[int, int] | None = None
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    @field_validator("file")
    @classmethod
    def _check_file(cls, value: str) -> str:
        return validate_artifact_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected a 64-character lowercase SHA-256 digest")
        return value


class NpzVariableDescriptor(BaseModel):
    """NPZ storage layout of one variable.

    ``chunk_axis`` names the variable dim along which chunks split the
    payload (None for unsharded variables); ``chunks`` hold the contiguous,
    non-overlapping, fully-covering per-chunk file records.
    """

    model_config = ConfigDict(extra="forbid")

    full_shape: tuple[int, ...]
    dtype: str
    chunk_axis: str | None = None
    chunks: list[NpzChunkRecord] = Field(min_length=1)


class NpzProductDescriptor(BaseModel):
    """NPZ 2.x per-product storage descriptor (``npz.product/2``)."""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, NpzVariableDescriptor]


def _read_chunk(path: Path, record: NpzChunkRecord) -> np.ndarray:
    """Read one chunk file and verify key set, dtype, shape and hash.

    ``allow_pickle`` is never enabled: chunk files hold native-dtype arrays
    only, so a payload that would require pickle is rejected by NumPy. The
    content hash covers the dtype/shape/order/selection header plus the
    C-order payload bytes, so reinterpreted payloads never verify.
    """
    try:
        with np.load(path) as npz:
            keys = set(npz.files)
            if record.key not in keys:
                raise ArtifactCorruptError(
                    f"artifact chunk {path} holds keys {sorted(keys)}, "
                    f"missing the declared key {record.key!r}"
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
    actual = chunk_content_hash(array, record.logical_range)
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
        record: NpzChunkRecord,
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
    """Lazy handle over a variable split into chunks along one named dim.

    ``materialize()`` concatenates all chunks (an explicit full read);
    ``materialize_selection(indexers)`` reads only the chunks the selection
    touches for contiguous slices and point indices along the shard axis,
    and falls back to a full read for strided selections.
    """

    def __init__(
        self,
        chunks: list[tuple[Path, NpzChunkRecord]],
        variable_schema: VariableSchema,
        *,
        shape: tuple[int, ...],
        axis: int = 0,
        owner: str,
    ) -> None:
        super().__init__(variable_schema, owner=owner, read_only=True)
        if not shape or len(shape) != len(variable_schema.dims):
            raise ValueError(
                f"sharded handle for variable {variable_schema.name!r} needs "
                "a full shape matching the variable dims"
            )
        if not 0 <= axis < len(shape):
            raise ValueError(
                f"sharded handle for variable {variable_schema.name!r} got "
                f"invalid shard axis {axis}"
            )
        expected = 0
        ranges: list[tuple[int, int]] = []
        for _path, record in chunks:
            if record.logical_range is None:
                raise ValueError(
                    f"sharded handle chunk {record.file!r} misses its "
                    "logical_range"
                )
            start, stop = record.logical_range
            if start != expected:
                raise ValueError(
                    f"sharded handle chunks must be contiguous: expected start "
                    f"{expected}, got {start} ({record.file})"
                )
            ranges.append((start, stop))
            expected = stop
        if expected != shape[axis]:
            raise ValueError(
                f"sharded chunks cover [0, {expected}) but the variable shape "
                f"declares {shape[axis]} rows along axis {axis}"
            )
        self._chunks = chunks
        self._ranges = ranges
        self._shape = tuple(shape)
        self._axis = axis

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
        return np.concatenate(parts, axis=self._axis)

    def materialize_selection(self, indexers: tuple[Any, ...]) -> np.ndarray:
        """Read only the chunks the selection touches.

        Fast path: point indices and contiguous (step-1) slices along the
        shard axis prune unread chunks. Strided selections fall back to an
        explicit full read.
        """
        self._check_live()
        if not indexers:
            return self.materialize()
        axis = self._axis
        axis_size = self._shape[axis]
        selector = indexers[axis]
        prefix = indexers[:axis]
        suffix = indexers[axis + 1 :]
        if isinstance(selector, (int, np.integer)) and not isinstance(
            selector, bool
        ):
            row = int(selector)
            if row < 0:
                row += axis_size
            for (path, record), (start, stop) in zip(
                self._chunks, self._ranges, strict=True
            ):
                if start <= row < stop:
                    return _read_chunk(path, record)[
                        (*prefix, row - start, *suffix)
                    ]
            raise IndexError(
                f"index {selector} out of bounds for axis of size {axis_size}"
            )
        if isinstance(selector, slice):
            start, stop, step = selector.indices(axis_size)
            if step == 1:
                reduced = sum(
                    1
                    for item in prefix
                    if isinstance(item, (int, np.integer))
                    and not isinstance(item, bool)
                )
                concat_axis = axis - reduced
                if start >= stop:
                    full_shape = list(self._shape)
                    full_shape[axis] = 0
                    empty = np.empty(
                        tuple(full_shape), dtype=self._variable_schema.dtype
                    )
                    return empty[(*prefix, slice(None), *suffix)]
                parts: list[np.ndarray] = []
                for (path, record), (c0, c1) in zip(
                    self._chunks, self._ranges, strict=True
                ):
                    if c0 < stop and c1 > start:
                        local = slice(max(start, c0) - c0, min(stop, c1) - c0)
                        parts.append(
                            _read_chunk(path, record)[(*prefix, local, *suffix)]
                        )
                if len(parts) == 1:
                    return parts[0]
                return np.concatenate(parts, axis=concat_axis)
        return self.materialize()[indexers]


class NpzStorageAdapter:
    """NPZ 2.x storage adapter: native dtypes, per-chunk content hashes."""

    ADAPTER_ID: ClassVar[str] = "npz/2"
    DESCRIPTOR_SCHEMA: ClassVar[str] = NPZ_DESCRIPTOR_SCHEMA

    @property
    def adapter_id(self) -> str:
        """Stable adapter identifier recorded in manifests."""
        return self.ADAPTER_ID

    @property
    def descriptor_schema(self) -> str:
        """Schema id of the NPZ storage descriptor."""
        return self.DESCRIPTOR_SCHEMA

    def write_product(
        self,
        name: str,
        dataset: Dataset,
        directory: Path,
        *,
        shard_target_bytes: int,
        file_stem: str,
        layout: str = "sharded",
    ) -> ProductStorage:
        """Persist one runtime-backed dataset as per-variable chunk files.

        ``layout="single"`` writes one payload file holding every variable
        under its own key (no external sharding); ``layout="sharded"``
        splits variables along a planned named axis into byte-targeted
        chunk files.
        """
        directory = Path(directory)
        arrays: dict[str, np.ndarray] = {}
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
            arrays[variable.name] = array
        if layout == "single":
            variables = self._write_single(file_stem, dataset.schema, arrays, directory)
        else:
            variables = {
                variable.name: self._write_variable(
                    file_stem,
                    variable,
                    arrays[variable.name],
                    directory,
                    shard_target_bytes,
                )
                for variable in dataset.schema.variables
            }
        return build_product_storage(dataset.schema, variables)

    @staticmethod
    def _write_single(
        stem: str,
        schema: ProductSchema,
        arrays: dict[str, np.ndarray],
        directory: Path,
    ) -> dict[str, NpzVariableDescriptor]:
        """Write all variables as keys of one payload file (``single``)."""
        filename = f"{stem}.npz"
        np.savez(directory / filename, **dict(arrays))
        variables: dict[str, NpzVariableDescriptor] = {}
        for variable in schema.variables:
            array = arrays[variable.name]
            variables[variable.name] = NpzVariableDescriptor(
                full_shape=tuple(array.shape),
                dtype=np.dtype(array.dtype).str,
                chunks=[
                    NpzChunkRecord(
                        file=filename,
                        key=variable.name,
                        shape=tuple(array.shape),
                        dtype=np.dtype(array.dtype).str,
                        sha256=chunk_content_hash(array),
                    )
                ],
            )
        return variables

    def parse_storage(self, entry: ProductEntry) -> NpzProductDescriptor:
        """Strictly parse and validate the NPZ descriptor of one entry."""
        storage = entry.storage
        if storage.adapter != self.ADAPTER_ID:
            raise ArtifactAdapterError(
                f"product {entry.name!r} requires adapter {storage.adapter!r}, "
                f"not {self.ADAPTER_ID!r}"
            )
        if storage.descriptor_schema != self.DESCRIPTOR_SCHEMA:
            raise ArtifactUnsupportedError(
                f"product {entry.name!r} uses NPZ descriptor schema "
                f"{storage.descriptor_schema!r}; this adapter supports "
                f"{self.DESCRIPTOR_SCHEMA!r}"
            )
        try:
            descriptor = NpzProductDescriptor.model_validate(storage.descriptor)
        except ValidationError as exc:
            raise ArtifactCorruptError(
                f"NPZ storage descriptor of product {entry.name!r} is "
                f"invalid: {exc}"
            ) from exc
        self._validate_descriptor(entry, descriptor)
        return descriptor

    def referenced_files(self, entry: ProductEntry) -> dict[str, str]:
        """Report chunk files referenced by one entry (file -> owner)."""
        storage = entry.storage
        if storage.adapter != self.ADAPTER_ID:
            raise ArtifactAdapterError(
                f"product {entry.name!r} requires adapter {storage.adapter!r}, "
                f"not {self.ADAPTER_ID!r}"
            )
        if storage.descriptor_schema != self.DESCRIPTOR_SCHEMA:
            raise ArtifactUnsupportedError(
                f"product {entry.name!r} uses NPZ descriptor schema "
                f"{storage.descriptor_schema!r}; this adapter supports "
                f"{self.DESCRIPTOR_SCHEMA!r}"
            )
        try:
            descriptor = NpzProductDescriptor.model_validate(storage.descriptor)
        except ValidationError as exc:
            raise ArtifactCorruptError(
                f"NPZ storage descriptor of product {entry.name!r} is "
                f"invalid: {exc}"
            ) from exc
        return {
            chunk.file: f"{entry.name}.{name}"
            for name, variable in descriptor.variables.items()
            for chunk in variable.chunks
        }

    def open_product(
        self, entry: ProductEntry, directory: Path
    ) -> RuntimeProductBacking:
        """Open a stored product as lazily-reading runtime handles.

        The descriptor is strictly parsed and cross-validated first; every
        chunk path is re-resolved under the artifact root and checked for
        existence. Payloads themselves are only read on demand.
        """
        directory = Path(directory)
        descriptor = self.parse_storage(entry)
        handles: dict[str, Any] = {}
        for variable in entry.product_schema.variables:
            variable_descriptor = descriptor.variables[variable.name]
            owner = f"npz:{entry.name}"
            paths = [
                resolve_artifact_path(directory, chunk.file)
                for chunk in variable_descriptor.chunks
            ]
            for path in paths:
                if not path.is_file():
                    raise ArtifactNotFoundError(
                        f"artifact chunk file is missing: {path}"
                    )
            if len(variable_descriptor.chunks) == 1:
                handles[variable.name] = NpzArrayHandle(
                    paths[0], variable_descriptor.chunks[0], variable, owner=owner
                )
                continue
            axis_index = variable.dims.index(variable_descriptor.chunk_axis)
            handles[variable.name] = ShardedNpzArrayHandle(
                list(zip(paths, variable_descriptor.chunks, strict=True)),
                variable,
                shape=tuple(variable_descriptor.full_shape),
                axis=axis_index,
                owner=owner,
            )
        return DictProductBacking(handles)

    def verify_product(self, entry: ProductEntry, directory: Path) -> None:
        """Re-read every chunk of a freshly written product (with hashing).

        Called by the transactional writer before chunks are published:
        each chunk file is reopened, and its key set, dtype, shape and
        content hash are verified against the descriptor.
        """
        directory = Path(directory)
        descriptor = self.parse_storage(entry)
        for variable_descriptor in descriptor.variables.values():
            for chunk in variable_descriptor.chunks:
                _read_chunk(resolve_artifact_path(directory, chunk.file), chunk)

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
    def _plan_chunk_axis(
        array: np.ndarray, shard_target_bytes: int
    ) -> tuple[int, int] | None:
        """Choose (axis index, rows per chunk) for sharding, or None.

        The first axis whose per-row payload fits the target wins, so scan
        axes (kept whole per point for point-view pruning) are preferred
        over trajectory axes, which are preferred over time; scalar and
        small variables stay unsharded.
        """
        if (
            array.ndim == 0
            or shard_target_bytes <= 0
            or array.nbytes <= shard_target_bytes
        ):
            return None
        for axis, size in enumerate(array.shape):
            if size > 1 and array.nbytes // size <= shard_target_bytes:
                chunk_count = math.ceil(array.nbytes / shard_target_bytes)
                return axis, max(1, math.ceil(size / chunk_count))
        for axis, size in enumerate(array.shape):
            if size > 1:
                return axis, 1
        return None

    @staticmethod
    def _write_variable(
        stem: str,
        variable: VariableSchema,
        array: np.ndarray,
        directory: Path,
        shard_target_bytes: int,
    ) -> NpzVariableDescriptor:
        safe_variable = _sanitize_stem(variable.name)
        plan = NpzStorageAdapter._plan_chunk_axis(array, shard_target_bytes)
        if plan is not None:
            axis, rows_per_chunk = plan
            size = array.shape[axis]
            records = []
            for index, start in enumerate(range(0, size, rows_per_chunk)):
                stop = min(start + rows_per_chunk, size)
                selector = (slice(None),) * axis + (slice(start, stop),)
                chunk = np.ascontiguousarray(array[selector])
                filename = f"{stem}__{safe_variable}__{index:04d}.npz"
                np.savez(directory / filename, data=chunk)
                records.append(
                    NpzChunkRecord(
                        file=filename,
                        key=_KEY,
                        logical_range=(start, stop),
                        shape=tuple(chunk.shape),
                        dtype=np.dtype(chunk.dtype).str,
                        sha256=chunk_content_hash(chunk, (start, stop)),
                    )
                )
            return NpzVariableDescriptor(
                full_shape=tuple(array.shape),
                dtype=np.dtype(array.dtype).str,
                chunk_axis=variable.dims[axis],
                chunks=records,
            )
        filename = f"{stem}__{safe_variable}.npz"
        np.savez(directory / filename, data=array)
        return NpzVariableDescriptor(
            full_shape=tuple(array.shape),
            dtype=np.dtype(array.dtype).str,
            chunks=[
                NpzChunkRecord(
                    file=filename,
                    key=_KEY,
                    shape=tuple(array.shape),
                    dtype=np.dtype(array.dtype).str,
                    sha256=chunk_content_hash(array),
                )
            ],
        )

    @staticmethod
    def _validate_descriptor(
        entry: ProductEntry, descriptor: NpzProductDescriptor
    ) -> None:
        """Cross-validate a parsed descriptor against the product schema."""
        schema_vars = {variable.name for variable in entry.product_schema.variables}
        missing = sorted(schema_vars - set(descriptor.variables))
        extra = sorted(set(descriptor.variables) - schema_vars)
        if missing or extra:
            raise ArtifactCorruptError(
                f"NPZ descriptor of product {entry.name!r} does not match "
                f"the product schema (missing: {missing}, extra: {extra})"
            )
        files: dict[tuple[str, str], str] = {}
        for name, variable_descriptor in descriptor.variables.items():
            for chunk in variable_descriptor.chunks:
                # One payload file may hold several variables under distinct
                # keys (``single`` layout); a (file, key) pair is unique.
                location = (chunk.file, chunk.key)
                previous = files.setdefault(location, name)
                if previous != name:
                    raise ArtifactCorruptError(
                        f"chunk {chunk.file!r} key {chunk.key!r} is referenced "
                        f"by both {entry.name}.{previous} and {entry.name}.{name}"
                    )
        for variable in entry.product_schema.variables:
            _validate_variable_descriptor(
                entry.name, variable, entry.product_schema,
                descriptor.variables[variable.name],
            )


def _validate_variable_descriptor(
    product: str,
    variable: VariableSchema,
    schema: ProductSchema,
    descriptor: NpzVariableDescriptor,
) -> None:
    """Validate one variable's chunk mapping against its schema."""
    label = f"variable {variable.name!r} of product {product!r}"
    expected_dtype = np.dtype(variable.dtype)
    if np.dtype(descriptor.dtype) != expected_dtype:
        raise ArtifactCorruptError(
            f"{label} descriptor has dtype {descriptor.dtype!r}, expected "
            f"{expected_dtype.str!r}"
        )
    if len(descriptor.full_shape) != len(variable.dims):
        raise ArtifactCorruptError(
            f"{label} descriptor has rank {len(descriptor.full_shape)}, "
            f"expected {len(variable.dims)}"
        )
    for dim, size in zip(variable.dims, descriptor.full_shape, strict=True):
        axis = schema.axis(dim)
        if axis.size is not None and axis.size != size:
            raise ArtifactCorruptError(
                f"{label} full shape {descriptor.full_shape} does not match "
                f"the closed axis {axis.name!r} of size {axis.size}"
            )
    for chunk in descriptor.chunks:
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
    if len(descriptor.chunks) == 1:
        chunk = descriptor.chunks[0]
        if descriptor.chunk_axis is not None:
            raise ArtifactCorruptError(
                f"{label} is unsharded but declares a chunk_axis"
            )
        if chunk.logical_range is not None:
            raise ArtifactCorruptError(
                f"{label} is unsharded but declares a logical_range"
            )
        if tuple(chunk.shape) != tuple(descriptor.full_shape):
            raise ArtifactCorruptError(
                f"{label} chunk shape {chunk.shape} does not match the "
                f"declared full shape {descriptor.full_shape}"
            )
        return
    if not variable.dims:
        raise ArtifactCorruptError(f"{label} is scalar but has multiple chunks")
    if descriptor.chunk_axis is None:
        raise ArtifactCorruptError(
            f"{label} is sharded but declares no chunk_axis"
        )
    if descriptor.chunk_axis not in variable.dims:
        raise ArtifactCorruptError(
            f"{label} declares unknown chunk axis {descriptor.chunk_axis!r}"
        )
    axis_index = variable.dims.index(descriptor.chunk_axis)
    expected_start = 0
    for chunk in descriptor.chunks:
        if chunk.logical_range is None:
            raise ArtifactCorruptError(
                f"{label} chunk {chunk.file!r} misses its logical_range"
            )
        start, stop = chunk.logical_range
        if not 0 <= start < stop:
            raise ArtifactCorruptError(
                f"{label} chunk {chunk.file!r} has out-of-bounds range "
                f"[{start}, {stop})"
            )
        for index, size in enumerate(chunk.shape):
            if index == axis_index:
                if size != stop - start:
                    raise ArtifactCorruptError(
                        f"{label} chunk {chunk.file!r} shape {chunk.shape} "
                        f"does not match its range [{start}, {stop})"
                    )
            elif size != descriptor.full_shape[index]:
                raise ArtifactCorruptError(
                    f"{label} chunk {chunk.file!r} shape {chunk.shape} does "
                    f"not match the full shape {descriptor.full_shape} off "
                    "the chunk axis"
                )
        if start != expected_start:
            raise ArtifactCorruptError(
                f"{label} chunks overlap, gap or are out of order at "
                f"[{start}, {stop}); expected start {expected_start}"
            )
        expected_start = stop
    if expected_start != descriptor.full_shape[axis_index]:
        raise ArtifactCorruptError(
            f"{label} chunks cover [0, {expected_start}) along "
            f"{descriptor.chunk_axis!r} but the full shape declares "
            f"{descriptor.full_shape[axis_index]}"
        )


def build_product_storage(
    schema: ProductSchema, variables: dict[str, NpzVariableDescriptor]
) -> ProductStorage:
    """Assemble the ProductStorage (summary + descriptor) of written chunks.

    ``variables`` must cover exactly the schema variables; used both by the
    streaming writer and by migration tooling that writes chunk files
    directly.
    """
    schema_names = {variable.name for variable in schema.variables}
    if set(variables) != schema_names:
        raise ValueError(
            f"chunk variables {sorted(variables)} do not match the product "
            f"schema variables {sorted(schema_names)}"
        )
    summary: dict[str, StorageVariableSummary] = {}
    for variable in schema.variables:
        variable_descriptor = variables[variable.name]
        itemsize = np.dtype(variable.dtype).itemsize
        nbytes = (
            int(np.prod(variable_descriptor.full_shape)) * itemsize
            if variable_descriptor.full_shape
            else itemsize
        )
        summary[variable.name] = StorageVariableSummary(
            full_shape=tuple(variable_descriptor.full_shape),
            dtype=np.dtype(variable_descriptor.dtype).str,
            nbytes=nbytes,
            chunk_count=len(variable_descriptor.chunks),
        )
    descriptor = NpzProductDescriptor(variables=dict(variables))
    return ProductStorage(
        adapter=NpzStorageAdapter.ADAPTER_ID,
        descriptor_schema=NPZ_DESCRIPTOR_SCHEMA,
        summary=summary,
        descriptor=descriptor.model_dump(mode="json"),
    )


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
