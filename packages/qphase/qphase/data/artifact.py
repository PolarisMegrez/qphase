"""qphase: Artifact Reference Contracts
---------------------------------------------------------
Freezes the durable-reference side of the data product contract. An
``ArtifactRef`` is the only cross-process/restart reference: it carries an
artifact id, a product name, the product schema and a *trusted storage adapter
id* — never runtime handles, devices, leases, arrays,
cache state or free-form provenance dicts (provenance belongs to the data
product and the artifact manifest; the storage context is resolved by the
artifact store).

A ref never names Python code: the storage adapter id is resolved through the
core adapter registry, so dereferencing a persisted ref cannot import or
execute arbitrary modules.

Public API
----------
ArtifactRef
    Durable, JSON-serializable reference to a persisted product.
DataMaterializerProtocol
    Conversion contract between runtime handles and artifacts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema import ProductSchema

if TYPE_CHECKING:
    from typing import Any

    from .product import DataProduct, RuntimeProductBacking

__all__ = [
    "ArtifactRef",
    "DataMaterializerProtocol",
]

_ADAPTER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*/[0-9]+$")


class ArtifactRef(BaseModel):
    """Durable, cross-process reference to a persisted data product.

    Contains identity only: artifact id, product name, product schema, the
    registered storage adapter id. Recovery
    resolves the adapter through the trusted registry and the artifact
    location through an artifact resolver — the ref itself names no code and
    no filesystem location.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    product_schema: ProductSchema
    storage_adapter: str = Field(
        description="Registered storage adapter id (for example 'npz/3')."
    )

    @field_validator("storage_adapter")
    @classmethod
    def _check_storage_adapter(cls, value: str) -> str:
        if not _ADAPTER_ID_PATTERN.match(value):
            raise ValueError(
                "storage_adapter must be a registry id in 'name/version' "
                f"syntax, got {value!r}"
            )
        return value


@runtime_checkable
class DataMaterializerProtocol(Protocol):
    """Conversion contract between runtime backings and artifact references.

    Implementations must never bypass the lease lifecycle or use the artifact
    store as a session cache.
    """

    def to_artifact(self, product: DataProduct, **kwargs: Any) -> ArtifactRef:
        """Persist a product and return its durable reference."""
        ...

    def to_backing(
        self, ref: ArtifactRef, *, device: str | None = None
    ) -> RuntimeProductBacking:
        """Open an artifact reference as a runtime backing on a device."""
        ...
