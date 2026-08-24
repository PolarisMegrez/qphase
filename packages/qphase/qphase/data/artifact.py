"""qphase: Artifact Reference Contracts
---------------------------------------------------------
Freezes the durable-reference side of the data product contract. An
``ArtifactRef`` is the only cross-process/restart reference: it carries an
artifact id, the product schema, a loader reference and a content hash —
never runtime handles, devices, leases, arrays, cache state or free-form
provenance dicts (provenance belongs to the data product and the artifact
manifest; the loader's storage context is resolved by the artifact store).

Public API
----------
ArtifactRef
    Durable, JSON-serializable reference to a persisted product.
DataMaterializerProtocol
    Conversion contract between runtime handles and artifacts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema import ProductSchema

if TYPE_CHECKING:
    from .product import DataProduct, RuntimeProductBacking

__all__ = [
    "ArtifactRef",
    "DataMaterializerProtocol",
]

_DOTTED_TARGET_PATTERN = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class ArtifactRef(BaseModel):
    """Durable, cross-process reference to a persisted data product.

    Contains identity and loading information only; recovery goes through the
    referenced loader, which is the single public restore entry point.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    product_schema: ProductSchema
    loader: str = Field(
        description="Stable loader reference in 'module:attr' syntax."
    )
    content_hash: str = Field(
        description="Content hash of the persisted payload."
    )
    hash_algorithm: str = Field(
        default="sha256",
        description="Algorithm of ``content_hash``; fixed to sha256 unless a "
        "future schema version declares otherwise.",
    )

    @field_validator("loader")
    @classmethod
    def _check_loader(cls, value: str) -> str:
        if not _DOTTED_TARGET_PATTERN.match(value):
            raise ValueError(
                f"loader must use stable 'module:attr' syntax, got {value!r}"
            )
        return value


@runtime_checkable
class DataMaterializerProtocol(Protocol):
    """Conversion contract between runtime backings and artifact references.

    Implementations must never bypass the lease lifecycle or use the artifact
    store as a session cache.
    """

    def to_artifact(
        self, product: DataProduct, **kwargs: Any
    ) -> ArtifactRef:
        """Persist a product and return its durable reference."""
        ...

    def to_backing(
        self, ref: ArtifactRef, *, device: str | None = None
    ) -> RuntimeProductBacking:
        """Open an artifact reference as a runtime backing on a device."""
        ...
