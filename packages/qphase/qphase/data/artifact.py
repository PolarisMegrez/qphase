"""qphase: Artifact Reference Contracts
---------------------------------------------------------
Freezes the durable-reference side of the data product contract. An
``ArtifactRef`` is the only cross-process/restart reference: it carries an
artifact id, the product schema, a loader reference, a content hash and
provenance — never arrays. ``DataMaterializerProtocol`` is implemented by
resource packages to convert between runtime handles and artifact-backed
products without bypassing the core lease lifecycle.

Public API
----------
ArtifactRef
    Durable, JSON-serializable reference to a persisted product.
DataMaterializerProtocol
    Conversion contract between runtime handles and artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .schema import ProductSchema

if TYPE_CHECKING:
    from .handles import DataHandleProtocol
    from .product import DataProduct

__all__ = [
    "ArtifactRef",
    "DataMaterializerProtocol",
]


class ArtifactRef(BaseModel):
    """Durable, cross-process reference to a persisted data product.

    Contains only identity and loading information; recovery goes through the
    referenced loader, which is the single public restore entry point.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    product_schema: ProductSchema
    loader: str = Field(
        description="Dotted path or entry-point reference of the loader."
    )
    content_hash: str = Field(
        description="Content hash of the persisted payload."
    )
    provenance: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class DataMaterializerProtocol(Protocol):
    """Conversion contract between runtime handles and artifact references."""

    def to_artifact(self, product: DataProduct, **kwargs: Any) -> ArtifactRef:
        """Persist a product and return its durable reference."""
        ...

    def to_handle(
        self, ref: ArtifactRef, *, device: str | None = None
    ) -> DataHandleProtocol:
        """Open an artifact reference as a runtime handle on a device."""
        ...
