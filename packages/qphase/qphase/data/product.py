"""qphase: Data Product Contract
---------------------------------------------------------
Freezes the semantic layer of typed data products. A ``DataProduct`` couples a
``ProductSchema`` with provenance and a backing: either an in-process runtime
handle or a durable artifact reference. The two backings are distinct types
and must never be conflated.

Public API
----------
DataProduct
    Structural protocol of a typed data product.
DataBacking
    Union of the two legal backings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .artifact import ArtifactRef
from .handles import DataHandleProtocol
from .schema import ProductSchema

__all__ = [
    "DataBacking",
    "DataProduct",
]

#: A product is backed either by an in-process handle or by an artifact ref.
DataBacking = DataHandleProtocol | ArtifactRef


@runtime_checkable
class DataProduct(Protocol):
    """Structural protocol of a typed data product.

    Products are the semantic layer of the data contract: downstream consumers
    select them by kind/quantity/fields, not by internal dict paths or labels.
    """

    @property
    def schema(self) -> ProductSchema:
        """The product's machine-readable schema."""
        ...

    @property
    def provenance(self) -> Mapping[str, Any]:
        """JSON-serializable provenance (plugin/config/backend fingerprints)."""
        ...

    @property
    def backing(self) -> DataBacking:
        """Runtime handle or artifact reference backing this product."""
        ...
