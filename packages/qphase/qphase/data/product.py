"""qphase: Data Product Contract
---------------------------------------------------------
Freezes the semantic layer of typed data products. A ``DataProduct`` couples a
``ProductSchema`` with provenance and a backing. Because one product may hold
variables with heterogeneous dtypes and shapes, a runtime backing is a mapping
from variable name to per-variable handle — never a single flat handle. The
two legal backings (runtime vs. artifact) are distinct types and must never be
conflated.

Public API
----------
RuntimeProductBacking
    Mapping of variable names to their runtime handles.
DataProduct
    Structural protocol of a typed data product.
DataBacking
    Union of the two legal backings.
validate_backing
    Check that a runtime backing matches a product schema exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .artifact import ArtifactRef
from .handles import DataHandleProtocol
from .schema import ProductSchema

__all__ = [
    "DataBacking",
    "DataProduct",
    "RuntimeProductBacking",
    "validate_backing",
]

@runtime_checkable
class RuntimeProductBacking(Protocol):
    """Runtime backing of a data product: one handle per schema variable."""

    @property
    def variables(self) -> Mapping[str, DataHandleProtocol]:
        """Mapping from variable name to its runtime handle."""
        ...


@runtime_checkable
class DataProduct(Protocol):
    """Structural protocol of a typed data product.

    Products are the semantic layer of the data contract: downstream consumers
    select them by kind/quantity/fields, not by internal dict paths or labels.
    Different variables may reside on different devices; planners must handle
    transfers explicitly — implicit device-to-host copies are forbidden.
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
    def backing(self) -> RuntimeProductBacking | ArtifactRef:
        """Runtime backing or artifact reference backing this product."""
        ...


#: A product is backed either by runtime per-variable handles or an artifact.
DataBacking = RuntimeProductBacking | ArtifactRef


def validate_backing(
    schema: ProductSchema, backing: RuntimeProductBacking
) -> None:
    """Check that a runtime backing matches a product schema exactly.

    Every schema variable must correspond to exactly one handle whose dtype
    matches the variable and whose shape matches the variable's dims over
    closed axes. Missing, extra or mismatched handles are rejected.

    Raises
    ------
    ValueError
        On any mismatch between schema and backing.

    """
    handles = dict(backing.variables)
    expected = {variable.name for variable in schema.variables}
    missing = expected - set(handles)
    extra = set(handles) - expected
    if missing:
        raise ValueError(f"backing misses variables: {sorted(missing)}")
    if extra:
        raise ValueError(f"backing has unknown variables: {sorted(extra)}")

    for variable in schema.variables:
        handle = handles[variable.name]
        if handle.variable_schema != variable:
            raise ValueError(
                f"variable {variable.name!r}: handle variable_schema does not "
                "match the product schema"
            )
        if np.dtype(handle.dtype) != np.dtype(variable.dtype):
            raise ValueError(
                f"variable {variable.name!r}: handle dtype {handle.dtype!r} "
                f"does not match schema dtype {variable.dtype!r}"
            )
        if len(handle.shape) != len(variable.dims):
            raise ValueError(
                f"variable {variable.name!r}: handle shape {handle.shape} does "
                f"not match dims {variable.dims}"
            )
        for dim, size in zip(variable.dims, handle.shape, strict=True):
            axis = schema.axis(dim)
            if axis.size is not None and axis.size != size:
                raise ValueError(
                    f"variable {variable.name!r}: handle size {size} on axis "
                    f"{dim!r} does not match closed axis size {axis.size}"
                )
