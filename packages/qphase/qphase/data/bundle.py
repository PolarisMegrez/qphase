"""qphase: Generic Data Bundle
---------------------------------------------------------
The generic bundle is the resource-independent restore result of the current
artifact: a named product collection plus the persisted
:class:`~qphase.data.store.BundleDescriptor`. Core can always restore it,
even when the resource package that produced the artifact is not installed;
resource packages register bundle adapters to restore concrete bundles
(e.g. ``SDEDataBundle``) instead.

The bundle implements the core ``ResultProtocol``/``DatasetResultProtocol``
surface so restored artifacts plug into scheduler input plumbing
(``input.mode=dataset``/``map``) without the producing engine.

Public API
----------
GenericDataBundle
    Resource-independent bundle of typed datasets with a descriptor.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..core.dataset import DatasetSaveReport
from .datasets import Dataset
from .schema import AxisRole
from .store import ARTIFACT_SCHEMA_VERSION, BundleDescriptor, save_products

__all__ = [
    "GenericDataBundle",
]


class GenericDataBundle:
    """Product collection restored from the current artifact (generic bundle).

    ``axes``/``shape``/``point_view`` describe the parameter (scan) axes of
    the anchor product — the first product declaring parameter axes. Axis
    values come from parameter coordinate variables when present, else from
    the integer index range.
    """

    schema_version = "generic.dataset_bundle/1"

    def __init__(
        self,
        products: Mapping[str, Dataset],
        descriptor: BundleDescriptor,
        *,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._products = dict(products)
        self._descriptor = descriptor
        self._provenance = dict(provenance or {})
        self._metadata = dict(metadata or {})

    # -- bundle surface ---------------------------------------------------

    @property
    def products(self) -> dict[str, Dataset]:
        """The restored products by name."""
        return dict(self._products)

    @property
    def bundle_descriptor(self) -> BundleDescriptor:
        """The persisted bundle descriptor."""
        return self._descriptor

    @property
    def provenance(self) -> dict[str, Any]:
        """The artifact provenance mapping."""
        return dict(self._provenance)

    def require(self, name: str) -> Dataset:
        """Return one product, raising ``KeyError`` when absent."""
        try:
            return self._products[name]
        except KeyError:
            raise KeyError(
                f"bundle has no product {name!r}; available: {sorted(self._products)}"
            ) from None

    # -- ResultProtocol -----------------------------------------------------

    @property
    def data(self) -> dict[str, Dataset]:
        """Product mapping (``GenericResult``-compatible access)."""
        return dict(self._products)

    @property
    def metadata(self) -> dict[str, Any]:
        """Bundle metadata (artifact id, bundle type)."""
        return {
            "bundle_type": self._descriptor.type_id,
            **self._metadata,
        }

    def save(self, path: str | Path) -> None:
        """Re-persist the products with the same bundle descriptor."""
        save_products(
            Path(path),
            self._products,
            provenance=self._provenance,
            bundle=self._descriptor,
        )

    # -- DatasetResultProtocol ----------------------------------------------

    def _anchor(self) -> Dataset | None:
        for product in self._products.values():
            if any(axis.role == AxisRole.PARAMETER for axis in product.axes):
                return product
        return None

    @property
    def axes(self) -> Mapping[str, Any]:
        """Parameter axis values of the anchor product."""
        anchor = self._anchor()
        if anchor is None:
            return {}
        coordinates = {
            coordinate.dims[0]: coordinate
            for coordinate in anchor.coordinates()
            if coordinate.role == "parameter" and len(coordinate.dims) == 1
        }
        values: dict[str, Any] = {}
        for axis in anchor.axes:
            if axis.role != AxisRole.PARAMETER:
                continue
            coordinate = coordinates.get(axis.name)
            if coordinate is not None:
                values[axis.name] = anchor.coordinate(coordinate.name)
            else:
                values[axis.name] = np.arange(axis.size or 0)
        return values

    @property
    def shape(self) -> tuple[int, ...]:
        """Scan shape over the anchor product's parameter axes."""
        anchor = self._anchor()
        if anchor is None:
            return ()
        return tuple(
            axis.size or 0 for axis in anchor.axes if axis.role == AxisRole.PARAMETER
        )

    def point_view(self, index: tuple[int, ...]) -> GenericDataBundle:
        """View the bundle at one scan point of the parameter grid."""
        anchor = self._anchor()
        selection: dict[str, int] = {}
        if anchor is not None:
            parameter_axes = [
                axis.name for axis in anchor.axes if axis.role == AxisRole.PARAMETER
            ]
            selection = dict(zip(parameter_axes, index, strict=True))
        views = {
            name: (
                product.point_view(**selection)
                if selection
                and all(axis in {a.name for a in product.axes} for axis in selection)
                else product
            )
            for name, product in self._products.items()
        }
        return GenericDataBundle(
            views,
            self._descriptor,
            provenance=self._provenance,
            metadata={**self._metadata, "point_index": tuple(index)},
        )

    def save_dataset(
        self,
        path: str | Path,
        *,
        layout: str,
        shard_target_bytes: int,
    ) -> DatasetSaveReport:
        """Persist through the current manifest pipeline (DatasetResultProtocol)."""
        root = Path(path)
        resolved = "single" if layout == "single" else "sharded"
        manifest = save_products(
            root,
            self._products,
            provenance=self._provenance,
            shard_target_bytes=shard_target_bytes,
            bundle=self._descriptor,
            layout=resolved,
        )
        files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
        return DatasetSaveReport(
            resolved,
            files,
            loader="+".join(
                sorted({entry.storage.adapter for entry in manifest.products})
            ),
            schema_version=ARTIFACT_SCHEMA_VERSION,
        )
