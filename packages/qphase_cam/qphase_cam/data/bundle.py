"""qphase_cam: the product bundle
---------------------------------------------------------
CAM products form the core :class:`~qphase.data.GenericDataBundle` — there is
no CAM-specific bundle class and no CAM-specific artifact adapter. A second
result type would duplicate what the generic bundle already restores, and
Phase 10E is the phase that owns any engine-level bundle.

The persisted descriptor is deliberately the *generic* one
(``generic.dataset_bundle/1`` + the ``generic/1`` adapter): a CAM bundle type
would need a registered adapter, and registering one is a packaging decision
Phase 10E owns. Resource semantics still survive, through
``descriptor.product_roles`` (role → job-local product name) and through each
product's own ``product`` attribute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qphase.data import BundleDescriptor, Dataset, GenericDataBundle

__all__ = [
    "CAM_BUNDLE_ADAPTER_ID",
    "CAM_BUNDLE_TYPE_ID",
    "CAM_PRODUCT_ROLES",
    "cam_bundle",
]

#: Bundle type of a CAM product collection. The core ``generic/1`` adapter is
#: used on purpose: a CAM-specific bundle type would need a registered adapter,
#: and Phase 10E owns the engine handover that could register one.
CAM_BUNDLE_TYPE_ID = "generic.dataset_bundle/1"
CAM_BUNDLE_ADAPTER_ID = "generic/1"

#: Semantic product role restored by the bundle descriptor, keyed by the
#: ``product`` attribute of a product.
CAM_PRODUCT_ROLES = {
    "state": "state",
    "state_scan": "state",
    "candidates": "candidates",
}


def cam_bundle(
    products: Mapping[str, Dataset],
    *,
    provenance: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    parents: Sequence[str] = (),
) -> GenericDataBundle:
    """Group CAM products into the core generic bundle.

    The semantic role of each product comes from its own ``product`` attribute,
    so the bundle stays resource-generic while ``descriptor.product_roles``
    still answers "which product is the state?".
    """
    roles: dict[str, str] = {}
    for name, product in products.items():
        product_kind = str(product.attributes.get("product", name))
        roles[CAM_PRODUCT_ROLES.get(product_kind, name)] = name
    descriptor = BundleDescriptor(
        type_id=CAM_BUNDLE_TYPE_ID,
        adapter_id=CAM_BUNDLE_ADAPTER_ID,
        descriptor_schema=CAM_BUNDLE_TYPE_ID,
        descriptor={},
        product_roles=roles,
    )
    return GenericDataBundle(
        products,
        descriptor,
        provenance=provenance,
        metadata=metadata,
        parents=parents,
    )
