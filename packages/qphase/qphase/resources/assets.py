"""qphase: Resource Asset Declarations
---------------------------------------------------------
Defines how a resource package declares non-standard root-level assets (such as
``math/``, ``serialization/`` or ``_native/``) and how the provenance of a
resolved asset (package-owned vs. project overlay vs. third-party) is recorded.

Public API
----------
ResourceAssetDeclaration
    Manifest declaration of one additional package asset.
AssetOrigin
    Provenance of a resolved catalog asset.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AssetOrigin",
    "ResourceAssetDeclaration",
]


class AssetOrigin(str, Enum):
    """Provenance of an asset resolved into a resource catalog.

    Package-owned assets ship with the resource distribution; project overlays
    come from the local project (for example ``.qphase_plugins.yaml``);
    third-party assets come from another installed distribution. Overlays are
    never written back into the resource manifest.
    """

    PACKAGE = "package"
    PROJECT_OVERLAY = "project_overlay"
    THIRD_PARTY = "third_party"


class ResourceAssetDeclaration(BaseModel):
    """Declaration of one additional root-level package asset.

    Standard optional directories (``math/``, ``serialization/``, ``_native/``)
    and any package-specific extra asset must be declared with a purpose and a
    visibility, so that the manifest remains the authoritative asset inventory.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Asset path relative to the package root (e.g. 'math')."
    )
    kind: Literal["module", "directory"] = Field(
        description="Whether the asset is a single module or a directory."
    )
    visibility: Literal["public", "private"] = Field(
        default="private",
        description="Public assets are part of the package's stable API.",
    )
    purpose: str = Field(
        default="",
        description="Short statement of why this asset exists.",
    )
