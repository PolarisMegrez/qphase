"""qphase: Artifact Error Taxonomy
---------------------------------------------------------
Differentiated error types for the artifact store, so service/GUI layers can
distinguish *not found*, *unsupported schema*, *corrupt manifest* and
*missing/unknown adapter* instead of collapsing every failure into one opaque
error.

Public API
----------
ArtifactError
    Base class of all artifact store errors.
ArtifactNotFoundError
    Artifact directory, manifest or referenced payload file is missing.
ArtifactUnsupportedError
    Manifest or descriptor schema version is not supported.
ArtifactCorruptError
    Manifest/payload failed structural or cross-field validation.
ArtifactAdapterError
    Storage adapter is unknown, duplicated or cannot materialize.
"""

from __future__ import annotations

from ..core.errors import QPhaseError

__all__ = [
    "ArtifactAdapterError",
    "ArtifactCorruptError",
    "ArtifactError",
    "ArtifactNotFoundError",
    "ArtifactUnsupportedError",
]


class ArtifactError(QPhaseError):
    """Base class of all artifact store errors."""


class ArtifactNotFoundError(ArtifactError, FileNotFoundError):
    """Artifact directory, manifest or referenced payload file is missing.

    Also subclasses :class:`FileNotFoundError` so existing callers that map
    missing artifacts to 404 keep working.
    """


class ArtifactUnsupportedError(ArtifactError):
    """Manifest or descriptor schema version is not supported."""


class ArtifactCorruptError(ArtifactError):
    """Manifest or payload failed structural/cross-field validation."""


class ArtifactAdapterError(ArtifactError):
    """Storage adapter is unknown, duplicated or cannot materialize."""
