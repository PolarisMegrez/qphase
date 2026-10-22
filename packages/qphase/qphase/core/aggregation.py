"""qphase: Generic result aggregation utilities
---------------------------------------------------------
Core helpers for collecting and sorting results that come from multiple
upstream Jobs or from a result directory. Resource packages use these
utilities to implement their own cross-job workflows.

Public API
----------
AggregateResult
    Container that wraps multiple ``ResultProtocol`` objects.
DirectoryInputResult
    Wrapper that passes an existing result directory as input data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .protocols import ResultProtocol


@dataclass
class AggregateResult(ResultProtocol):
    """Container that aggregates multiple ``ResultProtocol`` objects.

    Parameters
    ----------
    results : dict[str, ResultProtocol]
        Mapping from job/result name to result object.
    meta : dict[str, Any], optional
        Additional metadata about the aggregation.

    """

    results: dict[str, ResultProtocol]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> dict[str, ResultProtocol]:
        """Return the aggregated result dictionary."""
        return self.results

    @property
    def metadata(self) -> dict[str, Any]:
        """Return aggregation metadata."""
        return self.meta

    @property
    def label(self) -> Any:
        """Return aggregation label if any."""
        return self.meta.get("label")

    def save(self, path: str | Path) -> None:
        """Save is not implemented for aggregated views.

        Resource packages materialize aggregated data as typed products
        through their own analysers/exporters.
        """
        raise NotImplementedError(
            "AggregateResult cannot be saved directly; persist the aggregated "
            "data as typed products instead."
        )

    def items(self):
        """Iterate over (name, result) pairs."""
        return self.results.items()

    def values(self):
        """Iterate over result objects."""
        return self.results.values()

    def sort_by_metadata(
        self, param_name: str
    ) -> list[tuple[str, ResultProtocol, Any]]:
        """Return (name, result, param_value) tuples sorted by a metadata field.

        The field is looked up in ``result.metadata.get("params", {})`` first,
        then directly in ``result.metadata``.

        """
        keyed: list[tuple[Any, str, ResultProtocol]] = []
        for name, result in self.results.items():
            meta = getattr(result, "metadata", {}) or {}
            params = meta.get("params", {}) if isinstance(meta, dict) else {}
            value = params.get(param_name, meta.get(param_name))
            keyed.append((_sort_key(value), name, result))
        keyed.sort(key=lambda item: item[0])
        return [(name, result, value) for _, name, result in keyed]


@dataclass
class DirectoryInputResult(ResultProtocol):
    """Wrapper that passes an existing result directory as input data.

    Resource-package engines/analyzers receive the directory ``Path`` as
    ``data`` and are responsible for loading their own result files.

    """

    path: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Path:
        """Return the directory path."""
        return self.path

    @property
    def metadata(self) -> dict[str, Any]:
        """Return directory input metadata."""
        return self.meta

    @property
    def label(self) -> Any:
        """Return label if any."""
        return self.meta.get("label")

    def save(self, path: str | Path) -> None:
        """Directory input is read-only."""
        raise NotImplementedError("DirectoryInputResult is read-only")


def _sort_key(value: Any) -> tuple[int, Any]:
    """Return a sort key that handles numeric and non-numeric values."""
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))
