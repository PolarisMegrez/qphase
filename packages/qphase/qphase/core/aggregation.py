"""qphase: Generic result aggregation utilities
---------------------------------------------------------
Core helpers for collecting, sorting, and exporting results that come from
multiple upstream jobs or from a run directory. Resource packages can use
these utilities to implement their own cross-job workflows without duplicating
generic serialization logic.

Public API
----------
AggregateResult
    Container that wraps multiple ``ResultProtocol`` objects.
iter_directory_results
    Discover result files under a run directory.
write_table_csv
    Write a list of dict rows to a CSV file.
write_columns_csv
    Write a frequency/axis column plus named value columns to CSV.
write_npz_bundle
    Write an NPZ bundle with schema/version metadata.
write_pkl_bundle
    Write a pickle bundle with schema/version metadata.
"""

from __future__ import annotations

import csv
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qphase.core.errors import QPhaseError

from .protocols import ResultProtocol

QPHASE_BUNDLE_SCHEMA_VERSION = "1.0"


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

        Resource packages should use the export helpers in this module to
        materialize aggregated data.
        """
        raise NotImplementedError(
            "AggregateResult cannot be saved directly; use the export helpers."
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
    """Wrapper that passes an existing run directory as input data.

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


def iter_directory_results(run_dir: str | Path, pattern: str = "*.npz") -> list[Path]:
    """Discover result files under a run directory.

    The default discovery strategy mirrors the common layout produced by
    parameter scans: ``run_dir/<job_name>/<result>.npz``. If no files are
    found in subdirectories, the run directory itself is scanned.

    Parameters
    ----------
    run_dir : str | Path
        Run directory or a single result file.
    pattern : str, optional
        Glob pattern for result files, by default ``"*.npz"``.

    Returns
    -------
    list[Path]
        Sorted list of discovered file paths.

    Raises
    ------
    QPhaseError
        If the directory does not exist or no files match the pattern.

    """
    root = Path(run_dir)
    if root.is_file():
        return [root]
    if not root.exists():
        raise QPhaseError(f"Run directory does not exist: {root}")

    files = sorted(path for path in root.glob(f"*/{pattern}") if path.is_file())
    if not files:
        files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise QPhaseError(f"No result files matching {pattern!r} found under {root}")
    return files


def load_directory_results(
    run_dir: str | Path,
    loader: Callable[[Path], ResultProtocol] | None = None,
    pattern: str = "*.npz",
) -> dict[str, ResultProtocol] | dict[str, Path]:
    """Discover and optionally load results under a run directory.

    Parameters
    ----------
    run_dir : str | Path
        Run directory or single file.
    loader : Callable[[Path], ResultProtocol] | None, optional
        Loader to use for each discovered file. If ``None``, paths are returned.
    pattern : str, optional
        Glob pattern for result files.

    Returns
    -------
    dict[str, ResultProtocol] | dict[str, Path]
        Mapping from a derived result name to the loaded result or path.

    """
    paths = iter_directory_results(run_dir, pattern)
    if loader is None:
        return {path.parent.name or path.stem: path for path in paths}

    loaded: dict[str, ResultProtocol] = {}
    for path in paths:
        name = path.parent.name or path.stem
        loaded[name] = loader(path)
    return loaded


def write_table_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> Path:
    """Write a list of dict rows to a CSV file.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows to write.
    path : str | Path
        Output file path.
    fieldnames : list[str] | None, optional
        Explicit CSV header. If ``None``, keys from the first row are used.

    Returns
    -------
    Path
        The written file path.

    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise QPhaseError("Cannot write empty CSV without explicit fieldnames")

    header = fieldnames if fieldnames is not None else list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_columns_csv(
    axis: Any,
    columns: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write an axis column plus named value columns to CSV.

    Parameters
    ----------
    axis : array-like
        Common axis values (e.g. frequency).
    columns : dict[str, array-like]
        Mapping from column label to values aligned with ``axis``.
    path : str | Path
        Output file path.

    Returns
    -------
    Path
        The written file path.

    """
    import numpy as np

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = list(columns.keys())
    axis_arr = np.asarray(axis)
    col_arrays = {label: np.asarray(columns[label]) for label in labels}

    if axis_arr.ndim != 1:
        raise QPhaseError("write_columns_csv expects a 1-D axis")
    for label, arr in col_arrays.items():
        if arr.shape[0] != axis_arr.shape[0]:
            raise QPhaseError(
                f"Column '{label}' length {arr.shape[0]} does not match "
                f"axis length {axis_arr.shape[0]}"
            )

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", *labels])
        for i, value in enumerate(axis_arr):
            writer.writerow([value, *(col_arrays[label][i] for label in labels)])
    return out


def _bundle_meta() -> dict[str, str]:
    """Return standard bundle metadata fields."""
    from datetime import UTC, datetime

    return {
        "__schema_version__": QPHASE_BUNDLE_SCHEMA_VERSION,
        "__created_by__": "qphase",
        "__created_at__": datetime.now(UTC).isoformat(),
    }


def write_npz_bundle(path: str | Path, **arrays_and_meta: Any) -> Path:
    """Write a compressed NPZ bundle with schema/version metadata.

    Schema fields are injected automatically unless explicitly provided.

    Parameters
    ----------
    path : str | Path
        Output file path.
    **arrays_and_meta : Any
        Arrays or metadata to store in the bundle.

    Returns
    -------
    Path
        The written file path.

    """
    import numpy as np

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(arrays_and_meta)
    for key, value in _bundle_meta().items():
        kwargs.setdefault(key, value)

    # Convert dict/list metadata to object arrays so np.savez can store them.
    for key, value in list(kwargs.items()):
        if isinstance(value, (dict, list)):
            kwargs[key] = np.array(value, dtype=object)

    np.savez_compressed(out, **kwargs)
    return out


def write_pkl_bundle(path: str | Path, rows: list[Any]) -> Path:
    """Write a pickle bundle with schema/version metadata.

    Parameters
    ----------
    path : str | Path
        Output file path.
    rows : list[Any]
        Serializable rows/objects to store under the ``rows`` key.

    Returns
    -------
    Path
        The written file path.

    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = {"rows": rows}
    bundle.update(_bundle_meta())

    with out.open("wb") as handle:
        pickle.dump(bundle, handle)
    return out


def _sort_key(value: Any) -> tuple[int, Any]:
    """Return a sort key that handles numeric and non-numeric values."""
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))
