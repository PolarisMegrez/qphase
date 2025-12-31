"""Result Loader
=============

Utilities for loading results from disk, primarily for resume capability.

Public API
----------
load_result
    Load a result from a file.
GenericResult
    Generic container for loaded results.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import QPhaseError
from .protocols import ResultProtocol


@dataclass
class GenericResult:
    """Generic container for loaded results.

    Parameters
    ----------
    _data : Any
        The result data.
    _metadata : dict[str, Any]
        Metadata associated with the result.

    """

    _data: Any
    _metadata: dict[str, Any]

    @property
    def data(self) -> Any:
        """Get the result data."""
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the result metadata."""
        return self._metadata

    def save(self, path: str | Path) -> None:
        """Save the result (Not Implemented).

        Parameters
        ----------
        path : str | Path
            Path to save to.

        Raises
        ------
        NotImplementedError
            Always raises as GenericResult is read-only.

        """
        raise NotImplementedError("GenericResult is read-only")


def load_result(job_name: str, job_dir: Path) -> ResultProtocol:
    """Attempt to load a result from a job directory.

    Parameters
    ----------
    job_name : str
        Name of the job (used to guess filename).
    job_dir : Path
        Directory containing the job output.

    Returns
    -------
    ResultProtocol
        The loaded result.

    Raises
    ------
    QPhaseError
        If no supported result file is found.

    """
    if not job_dir.exists():
        raise QPhaseError(f"Job directory not found: {job_dir}")

    # Potential filenames to check
    # 1. {job_name}.npz (Default for SDE)
    # 2. {job_name}.json
    # 3. result.npz (Generic fallback)
    # 4. result.json
    candidates = [
        job_dir / f"{job_name}.npz",
        job_dir / f"{job_name}.json",
        job_dir / "result.npz",
        job_dir / "result.json",
    ]

    for path in candidates:
        if path.exists():
            return _load_file(path)

    # Scan directory for any .npz or .json if exact match failed
    for ext in [".npz", ".json"]:
        matches = list(job_dir.glob(f"*{ext}"))
        if len(matches) == 1:
            return _load_file(matches[0])

    raise QPhaseError(f"No supported result file found in {job_dir}")


def _load_file(path: Path) -> ResultProtocol:
    """Load a specific file into a GenericResult."""
    try:
        if path.suffix == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=True) as npz:
                # Heuristic to extract data
                if "data" in npz:
                    data = npz["data"]
                else:
                    # If no 'data' key, return the whole dict-like object
                    data = dict(npz)

                # Heuristic to extract metadata
                metadata = {}
                if "meta" in npz:
                    meta_item = npz["meta"]
                    # Handle 0-d array wrapping dict
                    if meta_item.ndim == 0:
                        metadata = meta_item.item()
                    else:
                        metadata = {str(k): v for k, v in enumerate(meta_item)}

                # Special handling for SDE results:
                # If data is a numpy array, it might be the trajectory.
                # We return it as is.
                return GenericResult(data, metadata)

        elif path.suffix == ".json":
            import json

            with open(path, encoding="utf-8") as f:
                content = json.load(f)
                # Assume standard structure if possible
                if isinstance(content, dict) and "data" in content:
                    return GenericResult(content["data"], content.get("metadata", {}))
                return GenericResult(content, {})

        else:
            raise QPhaseError(f"Unsupported file extension: {path.suffix}")

    except Exception as e:
        raise QPhaseError(f"Failed to load result from {path}: {e}") from e
