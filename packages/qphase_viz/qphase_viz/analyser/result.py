"""qphase_viz: Analysis Result
---------------------------------------------------------
Container for analysis results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qphase.core.errors import QPhaseError
from qphase.core.protocols import ResultProtocol


@dataclass
class AnalysisResult(ResultProtocol):
    """Container for analysis results.

    Attributes
    ----------
    data_dict : dict[str, Any]
        The analysis data (e.g., PSD arrays, frequencies).
    meta : dict[str, Any]
        Metadata about the analysis.

    """

    data_dict: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Any:
        """Get the analysis data."""
        return self.data_dict

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the metadata."""
        return self.meta

    @property
    def label(self) -> Any:
        """Get the label from metadata."""
        return self.meta.get("label")

    @label.setter
    def label(self, value: Any) -> None:
        """Set the label in metadata."""
        self.meta["label"] = value

    def save(self, path: str | Path) -> None:
        """Save analysis results to a file (npz)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save as npz
            # We assume data_dict contains numpy arrays or compatible types
            # Wrap meta in object array to allow saving dict in npz
            meta_arr = np.array(self.meta, dtype=object)
            np.savez_compressed(path, **self.data_dict, meta=meta_arr)
        except Exception as e:
            raise QPhaseError(f"Failed to save AnalysisResult to {path}: {e}") from e

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisResult":
        """Load analysis results from a file."""
        path = Path(path)
        if not path.exists():
            raise QPhaseError(f"File not found: {path}")

        try:
            with np.load(path, allow_pickle=True) as npz:
                data_dict = {k: npz[k] for k in npz.files if k != "meta"}
                meta = npz["meta"].item() if "meta" in npz else {}
                return cls(data_dict=data_dict, meta=meta)
        except Exception as e:
            raise QPhaseError(f"Failed to load AnalysisResult from {path}: {e}") from e
