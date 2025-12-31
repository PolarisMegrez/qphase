"""Simulation Result
=================

Container for SDE simulation results, supporting serialization and deserialization.

Public API
----------
SDEResult
    Container for SDE simulation results.
AnalysisResult
    Container for analysis results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qphase.core.errors import QPhaseError


@dataclass
class SDEResult:
    """Container for SDE simulation results.

    Attributes
    ----------
    trajectory : Any
        The trajectory data (e.g., numpy array or TrajectorySet).
    meta : dict[str, Any]
        Metadata about the simulation (config, runtime info, etc.).
    kind : str
        Type of result ("trajectory", "psd", etc.).

    """

    trajectory: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    kind: Literal["trajectory", "psd"] = "trajectory"

    @property
    def data(self) -> Any:
        """Alias for trajectory to satisfy ResultProtocol."""
        return self.trajectory

    @property
    def metadata(self) -> dict[str, Any]:
        """Alias for meta to satisfy ResultProtocol."""
        return self.meta

    def save(self, path: str | Path) -> None:
        """Save the result to a file.

        Parameters
        ----------
        path : str | Path
            Path to save the result to.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert trajectory to numpy if possible for storage
        data_to_save = self.trajectory
        if hasattr(self.trajectory, "data"):
            data_to_save = self.trajectory.data

        # Extract time info if available
        t0 = getattr(self.trajectory, "t0", 0.0)
        dt = getattr(self.trajectory, "dt", 1.0)

        try:
            # Wrap meta in object array to allow saving dict in npz
            # np.savez expects arrays, so we wrap the dict
            meta_arr = np.array(self.meta, dtype=object)
            np.savez_compressed(
                path, data=data_to_save, t0=t0, dt=dt, meta=meta_arr, kind=self.kind
            )
        except Exception as e:
            raise QPhaseError(f"Failed to save SDEResult to {path}: {e}") from e

    @classmethod
    def load(cls, path: str | Path) -> "SDEResult":
        """Load a result from a file.

        Parameters
        ----------
        path : str | Path
            Path to load the result from.

        Returns
        -------
        SDEResult
            Loaded result object.

        """
        path = Path(path)
        if not path.exists():
            raise QPhaseError(f"File not found: {path}")
        try:
            with np.load(path, allow_pickle=True) as npz:
                data = npz["data"]
                t0 = float(npz["t0"])
                dt = float(npz["dt"])
                meta = npz["meta"].item() if "meta" in npz else {}
                kind = str(npz["kind"]) if "kind" in npz else "trajectory"

                # Reconstruct a simple trajectory object or just return data
                # For now, we return a simple object or the array
                # Ideally, we should use TrajectorySet, but to avoid circular
                # imports
                # we might just return the data or a simple wrapper.

                # Let's use a simple SimpleNamespace or similar if TrajectorySet
                # is not available
                # Or just keep it as data + meta

                # Construct a minimal object that mimics TrajectorySet
                class MinimalTrajectory:
                    def __init__(self, data, t0, dt):
                        self.data = data
                        self.t0 = t0
                        self.dt = dt

                traj = MinimalTrajectory(data, t0, dt)

                return cls(trajectory=traj, meta=meta, kind=kind)  # type: ignore[arg-type]

        except Exception as e:
            raise QPhaseError(f"Failed to load SDEResult from {path}: {e}") from e


# Alias for backward compatibility if needed
SimulationResult = SDEResult


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    results: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Any:
        return self.results

    @property
    def metadata(self) -> dict[str, Any]:
        return self.meta

    def save(self, path: str | Path) -> None:
        """Save analysis results.

        If results contains ResultProtocol objects, delegates saving to them
        with suffixed filenames. Otherwise saves as dictionary pickle/json?
        For now, assumes results are ResultProtocol.
        """
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # If path has extension, strip it for directory-like usage or suffixing
        if base_path.suffix:
            stem = base_path.stem
            parent = base_path.parent
        else:
            stem = base_path.name
            parent = base_path.parent

        for name, res in self.results.items():
            if hasattr(res, "save"):
                # e.g. path/job_name_psd
                sub_path = parent / f"{stem}_{name}"
                res.save(sub_path)
            else:
                # Fallback?
                pass
