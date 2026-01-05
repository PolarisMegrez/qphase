"""qphase_sde: Simulation Result
---------------------------------------------------------
Container for SDE simulation results, supporting serialization and deserialization.

Public API
----------
``SDEResult`` : Container for SDE simulation results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qphase.backend.xputil import convert_to_numpy
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

    """

    trajectory: Any = None
    analysis: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Any:
        """Alias for trajectory to satisfy ResultProtocol.

        If trajectory was dropped (e.g. after analysis), return analysis results.
        """
        if self.trajectory is not None:
            return self.trajectory
        if self.analysis:
            return self.analysis
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Alias for meta to satisfy ResultProtocol."""
        return self.meta

    @property
    def label(self) -> Any:
        """Get the label (e.g. parameter value) from metadata."""
        return self.meta.get("label")

    @label.setter
    def label(self, value: Any) -> None:
        """Set the label in metadata."""
        self.meta["label"] = value

    @property
    def index(self) -> Any:
        """Get the index (time/parameter) from the trajectory if available."""
        if hasattr(self.trajectory, "index"):
            return self.trajectory.index
        if hasattr(self.trajectory, "times"):
            return self.trajectory.times
        return None

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
        data_to_save = None
        if self.trajectory is not None:
            if hasattr(self.trajectory, "data"):
                data_to_save = convert_to_numpy(self.trajectory.data)
            else:
                data_to_save = convert_to_numpy(self.trajectory)

        # Extract time info if available
        t0 = getattr(self.trajectory, "t0", self.meta.get("t0", 0.0))
        dt = getattr(self.trajectory, "dt", self.meta.get("dt", 1.0))

        try:
            # Wrap meta in object array to allow saving dict in npz
            # np.savez expects arrays, so we wrap the dict
            meta_arr = np.array(self.meta, dtype=object)
            analysis_arr = np.array(self.analysis, dtype=object)

            save_kwargs = {
                "t0": t0,
                "dt": dt,
                "meta": meta_arr,
                "analysis": analysis_arr,
            }

            if data_to_save is not None:
                save_kwargs["data"] = data_to_save

            np.savez_compressed(path, **save_kwargs)
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
                data = npz["data"] if "data" in npz else None
                t0 = float(npz["t0"]) if "t0" in npz else 0.0
                dt = float(npz["dt"]) if "dt" in npz else 1.0
                meta = npz["meta"].item() if "meta" in npz else {}
                analysis = npz["analysis"].item() if "analysis" in npz else {}

                traj = None
                if data is not None:
                    # Construct a minimal object that mimics TrajectorySet
                    class MinimalTrajectory:
                        def __init__(self, data, t0, dt):
                            self.data = data
                            self.t0 = t0
                            self.dt = dt

                    traj = MinimalTrajectory(data, t0, dt)

                return cls(trajectory=traj, meta=meta, analysis=analysis)

        except Exception as e:
            raise QPhaseError(f"Failed to load SDEResult from {path}: {e}") from e


# Alias for backward compatibility if needed
SimulationResult = SDEResult
