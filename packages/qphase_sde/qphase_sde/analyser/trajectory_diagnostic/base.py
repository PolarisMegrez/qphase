"""Shared context and internal child protocol for trajectory diagnostics.

These children are implementation details of the registered
``analyser.trajectory_diagnostics`` plugin. They are deliberately not QPhase
plugins or scheduler jobs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class TrajectoryDiagnosticContext:
    """Host trajectory and lazily shared whole-state coordinate transforms."""

    values: np.ndarray
    dt: float
    t0: float
    modes: tuple[int, ...]
    mode_columns: tuple[int, ...]
    coordinate_builder: Callable[[np.ndarray], np.ndarray]
    _canonical_coordinates: np.ndarray | None = field(default=None, init=False)

    def series(self, mode: int) -> np.ndarray:
        """Return the host trajectory view for a configured physical mode."""
        index = self.modes.index(mode)
        return self.values[:, :, self.mode_columns[index]]

    def canonical_coordinates(self) -> np.ndarray:
        """Build canonical R coordinates once and share them across children."""
        if self._canonical_coordinates is None:
            self._canonical_coordinates = self.coordinate_builder(self.values)
        return self._canonical_coordinates


class TrajectoryDiagnosticChild(Protocol):
    """Internal result-mutating diagnostic unit run by the parent analyser."""

    name: str

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: Any,
    ) -> None: ...
