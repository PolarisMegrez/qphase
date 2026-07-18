"""Common solver output before fixed-capacity result packing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CAMSolution:
    """One converged or attempted CAM steady state."""

    state: Any
    residual: float
    success: bool
    method: str
    message: str = ""
    iterations: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CAMSolverOutput:
    """Solutions and optional scan axes returned by a solver plugin."""

    solutions: Any
    axes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
