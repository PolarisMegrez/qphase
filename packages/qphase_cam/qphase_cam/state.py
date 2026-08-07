"""Common solver output before fixed-capacity result packing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


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


@dataclass
class CAMBifurcationCandidate:
    """One attempted high-order equilibrium candidate."""

    state_vector: Any
    controls: dict[str, float]
    full_residual_norm: float
    search_residual_norm: float
    success: bool
    status: str
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CAMBifurcationOutput:
    """Variable-length candidate output returned by bifurcation solvers."""

    candidates: list[CAMBifurcationCandidate]
    target: str
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)


#: Union of all outputs a CAM solver plugin may return.
CAMSolverResult: TypeAlias = "CAMSolverOutput | CAMBifurcationOutput"
