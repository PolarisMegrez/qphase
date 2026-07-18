"""CAM solver plugins with lazy imports for lightweight spawned workers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import CAMSolver, CAMSolverConfig

_LAZY_SOLVERS = {
    "BatchedNewtonSolver": (".batched_newton", "BatchedNewtonSolver"),
    "ContinuationSolver": (".continuation", "ContinuationSolver"),
    "DeflationSolver": (".deflation", "DeflationSolver"),
    "MultistabilitySolver": (".multistability", "MultistabilitySolver"),
    "SteadyStateSolver": (".steady_state", "SteadyStateSolver"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_SOLVERS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BatchedNewtonSolver",
    "CAMSolver",
    "CAMSolverConfig",
    "ContinuationSolver",
    "DeflationSolver",
    "MultistabilitySolver",
    "SteadyStateSolver",
]
