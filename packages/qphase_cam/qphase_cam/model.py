"""Structural contracts for CAM-capable physical model plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable


@dataclass(frozen=True)
class CAMSymbolicSpec:
    """Symbolic matrices and canonical state symbols for Jacobian generation."""

    hamiltonian: Any
    diffusion: Any
    state_matrix: Any
    state_symbols: tuple[Any, ...]
    parameter_symbols: tuple[Any, ...]
    version: str = "1"


@runtime_checkable
class CAMModel(Protocol):
    """Capability required by CAM solvers."""

    name: ClassVar[str]
    n_modes: int
    params: dict[str, Any]
    steady_state_capacity: ClassVar[int]

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any: ...

    def cam_diffusion(self, state: Any, params: dict[str, Any]) -> Any: ...

    def cam_solution_sort_key(self, state: Any, params: dict[str, Any]) -> float: ...


@runtime_checkable
class CAMVectorModel(Protocol):
    """Optional direct canonical-coordinate capability for solver hot paths."""

    def cam_residual_vector(self, vector: Any, params: dict[str, Any]) -> Any: ...

    def cam_jacobian_vector(self, vector: Any, params: dict[str, Any]) -> Any: ...


@runtime_checkable
class CAMBogoliubovModel(CAMModel, Protocol):
    """Optional interaction block used for linearized fluctuation spectra."""

    def cam_bogoliubov_interaction(self, state: Any, params: dict[str, Any]) -> Any: ...


@runtime_checkable
class CAMBifurcationModel(Protocol):
    """Optional exact symbolic dynamics capability for bifurcation solvers."""

    @classmethod
    def cam_fpgen_dynamics(cls) -> Any: ...

    def cam_bifurcation_scales(self, params: dict[str, Any]) -> Any: ...
