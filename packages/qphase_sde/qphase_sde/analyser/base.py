"""qphase_sde: Analyzer Base Class
---------------------------------------------------------
Base class for all analyzers in the qphase_sde package, plus the shared
analyser helpers (physical-to-recorded mode-column mapping).

Public API
----------
``AnalyzerProtocol`` : Protocol for analyzers.
``resolve_mode_columns`` : Map physical mode indices to stored columns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginBase, PluginConfigBase, ResultProtocol

__all__ = [
    "Analyzer",
    "AnalyzerExecutionCapabilities",
    "AnalyzerProtocol",
    "AnalyzerWorkspaceEstimate",
    "AnalyzerWorkspaceRequest",
    "resolve_mode_columns",
]


def resolve_mode_columns(data: Any, modes: list[int]) -> list[int]:
    """Map physical mode indices to stored trajectory columns."""
    meta = getattr(data, "meta", None)
    mode_indices = meta.get("mode_indices") if isinstance(meta, dict) else None
    if mode_indices is None:
        return list(modes)

    mapping = {int(mode): index for index, mode in enumerate(mode_indices)}
    missing = [mode for mode in modes if mode not in mapping]
    if missing:
        raise ValueError(
            f"requested modes {missing} were not recorded; available modes are "
            f"{list(mapping)}"
        )
    return [mapping[mode] for mode in modes]


@dataclass(frozen=True)
class AnalyzerExecutionCapabilities:
    """Planner-visible execution properties of an analyser.

    ``execution_location`` describes where the analyser's largest temporary
    arrays live. It does not constrain small metadata transfers or final
    result arrays. An analyser must only advertise batching or streaming when
    its public implementation actually supports the corresponding protocol.
    """

    execution_location: Literal["backend", "host"] = "backend"
    requires_full_trajectory: bool = True
    supports_trajectory_batching: bool = False
    supports_time_streaming: bool = False


@dataclass(frozen=True)
class AnalyzerWorkspaceRequest:
    """Shape and dtype facts supplied to analyser workspace estimators."""

    trajectory_bytes: int
    n_traj: int
    saved_samples: int
    n_record_modes: int
    real_itemsize: int
    backend_name: str


@dataclass(frozen=True)
class AnalyzerWorkspaceEstimate:
    """Peak temporary memory attributed to one analyser invocation."""

    device_bytes: int = 0
    host_bytes: int = 0


@runtime_checkable
class AnalyzerProtocol(Protocol):
    """Protocol for analyzers."""

    def capabilities(self) -> AnalyzerExecutionCapabilities: ...

    def estimate_workspace(
        self, request: AnalyzerWorkspaceRequest
    ) -> AnalyzerWorkspaceEstimate: ...

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol: ...


class Analyzer(PluginBase, ABC):
    """Base class for analyzers.

    All analyzers must inherit from this class and implement the
    analyze method.
    """

    config_schema: ClassVar[type[PluginConfigBase]]

    def __init__(self, config: PluginConfigBase | None = None, **kwargs):
        """Initialize the analyzer.

        Parameters
        ----------
        config : PluginConfigBase, optional
            Configuration object. If None, created from kwargs.
        **kwargs : Any
            Configuration parameters if config is not provided.

        """
        if config is None:
            if hasattr(self, "config_schema"):
                self.config = self.config_schema(**kwargs)
            else:
                # Fallback for analyzers without specific config
                # This should ideally not happen if protocols are strictly followed
                pass
        else:
            self.config = config

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        """Return conservative planner-visible execution capabilities."""
        return AnalyzerExecutionCapabilities()

    def estimate_workspace(
        self, request: AnalyzerWorkspaceRequest
    ) -> AnalyzerWorkspaceEstimate:
        """Estimate temporary memory beyond the input trajectory itself."""
        workspace = request.trajectory_bytes // 2
        if self.capabilities().execution_location == "host":
            return AnalyzerWorkspaceEstimate(host_bytes=workspace)
        if request.backend_name == "cupy":
            return AnalyzerWorkspaceEstimate(device_bytes=workspace)
        return AnalyzerWorkspaceEstimate(host_bytes=workspace)

    @abstractmethod
    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        """Perform analysis on the data.

        Parameters
        ----------
        data : Any
            Input data for analysis.
        backend : BackendBase
            Backend to use for computation.

        Returns
        -------
        ResultProtocol
            Analysis results.

        """
        pass
