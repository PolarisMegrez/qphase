"""qphase_sde: Analyzer Base Class
---------------------------------------------------------
Base class for all analyzers in the qphase_sde package, plus the shared
analyser helpers (physical-to-recorded mode-column mapping).

The engine-facing contract types (execution capabilities, workspace
request/estimate, work estimates, reducer lifecycle) live in
``qphase_sde.contracts.analyser``; this module only hosts the plugin base
class that consumes them.

Public API
----------
``Analyzer`` : Base class for analyzers.
``resolve_mode_columns`` : Map physical mode indices to stored columns.
``trajectory_requirement`` : Requirement on the engine trajectory product.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginBase, PluginConfigBase, ResultProtocol
from qphase.data import DataKind, ProductRequirement

from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from ..contracts.quantities import SDEQuantity
from ..contracts.reducer import ReducerProtocol

__all__ = [
    "Analyzer",
    "resolve_mode_columns",
    "trajectory_requirement",
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


def trajectory_requirement(
    *, value_domain: Literal["real", "complex"] | None = "complex"
) -> ProductRequirement:
    """Return a requirement on the engine's ``trajectories`` time-series product."""
    return ProductRequirement(
        name="trajectories",
        kind=DataKind.TIME_SERIES,
        quantity=SDEQuantity.FIELD_AMPLITUDE.value,
        value_domain=value_domain,
        required_fields=["alpha"],
    )


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

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        """Return conservative planner-visible execution capabilities."""
        return AnalyserExecutionCapabilities()

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        """Estimate temporary memory beyond the input trajectory itself."""
        workspace = request.trajectory_bytes // 2
        if self.execution_capabilities().preferred_location == "host":
            return AnalyserWorkspaceEstimate(host_bytes=workspace)
        if request.backend_name == "cupy":
            return AnalyserWorkspaceEstimate(device_bytes=workspace)
        return AnalyserWorkspaceEstimate(host_bytes=workspace)

    def work_estimate(self, request: AnalyserWorkspaceRequest) -> WorkEstimate:
        """Heartbeat-only default: ``total=None`` never fabricates an ETA."""
        del request
        return WorkEstimate(unit="sample", total=None)

    def reducer(self) -> ReducerProtocol | None:
        """Return the reducer for batched/streaming execution, if supported."""
        return None

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
