"""qphase_sde: Analyzer Base Class
---------------------------------------------------------
Base class for all analyzers in the qphase_sde package, plus the shared
analyser helpers (physical-to-recorded mode-column mapping).

The analyser declaration types (execution capabilities, workspace
request/estimate, work estimates, reducer lifecycle) live in
``qphase_sde.contracts.analyser``; this module only hosts the plugin base
class that consumes them. ``Analyzer`` satisfies
``qphase_sde.contracts.analyser:AnalyserDeclarationContract`` via
``execution_capabilities``/``workspace``/``work_estimate``/
``input_requirements``/``output_specs``/``reducer``.

The engine's typed graph invokes ``execute()``. Built-in implementations may
share numerical kernels with the legacy direct ``analyze()`` API, but graph
execution consumes and returns typed Datasets without passing through an
``AnalysisResult``.

Public API
----------
``Analyzer`` : Base class for analyzers.
``resolve_mode_columns`` : Map physical mode indices to stored columns.
``trajectory_requirement`` : Requirement on the engine trajectory product.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal

from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginBase, PluginConfigBase, ResultProtocol
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    stack_datasets,
)

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
    "select_mode_columns",
    "map_trajectory_product",
    "trajectory_arrays",
    "check_execution_cancelled",
    "report_execution_progress",
    "trajectory_requirement",
]


def check_execution_cancelled(context: Any | None) -> None:
    """Raise when the scheduler cancelled an analyser invocation."""
    cancellation = getattr(context, "cancellation", None)
    if cancellation is not None:
        cancellation.raise_if_cancelled()


def report_execution_progress(
    context: Any | None,
    *,
    completed: float,
    unit: str,
    stage: str = "analyze",
    message: str = "",
) -> None:
    """Report absolute completed work of an analyser loop, when a reporter exists.

    Heartbeat-only by construction: no ``total`` is ever attached, matching the
    analyser ``WorkEstimate(total=None)`` contract that forbids fabricated
    ETAs. ``unit``/``stage`` must match the analyser's declared
    :meth:`Analyzer.work_estimate` conventions (default stage prefix
    ``"analyze"``).
    """
    progress = getattr(context, "progress", None)
    if progress is not None:
        progress.update(completed=completed, unit=unit, stage=stage, message=message)


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


def select_mode_columns(
    recorded_modes: tuple[int, ...], configured: list[int] | None
) -> tuple[list[int], list[int]]:
    modes = list(recorded_modes) if configured is None else list(configured)
    mapping = {mode: index for index, mode in enumerate(recorded_modes)}
    missing = [mode for mode in modes if mode not in mapping]
    if missing:
        raise ValueError(f"requested modes were not recorded: {missing}")
    return modes, [mapping[mode] for mode in modes]


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


def trajectory_arrays(
    product: Dataset,
) -> tuple[Any, float, float, tuple[int, ...]]:
    """Read one point's trajectory array and coordinate metadata."""
    if "alpha" not in {variable.name for variable in product.variables}:
        raise TypeError("trajectory product is missing the alpha variable")
    alpha = product.handle("alpha").materialize()
    if alpha.ndim == 4:
        if product.axis("scan").size != 1:
            raise ValueError("trajectory point input must have a singleton scan axis")
        alpha = alpha[0]
    if alpha.ndim != 3:
        raise ValueError("trajectory alpha must have shape (trajectory, time, channel)")
    axis_names = {axis.name for axis in product.axes}
    time_axis = product.axis("time") if "time" in axis_names else None
    t0 = (
        float(time_axis.start)
        if time_axis is not None and time_axis.start is not None
        else 0.0
    )
    dt = (
        float(time_axis.step)
        if time_axis is not None and time_axis.step is not None
        else 1.0
    )
    raw_modes = product.attributes.get("mode_indices")
    modes = (
        tuple(int(mode) for mode in raw_modes)
        if isinstance(raw_modes, list | tuple)
        else tuple(range(int(alpha.shape[2])))
    )
    return alpha, t0, dt, modes


def map_trajectory_product(
    product: Dataset,
    compute: Callable[[Dataset], Mapping[str, Dataset]],
    *,
    context: Any | None = None,
) -> Mapping[str, Dataset]:
    """Execute a typed point kernel over an optional scan axis."""
    if "scan" not in {axis.name for axis in product.axes}:
        check_execution_cancelled(context)
        result = compute(product)
        check_execution_cancelled(context)
        return result
    size = product.axis("scan").size
    if size is None:
        raise ValueError("trajectory product has an open scan axis")
    if size == 1:
        check_execution_cancelled(context)
        result = compute(product.point_view(scan=0))
        check_execution_cancelled(context)
        return result
    point_products = []
    for index in range(size):
        check_execution_cancelled(context)
        point_products.append(compute(product.point_view(scan=index)))
    check_execution_cancelled(context)
    names = set(point_products[0])
    if any(set(item) != names for item in point_products[1:]):
        raise ValueError("typed trajectory points produced different products")
    return {
        name: stack_datasets([item[name] for item in point_products], axis="scan")
        for name in names
    }


class Analyzer(PluginBase, ABC):
    """Base class for analyzers.

    All analyzers must inherit from this class and implement the abstract
    :meth:`execute` typed entry. Built-ins may retain the temporary direct
    :meth:`analyze` API. Together with
    the subclass-declared ``input_requirements``/``output_specs`` and the
    ``execution_capabilities``/``workspace``/``work_estimate``/``reducer``
    defaults below, this satisfies
    ``qphase_sde.contracts.analyser:AnalyserDeclarationContract``;
    ``execute()`` is the graph-facing typed entry and has no default legacy
    adapter.
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
        """Return the typed reducer for trajectory-batched execution."""
        factory = getattr(self, "create_result_accumulator", None)
        if not callable(factory):
            return None
        from .typed_reducer import TypedProductReducer

        return TypedProductReducer(self, label=str(getattr(self, "name", "analyser")))

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        """Declare the product set of a single-output analyser."""
        raise NotImplementedError(
            f"analyser {type(self).__name__!r} must declare output_specs()"
        )

    @abstractmethod
    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        """Consume typed products and return declared typed products."""

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        """Run the temporary direct analysis API when a plugin supports it.

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
        raise NotImplementedError(
            f"analyser {type(self).__name__!r} only supports typed execute()"
        )
