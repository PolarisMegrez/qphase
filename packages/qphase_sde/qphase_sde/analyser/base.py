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

The engine's typed graph invokes ``execute()``. The existing ``analyze()``
method is an internal numerical kernel; schema materialization stays behind
the typed execution boundary.

Public API
----------
``Analyzer`` : Base class for analyzers.
``resolve_mode_columns`` : Map physical mode indices to stored columns.
``trajectory_requirement`` : Requirement on the engine trajectory product.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

import numpy as np
from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginBase, PluginConfigBase, ResultProtocol
from qphase.data import DataKind, Dataset, ProductDeclaration, ProductRequirement

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

    All analyzers must inherit from this class and implement the abstract
    :meth:`analyze` numerical kernel. Together with
    the subclass-declared ``input_requirements``/``output_specs`` and the
    ``execution_capabilities``/``workspace``/``work_estimate``/``reducer``
    defaults below, this satisfies
    ``qphase_sde.contracts.analyser:AnalyserDeclarationContract``;
    ``execute()`` is the graph-facing typed entry. The default implementation
    adapts the kernel result to the analyser's declared Dataset products.
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
        if not callable(getattr(self, "create_result_accumulator", None)):
            return None
        from .typed_reducer import TypedProductReducer

        return TypedProductReducer(self, label=str(getattr(self, "name", "analyser")))

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        """Declare the product set of a single-output analyser."""
        raise NotImplementedError(
            f"analyser {type(self).__name__!r} must declare output_specs()"
        )

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Dataset]:
        """Execute one analyser on named typed products.

        The numerical kernels remain in ``analyze``; this boundary supplies a
        typed Dataset handles and materializes the analyser's declared products.
        Subclasses with a native typed path may override this method, while
        the default keeps the existing numerical implementation unchanged.
        """
        parameter_axes = [
            axis
            for product in inputs.values()
            for axis in product.axes
            if axis.role.value == "parameter"
        ]
        axes: dict[str, Any] = {}
        shape: tuple[int, ...] = ()
        params: dict[str, Any] = {}
        if isinstance(metadata, Mapping):
            source_params = metadata.get("params")
            if isinstance(source_params, Mapping):
                params.update(source_params)
            source_axes = metadata.get("scan_axes")
            if isinstance(source_axes, Mapping):
                for name, values in source_axes.items():
                    array = np.asarray(values)
                    if array.ndim == 1:
                        axes[str(name)] = array
                if axes:
                    source_shape = metadata.get("scan_shape")
                    if isinstance(source_shape, (tuple, list)) and source_shape:
                        shape = (int(np.prod(source_shape, dtype=np.int64)),)
                    else:
                        shape = tuple(
                            int(np.asarray(values).size) for values in axes.values()
                        )
        scan_param = getattr(self.config, "scan_param", None)
        if isinstance(scan_param, str):
            variable_name = f"parameter.{scan_param}"
            for product in inputs.values():
                if variable_name not in {item.name for item in product.variables}:
                    continue
                values = np.asarray(product.handle(variable_name).materialize())
                params[scan_param] = values
                shape = (int(values.size),)
                axes = {scan_param: values.reshape(-1)}
                break
        if not shape and parameter_axes:
            unique: dict[str, Any] = {
                axis.name: axis for axis in parameter_axes if axis.size != 1
            }
            if unique:
                shape = tuple(int(axis.size) for axis in unique.values())
                axes = {
                    name: np.arange(int(axis.size)) for name, axis in unique.items()
                }
        from ..contracts.bundle import SDEProvenance
        from ..result import SDEDataBundle
        from ..state import TrajectorySet
        from .result import AnalysisResult

        def kernel_input(products: Mapping[str, Dataset]) -> Any:
            trajectory_product = products.get("trajectories")
            if trajectory_product is None:
                return dict(products)
            alpha = trajectory_product.handle("alpha").materialize()
            if (
                alpha.ndim == 4
                and "scan" in {axis.name for axis in trajectory_product.axes}
                and trajectory_product.axis("scan").size == 1
            ):
                alpha = alpha[0]
            time_axis = (
                trajectory_product.axis("time")
                if "time" in {axis.name for axis in trajectory_product.axes}
                else None
            )
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
            trajectory_meta = {
                name: trajectory_product.attributes[name]
                for name in ("mode_indices", "valid_length")
                if name in trajectory_product.attributes
            }
            return TrajectorySet(data=alpha, t0=t0, dt=dt, meta=trajectory_meta)

        analysis_input: Any = inputs
        scanned_products = [
            product
            for product in inputs.values()
            if "scan" in {axis.name for axis in product.axes}
        ]
        if scanned_products and "trajectories" not in inputs:
            input_scan_size = scanned_products[0].axis("scan").size
            if input_scan_size is not None and input_scan_size > 1:
                analysis_input = SDEDataBundle(
                    dict(inputs),
                    SDEProvenance(),
                    scan_axes={"scan": np.arange(input_scan_size)},
                    scan_shape=(input_scan_size,),
                    meta={
                        "scan_parameter_values": dict(axes),
                    },
                )
        elif "trajectories" in inputs:
            analysis_input = kernel_input(inputs)
        elif params:
            analysis_input = SDEDataBundle(
                dict(inputs), SDEProvenance(), meta={"params": params}
            )

        scan_size: int | None
        trajectory = inputs.get("trajectories")
        has_scan_trajectory = (
            trajectory is not None
            and "scan" in {axis.name for axis in trajectory.axes}
            and trajectory.axis("scan").size not in (None, 1)
        )
        has_scan_input = bool(getattr(self, "preserves_input_scan", False)) and any(
            "scan" in {axis.name for axis in product.axes}
            and product.axis("scan").size not in (None, 1)
            for product in inputs.values()
        )
        if has_scan_trajectory or has_scan_input:
            if has_scan_trajectory:
                assert trajectory is not None
                scan_axis_size = trajectory.axis("scan").size
                assert scan_axis_size is not None
                scan_count = int(scan_axis_size)
                scan_bundle = None
            else:
                scan_count = 0
                for product in inputs.values():
                    if "scan" not in {axis.name for axis in product.axes}:
                        continue
                    scan_axis_size = product.axis("scan").size
                    if scan_axis_size not in (None, 1):
                        assert scan_axis_size is not None
                        scan_count = int(scan_axis_size)
                        break
                if scan_count == 0:
                    raise RuntimeError("scan product has no concrete scan size")
                scan_bundle = SDEDataBundle(
                    dict(inputs),
                    SDEProvenance(),
                    scan_axes={"scan": np.arange(scan_count)},
                    scan_shape=(scan_count,),
                    meta={"scan_parameter_values": dict(axes)},
                )
            payloads: list[Any] = []
            for index in range(scan_count):
                if scan_bundle is not None:
                    point_result = cast(
                        AnalysisResult,
                        self.analyze(scan_bundle.point_view((index,)), backend),
                    )
                else:
                    point_inputs = {
                        name: (
                            product.point_view(scan=index)
                            if "scan" in {axis.name for axis in product.axes}
                            else product
                        )
                        for name, product in inputs.items()
                    }
                    point_result = cast(
                        AnalysisResult,
                        self.analyze(kernel_input(point_inputs), backend),
                    )
                if not isinstance(point_result, AnalysisResult):
                    raise TypeError("analyser analyze() must return AnalysisResult")
                payloads.append(point_result.data_dict)
            raw_payload: Any = payloads
            scan_size = scan_count
        else:
            raw_result = cast(
                AnalysisResult,
                self.analyze(analysis_input, backend),
            )
            if not isinstance(raw_result, AnalysisResult):
                raise TypeError("analyser analyze() must return AnalysisResult")
            raw_payload = raw_result.data_dict
            scan_size = int(np.prod(shape, dtype=np.int64)) if shape else None
        builder = getattr(self, "build_products", None)
        if not callable(builder):
            raise NotImplementedError(
                f"analyser {type(self).__name__!r} has no typed product builder"
            )
        products = builder(raw_payload, scan_size=scan_size, label=label)
        if not products:
            raise ValueError(f"analyser {label!r} produced no typed products")
        return products

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
