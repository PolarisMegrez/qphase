"""qphase_sde: Sampled-Observable Transform (2.0)
---------------------------------------------------------
Typed ``observable_transform`` analyser: maps complex field trajectories to
real sampled-observable time series ``y_o(t) = Re[alpha^dagger Q_o alpha] -
c_o`` using the shared Hermitian quadratic mathematics of
``qphase_sde.math.observable`` (the same evaluator ``quadratic_moments``
uses). The output closes the open ``SAMPLED_OBSERVABLE_PRODUCT`` template:
it mirrors the source scan/trajectory/time axes and ``valid_length`` while
the channel axis indexes the declared observable channels.

The transform is a transient graph node: it declares no reducer, so its
output is only legal batch-locally (consumed in the same trajectory batch)
or in unbatched execution; the planner rejects published batched outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginConfigBase
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
)
from qphase.data.runtime import (
    BackendArrayHandle,
    DictProductBacking,
    HostArrayHandle,
)

from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
)
from ..contracts.quantities import SAMPLED_OBSERVABLE_PRODUCT, SDEQuantity
from ..math.observable import (
    QuadraticObservableSpec,
    compile_quadratic_observables,
    complex_matrix_json,
    evaluate_quadratic_observables,
    stage_quadratic_observables,
)
from .base import Analyzer, check_execution_cancelled, select_mode_columns

__all__ = [
    "ObservableTransformAnalyzer",
    "ObservableTransformConfig",
]


class ObservableTransformConfig(PluginConfigBase):
    """Configuration of one Hermitian quadratic observable transform."""

    observables: dict[str, QuadraticObservableSpec] = Field(
        min_length=1,
        description="Named Hermitian quadratic observables; names become "
        "the observable channel labels",
    )
    modes: list[int] | None = Field(
        None,
        min_length=1,
        description="Physical mode indices the matrices act on; None uses "
        "every recorded mode",
    )
    units: str = Field(
        min_length=1,
        description="Physical units shared by every channel of this instance",
    )
    quantity: Literal["sampled_observable"] = Field(
        SDEQuantity.SAMPLED_OBSERVABLE.value,
        description="Quantity of the output value variable (frozen)",
    )
    hermitian_tolerance: float = Field(
        1e-10,
        ge=0.0,
        description="Relative tolerance for Hermitian matrix validation",
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> ObservableTransformConfig:
        if self.modes is not None:
            if len(set(self.modes)) != len(self.modes):
                raise ValueError("modes must not contain duplicates")
            if any(mode < 0 for mode in self.modes):
                raise ValueError("modes must be non-negative")
        if any(not name.strip() for name in self.observables):
            raise ValueError("observable names must not be blank")
        return self


class ObservableTransformAnalyzer(Analyzer):
    """Map complex field trajectories to sampled-observable time series."""

    name: ClassVar[str] = "observable_transform"
    description: ClassVar[str] = (
        "Real sampled-observable time series from Hermitian quadratic forms"
    )
    config_schema: ClassVar[type[ObservableTransformConfig]] = (
        ObservableTransformConfig
    )

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(
            preferred_location="backend",
            supports_trajectory_batching=True,
            supports_time_streaming=False,
            supports_scan_tiling=True,
        )

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        """Estimate per-invocation and retained workspace of the transform.

        The retained fields cover the whole per-point output series: with no
        reducer there is no finalize stage, so the output product itself is
        the largest object alive until the last batch-local consumer (or the
        publisher) releases it.
        """
        config = cast(ObservableTransformConfig, self.config)
        n_observables = len(config.observables)
        n_modes = (
            len(config.modes) if config.modes is not None else request.n_record_modes
        )
        complex_itemsize = 2 * request.real_itemsize
        selected = (
            request.n_traj * request.saved_samples * n_modes * complex_itemsize
        )
        matrices = n_observables * n_modes * n_modes * complex_itemsize
        output = (
            request.n_traj
            * request.saved_samples
            * n_observables
            * request.real_itemsize
        )
        total_n_traj = request.total_n_traj or request.n_traj
        retained_output = (
            total_n_traj
            * request.saved_samples
            * n_observables
            * request.real_itemsize
        )
        if request.backend_name == "cupy":
            return AnalyserWorkspaceEstimate(
                device_bytes=selected + matrices + output,
                retained_device_bytes=retained_output,
            )
        return AnalyserWorkspaceEstimate(
            host_bytes=selected + matrices + output,
            retained_host_bytes=retained_output,
        )

    def input_requirements(self) -> list[ProductRequirement]:
        return [
            ProductRequirement(
                name="signal",
                kind=DataKind.TIME_SERIES,
                quantity=SDEQuantity.FIELD_AMPLITUDE.value,
                value_domain="complex",
                required_fields=["alpha", "valid_length"],
            )
        ]

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.TIME_SERIES,
                quantity=SDEQuantity.SAMPLED_OBSERVABLE.value,
                fields=["value", "valid_length"],
            ),
        )

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        signal = inputs.get("signal")
        if signal is None:
            raise TypeError("observable_transform requires a 'signal' input product")
        config = cast(ObservableTransformConfig, self.config)
        variables = {variable.name: variable for variable in signal.variables}
        if "alpha" not in variables or "valid_length" not in variables:
            raise TypeError(
                "observable_transform signal must carry alpha and valid_length"
            )
        alpha_handle = signal.handle("alpha")
        alpha = alpha_handle.materialize()
        if not np.issubdtype(alpha.dtype, np.complexfloating):
            raise ValueError("observable_transform expects a complex alpha variable")
        axis_by_name = {axis.name: axis for axis in signal.axes}
        if "time" not in axis_by_name:
            raise TypeError("observable_transform signal is missing the time axis")
        has_scan = "scan" in axis_by_name
        if alpha.ndim != (4 if has_scan else 3):
            shape = "(scan, trajectory, time, channel)" if has_scan else (
                "(trajectory, time, channel)"
            )
            raise ValueError(
                f"observable_transform alpha must have shape {shape}, "
                f"got {alpha.shape}"
            )
        n_channel = int(alpha.shape[-1])
        raw_modes = signal.attributes.get("mode_indices")
        recorded = (
            tuple(int(mode) for mode in raw_modes)
            if isinstance(raw_modes, list | tuple)
            else tuple(range(n_channel))
        )
        modes, columns = select_mode_columns(recorded, config.modes)
        compiled = compile_quadratic_observables(
            config.observables, len(modes), config.hermitian_tolerance
        )
        selected = alpha[..., columns]
        staged_matrices, staged_centers = stage_quadratic_observables(
            compiled, selected, backend
        )
        value = evaluate_quadratic_observables(
            selected,
            staged_matrices,
            staged_centers,
            backend=backend,
            tolerance=config.hermitian_tolerance,
        )
        check_execution_cancelled(context)

        axes: list[AxisSchema] = []
        dims_prefix: tuple[str, ...] = ()
        if has_scan:
            axes.append(axis_by_name["scan"])
            dims_prefix = ("scan",)
        trajectory_axis = axis_by_name.get("trajectory")
        if trajectory_axis is None:
            trajectory_axis = AxisSchema(
                name="trajectory",
                role=AxisRole.REALIZATION,
                size=int(alpha.shape[-3]),
            )
        axes.append(trajectory_axis)
        axes.append(axis_by_name["time"])
        axes.append(
            AxisSchema(
                name="channel", role=AxisRole.COMPONENT, size=len(compiled.names)
            )
        )
        value_variable = VariableSchema(
            name="value",
            dtype=np.dtype(value.dtype).str,
            value_domain="real",
            dims=(*dims_prefix, "trajectory", "time", "channel"),
            quantity=SDEQuantity.SAMPLED_OBSERVABLE.value,
            units=config.units,
        )
        valid_variable = variables["valid_length"]
        source_product = ""
        if metadata is not None:
            source_product = str(metadata.get("source_product", ""))
        attributes = dict(SAMPLED_OBSERVABLE_PRODUCT.attributes)
        attributes.update(
            {
                "source_analyser": str(label),
                "channel_names": list(compiled.names),
                "mode_labels": [int(mode) for mode in modes],
                "matrices_requested": [
                    complex_matrix_json(matrix)
                    for matrix in compiled.matrices_requested
                ],
                "matrices_used": [
                    complex_matrix_json(matrix) for matrix in compiled.matrices_used
                ],
                "centers": [float(center) for center in compiled.centers],
                "reference_matrices": [
                    complex_matrix_json(reference)
                    for reference in compiled.reference_matrices
                ],
                "hermitian_residuals": {
                    name: {"matrix": matrix_residual, "reference_matrix": ref_residual}
                    for name, matrix_residual, ref_residual in zip(
                        compiled.names,
                        compiled.matrix_residuals,
                        compiled.reference_residuals,
                        strict=True,
                    )
                },
                "hermitian_tolerance": float(compiled.tolerance),
                "source_product": source_product,
            }
        )

        schema = ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=axes,
            # The valid_length variable and its handle are the source
            # product's own: the censoring record propagates untouched, on
            # its original device.
            variables=[value_variable, valid_variable],
            attributes=attributes,
        )
        device = alpha_handle.device
        value_handle: Any
        if device == "cpu":
            value_handle = HostArrayHandle(
                np.asarray(value), value_variable, owner=str(label)
            )
        else:
            value_handle = BackendArrayHandle(
                value, value_variable, owner=str(label), device=device
            )
        product = TimeSeriesDataset(
            schema,
            DictProductBacking(
                {
                    "value": value_handle,
                    "valid_length": signal.handle("valid_length"),
                }
            ),
            provenance={"source_analyser": str(label)},
        )
        return {label: product}
