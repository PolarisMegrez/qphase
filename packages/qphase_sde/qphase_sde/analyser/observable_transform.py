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
from .base import (
    Analyser,
    check_execution_cancelled,
    requirement_source,
    select_mode_columns,
)

__all__ = [
    "ObservableTransformAnalyser",
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
        allow_inf_nan=False,
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


def validate_mode_indices(raw_modes: Any, n_channel: int) -> tuple[int, ...]:
    """Validate the ``mode_indices`` product attribute against the channel axis.

    Returns the recorded mode per channel, defaulting to the identity
    ordering when the attribute is absent. Modes must be unique non-negative
    integers; booleans are rejected even though ``bool`` subclasses ``int``.
    """
    if raw_modes is None:
        return tuple(range(n_channel))
    if not isinstance(raw_modes, list | tuple) or not all(
        isinstance(mode, int | np.integer) and not isinstance(mode, bool | np.bool_)
        for mode in raw_modes
    ):
        raise TypeError("observable_transform mode_indices must be a list of integers")
    recorded = tuple(int(mode) for mode in raw_modes)
    if any(mode < 0 for mode in recorded):
        raise ValueError("observable_transform mode_indices must be non-negative")
    if len(set(recorded)) != len(recorded):
        raise ValueError("observable_transform mode_indices must be unique")
    if len(recorded) != n_channel:
        raise ValueError(
            "observable_transform mode_indices must match the channel "
            f"axis size {n_channel}, got {len(recorded)}"
        )
    return recorded


class ObservableTransformAnalyser(Analyser):
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
        """Estimate per-invocation workspace; the transform retains nothing.

        With no reducer there is no cross-batch state: batch-local output is
        released once the same batch's last consumer finishes, so only the
        current physical batch (``request.n_traj``) counts. The invocation
        peak covers the selected complex view, the staged matrices and
        centers, the complex contraction ``z = alpha^dagger Q alpha`` and the
        final real output.
        """
        config = cast(ObservableTransformConfig, self.config)
        n_observables = len(config.observables)
        n_modes = (
            len(config.modes) if config.modes is not None else request.n_record_modes
        )
        complex_itemsize = 2 * request.real_itemsize
        batch_samples = request.n_traj * request.saved_samples
        invocation = (
            batch_samples * n_modes * complex_itemsize
            + n_observables * (n_modes * n_modes + 1) * complex_itemsize
            + batch_samples * n_observables * complex_itemsize
            + batch_samples * n_observables * request.real_itemsize
        )
        if request.backend_name == "cupy":
            return AnalyserWorkspaceEstimate(device_bytes=invocation)
        return AnalyserWorkspaceEstimate(host_bytes=invocation)

    def input_requirements(self) -> list[ProductRequirement]:
        return [
            ProductRequirement(
                name="signal",
                kind=DataKind.TIME_SERIES,
                quantity=SDEQuantity.FIELD_AMPLITUDE.value,
                value_domain="complex",
                required_fields=["alpha", "valid_length"],
                accepted_axes=["scan", "trajectory", "time", "channel"],
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
        axis_by_name = {axis.name: axis for axis in signal.axes}
        has_scan = "scan" in axis_by_name
        alpha_dims = ("scan", "trajectory", "time", "channel") if has_scan else (
            "trajectory",
            "time",
            "channel",
        )
        valid_dims = ("scan", "trajectory") if has_scan else ("trajectory",)
        alpha_variable = variables["alpha"]
        if tuple(alpha_variable.dims) != alpha_dims:
            raise TypeError(
                f"observable_transform alpha dims must be {alpha_dims}, "
                f"got {tuple(alpha_variable.dims)}"
            )
        valid_variable = variables["valid_length"]
        if tuple(valid_variable.dims) != valid_dims:
            raise TypeError(
                f"observable_transform valid_length dims must be {valid_dims}, "
                f"got {tuple(valid_variable.dims)}"
            )
        if np.dtype(valid_variable.dtype) != np.dtype("int64"):
            raise TypeError(
                "observable_transform valid_length must have dtype int64, "
                f"got {valid_variable.dtype}"
            )
        if valid_variable.value_domain != "real":
            raise TypeError(
                "observable_transform valid_length must be real-valued, "
                f"got {valid_variable.value_domain}"
            )
        if valid_variable.quantity != "valid_length":
            raise TypeError(
                "observable_transform valid_length quantity must be "
                f"'valid_length', got {valid_variable.quantity}"
            )
        expected_roles = {
            "scan": AxisRole.PARAMETER,
            "trajectory": AxisRole.REALIZATION,
            "time": AxisRole.COORDINATE,
            "channel": AxisRole.COMPONENT,
        }
        for axis_name in alpha_dims:
            axis = axis_by_name.get(axis_name)
            if axis is None:
                raise TypeError(
                    f"observable_transform signal is missing the {axis_name} axis"
                )
            if axis.role is not expected_roles[axis_name]:
                raise TypeError(
                    f"observable_transform {axis_name} axis role must be "
                    f"{expected_roles[axis_name].value}, got {axis.role.value}"
                )
        time_axis = axis_by_name["time"]
        if time_axis.coordinate != "regular" or time_axis.units != "time":
            raise TypeError(
                "observable_transform requires the canonical regular time axis"
            )
        alpha_handle = signal.handle("alpha")
        alpha = alpha_handle.materialize()
        if not np.issubdtype(alpha.dtype, np.complexfloating):
            raise ValueError("observable_transform expects a complex alpha variable")
        if alpha.ndim != len(alpha_dims):
            raise ValueError(
                f"observable_transform alpha must have shape {alpha_dims}, "
                f"got {alpha.shape}"
            )
        for position, axis_name in enumerate(alpha_dims):
            if axis_by_name[axis_name].size != int(alpha.shape[position]):
                raise ValueError(
                    f"observable_transform {axis_name} axis size "
                    f"{axis_by_name[axis_name].size} does not match alpha shape "
                    f"{alpha.shape}"
                )
        n_channel = int(alpha.shape[-1])
        recorded = validate_mode_indices(
            signal.attributes.get("mode_indices"), n_channel
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
        )
        check_execution_cancelled(context)

        axes: list[AxisSchema] = []
        dims_prefix: tuple[str, ...] = ()
        if has_scan:
            axes.append(axis_by_name["scan"])
            dims_prefix = ("scan",)
        axes.append(axis_by_name["trajectory"])
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
        source_product = requirement_source(metadata, "signal") or ""
        attributes = dict(SAMPLED_OBSERVABLE_PRODUCT.attributes)
        attributes.update(
            {
                "producer_analyser": str(label),
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
            provenance={
                "producer_analyser": str(label),
                "source_product": source_product,
            },
        )
        return {label: product}
