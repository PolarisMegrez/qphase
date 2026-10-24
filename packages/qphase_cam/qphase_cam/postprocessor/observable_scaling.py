"""Readout-resolved scaling annotation for a scanned CAM observable.

One annotation row per product: the series is the observable of one readout along
one control coordinate, so the row is a *scan case*, not a solution slot. The
series is read from the product — a ``state_scan`` or a candidate scan — and the
observable is evaluated on the stored states with the model's Hamiltonian; nothing
is re-solved.

The numbers come from :mod:`qphase_cam.scaling_analysis` (log-space fits, tolerance
prefixes, per-point exponents, outer extrapolation) and from the 10D3 Puiseux layer
(:func:`~qphase_cam.solver.scaling.free_exponent_fit`). This plugin adds no fitting
code of its own, no model name, no mode index and no default target exponent.

Negative results are statuses: a series whose solutions did not converge is
``not_converged``, a series too short for the requested order (or with a
non-positive response) is ``unidentifiable``, and neither produces a fabricated
fit. Structural problems — a missing control coordinate, two sides of one
perturbation, more than one solution per scan point, more than one candidate per
case — raise, because no annotation of that shape is meaningful.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.data import build_annotation_product
from qphase_cam.observable import CAMObservable, RayleighObservable
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.scaling_analysis import (
    fixed_exponent_tolerance_upper,
    local_log_exponents,
    puiseux_outer_extrapolation,
    series_identity,
    tolerance_prefix_upper,
    window_log_scaling_fits,
)
from qphase_cam.solver.scaling import (
    PRIMARY_EXPONENT_BOUNDS,
    ScalingFit,
    free_exponent_fit,
)

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .candidates import candidate_view
from .state_view import state_rows


class ObservableScalingConfig(CAMPostprocessorConfig):
    """Which series to build, how to shift it and which fit to run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    parameter: str = Field(
        min_length=1,
        description="Control coordinate the readout is resolved against",
    )
    critical_parameter: float = Field(
        default=0.0,
        description="Parameter value at which the scaling coordinate x vanishes",
    )
    critical_value: float | None = Field(
        default=None,
        description="Readout at x=0; required unless a scan point sits at the "
        "critical parameter or absolute_shift is False",
    )
    absolute_shift: bool = Field(
        default=True,
        description="Use |value - critical_value| as the response (the reference "
        "script's convention); with False the raw value is fitted and no anchor "
        "is needed",
    )
    order: int = Field(default=2, ge=1, description="Puiseux order of the free fit")
    bounds: tuple[float, float] = Field(
        default=PRIMARY_EXPONENT_BOUNDS,
        description="Exponent window of the free fit",
    )
    weighted: bool = False
    calculate_loo: bool = Field(
        default=True, description="Evaluate leave-one-out on the free fit"
    )
    leading_exponent: float | None = Field(
        default=None,
        gt=0.0,
        description="Exponent of the leading law, if the caller knows it",
    )
    leading_coefficient: float | None = Field(
        default=None,
        description="Leading amplitude, if the caller knows it; otherwise the "
        "first-point ratio is reported as such",
    )
    relative_tolerance: float = Field(
        default=0.05,
        gt=0.0,
        description="Relative tolerance of the leading-law prefix search",
    )
    minimum_points: int = Field(
        default=5,
        ge=2,
        description="Smallest prefix the fixed-exponent window may use",
    )
    window_maxima: tuple[float, ...] = Field(
        default=(),
        description="Windows of the per-window log-space fits",
    )
    inner_max: float | None = Field(
        default=None,
        gt=0.0,
        description="Inner window upper bound of the outer-extrapolation check",
    )
    observable: CAMObservable | None = Field(
        default=None,
        description="Readout to resolve; defaults to the Rayleigh frequency",
    )


#: Linear-fit fields of the annotation and their dtype.
_FIT_FIELDS: tuple[str, ...] = (
    "exponent",
    "exponent_sem",
    "rmse",
    "max_abs_residual",
    "aicc",
    "condition",
    "loo_rmse",
    "objective",
    "epsilon_scale",
    "profile_points",
)

#: Leading-law fields of the annotation.
_LEADING_FIELDS: tuple[str, ...] = (
    "leading_coefficient",
    "leading_relative_rmse",
    "leading_relative_max",
    "leading_prefix_upper",
    "fixed_exponent_prefix_upper",
)

#: Outer-extrapolation fields of the annotation.
_OUTER_FIELDS: tuple[str, ...] = (
    "inner_max",
    "outer_exponent",
    "outer_relative_rmse",
    "outer_max_abs_relative_error",
)


class ObservableScaling(CAMPostprocessor[ObservableScalingConfig]):
    """Resolve one readout's scaling against one control coordinate."""

    name: ClassVar[str] = "observable_scaling"
    description: ClassVar[str] = (
        "Readout-resolved scaling: free Puiseux fit, leading law and windows"
    )
    config_schema: ClassVar[type[ObservableScalingConfig]] = ObservableScalingConfig

    annotation_kind: ClassVar[str] = "observable_scaling"
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("state_scan", "candidates")
    requires_model: ClassVar[bool] = True
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        model = context.model
        adapter = QPSGenDynamicsAdapter.from_model(model)
        observable: Any = (
            self.config.observable
            if self.config.observable is not None
            else RayleighObservable(model)
        )
        if not isinstance(observable, CAMObservable):
            raise TypeError(
                "observable_scaling needs a CAMObservable readout, got "
                f"{type(observable).__name__}"
            )
        role = self._target_role(product, context)
        series = (
            self._state_scan_series(product, adapter, observable, model)
            if role == "state_scan"
            else self._candidate_series(product, context, adapter, observable, model)
        )
        return self._annotation(series, observable)

    # -- series -----------------------------------------------------------

    def _state_scan_series(
        self,
        product: Any,
        adapter: QPSGenDynamicsAdapter,
        observable: Any,
        model: Any,
    ) -> dict[str, Any]:
        parameters = [
            variable.name.removeprefix("parameter.")
            for variable in product.variables
            if variable.name.startswith("parameter.")
        ]
        if self.config.parameter not in parameters:
            raise ValueError(
                f"the state_scan product has no {self.config.parameter!r} "
                f"coordinate; it carries {sorted(parameters)}"
            )
        if len(parameters) > 1:
            raise ValueError(
                "a scaling series needs one scan coordinate, but the product "
                f"carries {sorted(parameters)}"
            )
        rows = state_rows(product)
        grouped: dict[int, list[Any]] = {}
        for row in rows:
            grouped.setdefault(row.scan_index, []).append(row)
        multiple = sorted(index for index, block in grouped.items() if len(block) > 1)
        if multiple:
            raise ValueError(
                "a scaling series needs one filled solution per scan point, but "
                f"scan points {multiple} hold several"
            )
        order = sorted(grouped)
        values: list[float] = []
        parameter_values: list[float] = []
        states: list[np.ndarray] = []
        converged: list[bool] = []
        for index in order:
            row = grouped[index][0]
            parameter_map = {**dict(model.params), **dict(row.parameters)}
            if self.config.parameter not in parameter_map:
                raise ValueError(
                    f"scan point {index} has no {self.config.parameter!r} value"
                )
            vector = np.asarray(matrix_to_vector(row.state), dtype=float)
            parameter_values.append(float(parameter_map[self.config.parameter]))
            values.append(float(observable.value(adapter, vector, parameter_map)))
            states.append(vector)
            converged.append(bool(row.success))
        return {
            "parameter_values": np.asarray(parameter_values, dtype=float),
            "values": np.asarray(values, dtype=float),
            "converged": converged,
            "states": states,
            "role": "state_scan",
        }

    def _candidate_series(
        self,
        product: Any,
        context: CAMAnnotationContext,
        adapter: QPSGenDynamicsAdapter,
        observable: Any,
        model: Any,
    ) -> dict[str, Any]:
        view = candidate_view(product, metadata=dict(context.metadata))
        counts: dict[int, int] = {}
        for row in view.rows:
            counts[row.case] = counts.get(row.case, 0) + 1
        multiple = sorted(case for case, count in counts.items() if count > 1)
        if multiple:
            raise ValueError(
                "a candidate scaling series needs one candidate per case, but "
                f"cases {multiple} hold several"
            )
        order = sorted(view.rows, key=lambda item: item.case)
        values: list[float] = []
        parameter_values: list[float] = []
        states: list[np.ndarray] = []
        converged: list[bool] = []
        for row in order:
            parameter_map = dict(model.params)
            parameter_map.update(view.meta.get("fixed_params", {}))
            for name, column in view.case_params.items():
                parameter_map[name] = float(np.asarray(column).reshape(-1)[row.case])
            parameter_map.update(row.controls)
            if self.config.parameter not in parameter_map:
                raise ValueError(
                    f"case {row.case} has no {self.config.parameter!r} value; it "
                    f"carries {sorted(parameter_map)}"
                )
            vector = np.asarray(row.state_vector, dtype=float)
            parameter_values.append(float(parameter_map[self.config.parameter]))
            values.append(float(observable.value(adapter, vector, parameter_map)))
            states.append(vector)
            converged.append(bool(row.success))
        return {
            "parameter_values": np.asarray(parameter_values, dtype=float),
            "values": np.asarray(values, dtype=float),
            "converged": converged,
            "states": states,
            "role": "candidates",
        }

    @staticmethod
    def _target_role(product: Any, context: CAMAnnotationContext) -> str:
        role = str(context.source_role or "")
        if not role:
            role = str(getattr(product, "attributes", {}).get("product", ""))
        if role in ("state_scan", "candidates"):
            return role
        raise ValueError(
            f"observable_scaling annotates state_scan or candidates; got {role!r}"
        )

    # -- annotation -------------------------------------------------------

    def _annotation(self, series: dict[str, Any], observable: Any) -> Any:
        parameters = np.asarray(series["parameter_values"], dtype=float)
        values = np.asarray(series["values"], dtype=float)
        converged = list(series["converged"])
        method: dict[str, Any] = {
            "service": "qphase_cam.scaling_analysis",
            "fit": "free_exponent_fit",
            "log_fit": "log y on log x (script _free_power_fit)",
            "observable": observable.identity.model_dump(mode="json"),
            "parameter": self.config.parameter,
            "critical_parameter": float(self.config.critical_parameter),
            "absolute_shift": bool(self.config.absolute_shift),
            "order": int(self.config.order),
            "bounds": list(self.config.bounds),
            "weighted": bool(self.config.weighted),
            "window_maxima": list(self.config.window_maxima),
            "inner_max": self.config.inner_max,
        }
        fields: dict[str, Any] = {
            "reason": np.asarray([""], dtype=str),
            "parameter": np.asarray([self.config.parameter], dtype=str),
            "observable_name": np.asarray([observable.identity.name], dtype=str),
            "observable_kind": np.asarray([observable.identity.kind.value], dtype=str),
            "charge_sector": np.asarray([observable.identity.sector.value], dtype=str),
            "critical_value_source": np.asarray([""], dtype=str),
            "leading_coefficient_source": np.asarray([""], dtype=str),
            "leading_law_status": np.asarray([""], dtype=str),
        }
        for name in (*_FIT_FIELDS, *_LEADING_FIELDS, *_OUTER_FIELDS):
            fields[name] = np.asarray([np.nan], dtype=float)
        fields["critical_parameter"] = np.asarray(
            [float(self.config.critical_parameter)], dtype=float
        )
        fields["critical_value"] = np.asarray([np.nan], dtype=float)
        for name in (
            "point_count",
            "dropped_point_count",
            "parameter_count",
            "inner_point_count",
            "outer_point_count",
        ):
            fields[name] = np.asarray([0], dtype=np.int64)
        fields["at_exponent_bound"] = np.asarray([-1], dtype=np.int8)
        fields["absolute_shift"] = np.asarray(
            [int(bool(self.config.absolute_shift))], dtype=np.int8
        )
        width = int(values.size)
        for name in (
            "point_parameter",
            "point_x",
            "point_response",
            "local_log_exponent",
        ):
            fields[name] = np.full((1, width), np.nan, dtype=float)
        fields["coefficients"] = np.full((1, int(self.config.order)), np.nan)
        window_count = len(self.config.window_maxima)
        fields["window_maximum"] = np.asarray(
            [list(self.config.window_maxima)], dtype=float
        ).reshape(1, window_count)
        for name in ("window_exponent", "window_log_rmse"):
            fields[name] = np.full((1, window_count), np.nan, dtype=float)
        fields["window_point_count"] = np.zeros((1, window_count), dtype=np.int64)

        def fail(status: CAMOutcomeStatus, reason: str) -> Any:
            fields["reason"] = np.asarray([reason], dtype=str)
            return self._product(
                fields,
                method,
                observable,
                status=status,
                point_count=int(values.size),
                window_count=window_count,
            )

        if not converged or not all(converged):
            return fail(
                CAMOutcomeStatus.NOT_CONVERGED,
                "at least one point of the series did not converge, so the "
                "series is incomplete",
            )
        x = np.abs(parameters - float(self.config.critical_parameter))
        if np.any(np.diff(x[np.argsort(x)]) == 0.0):
            raise ValueError(
                "the series has duplicate |parameter - critical_parameter| values "
                "(for example both sides of a symmetric perturbation); a scaling "
                "series needs one side"
            )
        anchor, anchor_source = self._critical_value(parameters, values, x)
        keep = np.isfinite(x) & np.isfinite(values) & (x > 0.0)
        dropped = int(np.count_nonzero(~keep))
        x = x[keep]
        observed = values[keep]
        if x.size == 0:
            raise ValueError(
                "no scan point has a finite, non-zero |parameter - "
                "critical_parameter|; there is no scaling series to fit"
            )
        order = np.argsort(x)
        x = x[order]
        observed = observed[order]
        response = np.abs(observed - anchor) if anchor is not None else observed
        fields["point_parameter"] = parameters[keep][order].reshape(1, -1)
        fields["point_x"] = x.reshape(1, -1)
        fields["point_response"] = response.reshape(1, -1)
        fields["critical_value"] = np.asarray(
            [np.nan if anchor is None else float(anchor)], dtype=float
        )
        fields["critical_value_source"] = np.asarray([anchor_source], dtype=str)
        fields["dropped_point_count"] = np.asarray([dropped], dtype=np.int64)
        fields["point_count"] = np.asarray([int(x.size)], dtype=np.int64)
        method["critical_value_source"] = anchor_source
        if np.any(response <= 0.0):
            return fail(
                CAMOutcomeStatus.UNIDENTIFIABLE,
                "the response is not strictly positive everywhere; a log-space "
                "scaling law is undefined there",
            )
        needed = int(self.config.order) + 3
        if x.size < needed:
            return fail(
                CAMOutcomeStatus.UNIDENTIFIABLE,
                f"{x.size} usable points for order {order} "
                f"(needs {needed}); no fit is attempted",
            )
        try:
            fit = free_exponent_fit(
                x,
                response,
                order=int(self.config.order),
                bounds=(float(self.config.bounds[0]), float(self.config.bounds[1])),
                weighted=bool(self.config.weighted),
                calculate_loo=bool(self.config.calculate_loo),
            )
        except ValueError as exc:
            return fail(CAMOutcomeStatus.UNIDENTIFIABLE, f"the fit failed: {exc}")
        self._fill_fit(fields, fit, order=int(self.config.order))
        if x.size >= 3:
            fields["local_log_exponent"] = local_log_exponents(x, response).reshape(
                1, -1
            )
        self._fill_leading(fields, x, response, method)
        self._fill_windows(fields, x, response, method)
        self._fill_outer(fields, x, response, method)
        method["point_count"] = int(x.size)
        method["dropped_point_count"] = dropped
        return self._product(
            fields,
            method,
            observable,
            status=CAMOutcomeStatus.VALID,
            point_count=int(x.size),
            window_count=window_count,
        )

    def _critical_value(
        self, parameters: np.ndarray, values: np.ndarray, x: np.ndarray
    ) -> tuple[float | None, str]:
        if self.config.critical_value is not None:
            return float(self.config.critical_value), "config"
        if not self.config.absolute_shift:
            return None, "not_used"
        at_critical = np.flatnonzero(x == 0.0)
        if at_critical.size:
            return float(values[int(at_critical[0])]), "zero_point"
        raise ValueError(
            "critical_value is required: no point sits at "
            f"parameter={self.config.critical_parameter} and no value was given"
        )

    @staticmethod
    def _fill_fit(fields: dict[str, Any], fit: ScalingFit, *, order: int) -> None:
        fields["exponent"] = np.asarray([float(fit.exponent)], dtype=float)
        fields["exponent_sem"] = np.asarray(
            [np.nan if fit.exponent_sem is None else float(fit.exponent_sem)],
            dtype=float,
        )
        fields["rmse"] = np.asarray([float(fit.rmse)], dtype=float)
        fields["max_abs_residual"] = np.asarray(
            [float(fit.max_abs_residual)], dtype=float
        )
        fields["aicc"] = np.asarray([float(fit.aicc)], dtype=float)
        fields["condition"] = np.asarray([float(fit.condition)], dtype=float)
        fields["loo_rmse"] = np.asarray(
            [np.nan if fit.loo_rmse is None else float(fit.loo_rmse)], dtype=float
        )
        fields["objective"] = np.asarray([float(fit.objective)], dtype=float)
        fields["epsilon_scale"] = np.asarray([float(fit.epsilon_scale)], dtype=float)
        fields["profile_points"] = np.asarray(
            [0 if fit.profile_points is None else int(fit.profile_points)],
            dtype=np.int64,
        )
        fields["parameter_count"] = np.asarray(
            [int(fit.parameter_count)], dtype=np.int64
        )
        fields["at_exponent_bound"] = np.asarray(
            [int(bool(fit.at_exponent_bound))], dtype=np.int8
        )
        coefficients = np.full(order, np.nan, dtype=float)
        coefficients[: len(fit.coefficients)] = np.asarray(
            fit.coefficients, dtype=float
        )
        fields["coefficients"] = coefficients.reshape(1, order)

    def _fill_leading(
        self,
        fields: dict[str, Any],
        x: np.ndarray,
        response: np.ndarray,
        method: dict[str, Any],
    ) -> None:
        exponent = self.config.leading_exponent
        if exponent is None:
            fields["leading_law_status"] = np.asarray(
                ["leading_exponent_not_given"], dtype=str
            )
            method["leading_law"] = "not evaluated"
            return
        resolved = float(exponent)
        if self.config.leading_coefficient is not None:
            coefficient = float(self.config.leading_coefficient)
            source = "config"
        else:
            coefficient = float(response[0] / x[0] ** resolved)
            source = "first_point_ratio"
        prediction = coefficient * x**resolved
        relative = np.abs(prediction / response - 1.0)
        fields["leading_coefficient"] = np.asarray([coefficient], dtype=float)
        fields["leading_coefficient_source"] = np.asarray([source], dtype=str)
        fields["leading_relative_rmse"] = np.asarray(
            [float(np.sqrt(np.mean(relative**2)))], dtype=float
        )
        fields["leading_relative_max"] = np.asarray(
            [float(np.max(relative))], dtype=float
        )
        prefix = tolerance_prefix_upper(x, relative <= self.config.relative_tolerance)
        fields["leading_prefix_upper"] = np.asarray(
            [np.nan if prefix is None else prefix], dtype=float
        )
        fixed = fixed_exponent_tolerance_upper(
            x,
            response,
            resolved,
            self.config.relative_tolerance,
            minimum_points=int(self.config.minimum_points),
        )
        fields["fixed_exponent_prefix_upper"] = np.asarray(
            [np.nan if fixed is None else fixed], dtype=float
        )
        fields["leading_law_status"] = np.asarray(["evaluated"], dtype=str)
        method.setdefault("leading_law", {})
        method["leading_law"] = {
            "exponent": resolved,
            "coefficient_source": source,
            "relative_tolerance": float(self.config.relative_tolerance),
            "prefix_upper": prefix,
            "fixed_exponent_prefix_upper": fixed,
        }

    def _fill_windows(
        self,
        fields: dict[str, Any],
        x: np.ndarray,
        response: np.ndarray,
        method: dict[str, Any],
    ) -> None:
        if not self.config.window_maxima:
            method["windows"] = []
            return
        fits = window_log_scaling_fits(x, response, self.config.window_maxima)
        fields["window_exponent"] = np.asarray(
            [[float(fit.exponent) for fit in fits]], dtype=float
        )
        fields["window_log_rmse"] = np.asarray(
            [[float(fit.log_rmse) for fit in fits]], dtype=float
        )
        fields["window_point_count"] = np.asarray(
            [[int(fit.point_count) for fit in fits]], dtype=np.int64
        )
        method["windows"] = [
            {
                "upper": fit.upper,
                "exponent": float(fit.exponent),
                "log_rmse": float(fit.log_rmse),
                "point_count": int(fit.point_count),
            }
            for fit in fits
        ]

    def _fill_outer(
        self,
        fields: dict[str, Any],
        x: np.ndarray,
        response: np.ndarray,
        method: dict[str, Any],
    ) -> None:
        if self.config.inner_max is None:
            method["outer_extrapolation"] = "not evaluated"
            return
        try:
            outer = puiseux_outer_extrapolation(
                x,
                response,
                inner_max=float(self.config.inner_max),
                order=int(self.config.order),
                bounds=(float(self.config.bounds[0]), float(self.config.bounds[1])),
                weighted=bool(self.config.weighted),
            )
        except ValueError as exc:
            fields["inner_max"] = np.asarray(
                [float(self.config.inner_max)], dtype=float
            )
            method["outer_extrapolation"] = f"unavailable: {exc}"
            return
        fields["inner_max"] = np.asarray([outer.inner_max], dtype=float)
        fields["outer_exponent"] = np.asarray([outer.exponent], dtype=float)
        fields["outer_relative_rmse"] = np.asarray([outer.relative_rmse], dtype=float)
        fields["outer_max_abs_relative_error"] = np.asarray(
            [outer.max_abs_relative_error], dtype=float
        )
        fields["inner_point_count"] = np.asarray(
            [outer.inner_point_count], dtype=np.int64
        )
        fields["outer_point_count"] = np.asarray(
            [outer.outer_point_count], dtype=np.int64
        )
        method["outer_extrapolation"] = outer.model_dump(mode="json")

    def _product(
        self,
        fields: dict[str, Any],
        method: dict[str, Any],
        observable: Any,
        *,
        status: CAMOutcomeStatus,
        point_count: int,
        window_count: int,
    ) -> Any:
        self.result_metadata = dict(method)
        self.result_metadata["status"] = status.value
        identity = series_identity(
            observable,
            parameter=str(self.config.parameter),
            x=fields["point_x"].reshape(-1),
            y=fields["point_response"].reshape(-1),
        )
        return build_annotation_product(
            annotation_kind=self.annotation_kind,
            target_kind="case",
            target_identity=[identity],
            status=[status.value],
            fields=fields,
            field_axes={
                "point_parameter": ("point",),
                "point_x": ("point",),
                "point_response": ("point",),
                "local_log_exponent": ("point",),
                "coefficients": ("coefficient",),
                "window_maximum": ("window",),
                "window_exponent": ("window",),
                "window_log_rmse": ("window",),
                "window_point_count": ("window",),
            },
            axes={
                "point": int(point_count),
                "coefficient": int(self.config.order),
                "window": int(window_count),
            },
            method_metadata=[dict(self.result_metadata)],
        )
