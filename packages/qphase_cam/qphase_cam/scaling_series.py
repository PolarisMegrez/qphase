"""qphase_cam: a scaling series and the response it is built from
------------------------------------------------------------------------
A scaling study resolves how a readout moves as a control approaches a critical point.
Two things made that awkward with only the scalar observable contract:

* a CAM matrix element is complex, and what scales is its *displacement*
  ``|R_ij − R_ij(critical)|`` or a norm ``‖δR‖_F`` — a positive real number that is
  already formed, not a complex value cast to float;
* the abscissa of a paper figure is the epsilon the run was configured with, an exact
  ``abs_epsilon``, which must not be recovered by the floating-point subtraction
  ``abs(parameter − critical)``.

This module names both. :class:`ScalingResponse` is the readout contract — an identity
plus a positive real response — and :class:`FunctionalScalingResponse` is the plain
closure implementation, so a complex displacement or a Frobenius norm is a response like
any other. :class:`ScalingSeries` is one pre-registered series: the physical parameter
values, the exact ``x``, the response and the convergence flag of every point.
:func:`evaluate_scaling_series` fits it with the shared
:mod:`qphase_cam.scaling_analysis` primitives and builds the same typed annotation
:class:`~qphase_cam.postprocessor.observable_scaling.ObservableScaling` produces,
through that plugin's own builder — there is no second fit and no second product format.

The reference value is what a report quotes and the anchor is what the fit uses; they
are separate fields and the method metadata names each. A response handed over here is
already a magnitude, so it is never shifted again, which is the
``absolute_shift=False`` convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.data import Dataset

from .observable import ObservableIdentity
from .postprocessor.observable_scaling import (
    ObservableScaling,
    ObservableScalingConfig,
)
from .scaling_analysis import series_identity

__all__ = [
    "FunctionalScalingResponse",
    "ScalingResponse",
    "ScalingSeries",
    "evaluate_scaling_series",
    "scaling_series",
]


@runtime_checkable
class ScalingResponse(Protocol):
    """A positive real response that scales with a control parameter.

    ``identity`` states what is measured (the same :class:`ObservableIdentity` an
    observable uses), and ``response`` returns the *already formed* magnitude at one
    state — ``|R_ij − R_ij(critical)|``, ``‖δR‖_F``, a variance, an integrated
    intensity. A complex quantity is never returned here: forming the magnitude is the
    response's job, so no caller has to guess whether a real part is meant.
    """

    identity: ObservableIdentity

    def response(
        self, adapter: Any, state: Any, params: Mapping[str, Any]
    ) -> float: ...


@dataclass(frozen=True)
class FunctionalScalingResponse:
    """A caller-supplied response, so a new measured quantity needs no new class.

    ``mapping`` states the measurement in readable text (for example
    ``"|Re R_ab - Re R_ab(critical)|"``), and ``metadata`` carries whatever provenance
    the caller wants recorded beside the identity.
    """

    identity: ObservableIdentity
    response_fn: Callable[[Any, Any, Mapping[str, Any]], float]
    mapping: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def response(self, adapter: Any, state: Any, params: Mapping[str, Any]) -> float:
        """Return the caller's response at one state."""
        return float(self.response_fn(adapter, state, params))

    def method(self) -> dict[str, Any]:
        """Return the provenance this response declares."""
        return {
            "identity": self.identity.model_dump(mode="json"),
            "mapping": self.mapping or self.identity.measurement_mapping,
            **dict(self.metadata),
        }


class ScalingSeries(BaseModel):
    """One pre-registered scaling series.

    ``parameter`` names the physical control (e.g. ``omega_b``), ``parameter_values``
    its value at every point and ``x`` the exact ``abs_epsilon`` the fit uses — the
    configured epsilon, not ``abs(parameter − critical_parameter)``. ``response`` is
    the positive magnitude of each point and ``converged`` its convergence flag; a
    series with a point that did not converge is refused rather than fitted
    partially, because a filled
    series is not the same thing as a complete one.

    ``reference_value`` is what a report quotes at the critical point and
    ``critical_parameter`` the critical control value itself; neither shifts
    ``response``, which is already a magnitude.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter: str = Field(min_length=1, description="Physical control name")
    parameter_values: tuple[float, ...]
    x: tuple[float, ...]
    response: tuple[float, ...]
    converged: tuple[bool, ...]
    critical_parameter: float
    reference_value: float | None = None
    source: str = ""
    method: dict[str, Any] = Field(default_factory=dict)

    def validated(self) -> ScalingSeries:
        """Return a validated copy, or raise.

        Raises
        ------
        ValueError
            If the arrays differ in length, if ``x`` is not finite, strictly positive
            and strictly increasing, if a response is not finite and strictly positive,
            or if any point did not converge.

        """
        size = len(self.parameter_values)
        for name in ("x", "response", "converged"):
            if len(getattr(self, name)) != size:
                raise ValueError(
                    f"the series has {len(getattr(self, name))} {name} entries for "
                    f"{size} parameter values"
                )
        if size == 0:
            raise ValueError("a scaling series needs at least one point")
        x = np.asarray(self.x, dtype=float)
        response = np.asarray(self.response, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x <= 0.0):
            raise ValueError("the series' x must be finite and strictly positive")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("the series' x must be strictly increasing")
        if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
            raise ValueError(
                "the series' response must be finite and strictly positive"
            )
        if not all(bool(value) for value in self.converged):
            raise ValueError(
                "the series has a point that did not converge; a scaling fit needs a "
                "complete series"
            )
        return self

    def identity(self, response: ScalingResponse) -> str:
        """Return the content identity of this series under one response.

        The identity covers the response's own identity, this series' physical
        control name and every ``(x, response)`` pair, so a product's
        ``target_identity`` and its recorded ``series_identity`` are the same string
        by construction.
        """
        self.validated()
        return series_identity(
            response,
            parameter=str(self.parameter),
            x=np.asarray(self.x, dtype=float),
            y=np.asarray(self.response, dtype=float),
        )


def scaling_series(
    *,
    parameter: str,
    critical_parameter: float,
    points: Sequence[Mapping[str, Any]],
    reference_value: float | None = None,
    source: str = "",
    method: Mapping[str, Any] | None = None,
) -> ScalingSeries:
    """Return a scaling series assembled from measured points.

    ``parameter`` names the physical control the points were measured along. Each
    point states that control's value (``parameter``), the exact configured
    ``abs_epsilon`` (``x``), the positive ``response`` and its own ``converged``
    flag — a point that does not state its convergence is an error, never a silent
    success. Nothing is recomputed: the caller's epsilon is the abscissa, and the
    response is not shifted against the critical value again.

    Raises
    ------
    ValueError
        If a point omits ``converged``, or if the assembled series fails
        :meth:`ScalingSeries.validated`.

    """
    converged: list[bool] = []
    for position, point in enumerate(points):
        if "converged" not in point:
            raise ValueError(
                f"scaling point {position} does not state whether it converged; a "
                "series is never filled in as if it had"
            )
        converged.append(bool(point["converged"]))
    return ScalingSeries(
        parameter=str(parameter),
        parameter_values=tuple(float(point["parameter"]) for point in points),
        x=tuple(float(point["x"]) for point in points),
        response=tuple(float(point["response"]) for point in points),
        converged=tuple(converged),
        critical_parameter=float(critical_parameter),
        reference_value=None if reference_value is None else float(reference_value),
        source=str(source),
        method=dict(method or {}),
    ).validated()


@dataclass(frozen=True)
class _IdentityOnly:
    """The identity field the shared builder reads from its readout argument."""

    identity: ObservableIdentity


def evaluate_scaling_series(
    response: ScalingResponse,
    series: ScalingSeries,
    *,
    config: ObservableScalingConfig | None = None,
) -> Dataset:
    """Fit one pre-registered series and return its typed scaling annotation.

    The fit is the one
    :class:`~qphase_cam.postprocessor.observable_scaling.ObservableScaling` runs — its
    own builder is called, so the free-exponent fit, the local exponents, the tolerance
    prefixes and the window fits are the shared implementation, not a copy. The
    response's identity and mapping travel into the product's method metadata, and the
    series' reference value is recorded as a report value only.

    The series' physical control name is authoritative: the default config takes it from
    the series, and a config that names another control is refused, so the product's
    ``target_identity`` and its recorded ``series_identity`` cannot disagree.
    """
    resolved = series.validated()
    identity = getattr(response, "identity", None)
    if not isinstance(identity, ObservableIdentity):
        raise TypeError(
            "a scaling response needs an ObservableIdentity; got "
            f"{type(identity).__name__}"
        )
    settings = config or ObservableScalingConfig(
        parameter=resolved.parameter,
        critical_parameter=float(resolved.critical_parameter),
        critical_value=resolved.reference_value,
        absolute_shift=False,
    )
    if str(settings.parameter) != str(resolved.parameter):
        raise ValueError(
            f"the config names the control {settings.parameter!r} but the series was "
            f"measured along {resolved.parameter!r}; one series has one identity"
        )
    if settings.absolute_shift:
        raise ValueError(
            "a pre-registered scaling series is already a magnitude; its config must "
            "set absolute_shift=False"
        )
    method: dict[str, Any] = {
        **dict(resolved.method),
        "source": resolved.source,
        "critical_parameter": float(resolved.critical_parameter),
        "reference_value": resolved.reference_value,
        "reference_value_role": "reported at the critical point, never subtracted",
        "x": "pre-registered abs_epsilon (not a parameter difference)",
        "response": "already formed positive magnitude; not shifted",
        "series_identity": resolved.identity(response),
    }
    if hasattr(response, "method"):
        method["response_contract"] = response.method()
    return ObservableScaling(settings).build_annotation(
        {
            "parameter_values": np.asarray(resolved.parameter_values, dtype=float),
            "values": np.asarray(resolved.response, dtype=float),
            "x": np.asarray(resolved.x, dtype=float),
            "response": np.asarray(resolved.response, dtype=float),
            "converged": list(resolved.converged),
        },
        _IdentityOnly(identity),
        method_extra=method,
    )
