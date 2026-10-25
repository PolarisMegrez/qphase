"""Series-level soft-mode tracking and Petermann scaling over a spectrum annotation.

This postprocessor reads a ``petermann_spectrum`` annotation product — it needs no
model and no backend, because the eigenproblems were already solved — and turns it
into one series: the tracked soft mode of each family (Hamiltonian, Bogoliubov,
monodromy and the CAM Jacobian) with its Petermann factor, plus an optional
log-space scaling fit of those factors against the control distance.

The series tracks **two kinds of mode**, and the difference is deliberate:

* the Hamiltonian, Bogoliubov and monodromy families are tracked as the *soft mode*
  of a spectrum whose eigenvalues are decay rates. The seed is the positive-real-part
  eigenvalue with the smallest magnitude (falling back to all eigenvalues), every
  later row takes the eigenvalue closest to the previous one, and a tracked mode
  whose real part is not positive is reported as ``unidentifiable`` — a damped mode is
  not a soft mode;
* the CAM Jacobian is the *critical mode* of a linear stability problem, the same
  quantity the historical scan reported: every row selects the eigenvalue with the
  largest real part, and a finite one is valid even when its real part is negative.
  A stable critical mode still has a Petermann factor ``left @ left``, so a stable
  branch must not be discarded as unidentifiable.

A family that cannot be tracked keeps a status and a reason instead of a substituted
index.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from qphase_cam.data import build_annotation_product
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.scaling_analysis import log_linear_scaling_fit

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig

__all__ = ["PetermannSeries", "PetermannSeriesConfig"]

#: Families the series tracks, in output order.
FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("hamiltonian", "hamiltonian_eigenvalues", "hamiltonian_petermann_factors"),
    ("bogoliubov", "bogoliubov_eigenvalues", "bogoliubov_petermann_factors"),
    ("monodromy", "monodromy_eigenvalues", "monodromy_petermann_factors"),
    ("cam_jacobian", "cam_jacobian_eigenvalues", "cam_jacobian_petermann_factors"),
)

#: Tracking rule per family. ``soft`` follows the historical soft tracker;
#: ``critical`` takes the slowest mode of each row, which is what the historical
#: scan did for the CAM Jacobian.
TRACKING_RULES: dict[str, str] = {"cam_jacobian": "critical"}

#: Rule of every family not named in :data:`TRACKING_RULES`.
DEFAULT_TRACKING_RULE = "soft"


class PetermannSeriesConfig(CAMPostprocessorConfig):
    """Optional parameter axis and fit window; tracking alone needs neither."""

    parameter: str | None = Field(
        default=None,
        description="Control coordinate used to order the series and to fit",
    )
    critical_value: float | None = Field(
        default=None,
        description="Parameter value where the scaling coordinate vanishes",
    )
    inner_max: float | None = Field(
        default=None,
        gt=0.0,
        description="Largest |parameter - critical_value| included in the fit",
    )


def track_soft_index(values: np.ndarray, previous: complex | None) -> int:
    """Return the soft-mode index of one row.

    The seed is the positive-real-part eigenvalue with the smallest magnitude; if
    every eigenvalue is damped the whole spectrum is considered, exactly as the
    reference scan does. Later rows take the eigenvalue nearest the previous one.
    """
    spectrum = np.asarray(values, dtype=complex).reshape(-1)
    if spectrum.size == 0:
        raise ValueError("an empty spectrum has no soft mode")
    if previous is None:
        candidates = np.flatnonzero(np.real(spectrum) > 0.0)
        pool = candidates if candidates.size else np.arange(spectrum.size)
        return int(pool[int(np.argmin(np.abs(spectrum[pool])))])
    return int(np.argmin(np.abs(spectrum - previous)))


def track_critical_index(values: np.ndarray) -> int:
    """Return the critical (slowest) mode index of one row.

    The critical mode of a linear stability problem is the eigenvalue with the
    largest real part. It is a per-row selection, not a tracked branch: there is no
    previous value to stay near, and a negative real part is *not* a rejection — a
    stable critical mode still has a Petermann factor.
    """
    spectrum = np.asarray(values, dtype=complex).reshape(-1)
    if spectrum.size == 0:
        raise ValueError("an empty spectrum has no critical mode")
    return int(np.argmax(np.real(spectrum)))


class PetermannSeries(CAMPostprocessor[PetermannSeriesConfig]):
    """Track the soft Petermann factors across one spectrum series."""

    name: ClassVar[str] = "petermann_series"
    description: ClassVar[str] = (
        "Soft-mode tracking and Petermann scaling over a spectrum annotation"
    )
    config_schema: ClassVar[type[PetermannSeriesConfig]] = PetermannSeriesConfig

    annotation_kind: ClassVar[str] = "petermann_series"
    #: The input is an annotation product, not a solution product.
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("petermann_spectrum",)
    requires_model: ClassVar[bool] = False
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        del context
        fields = self._columns(product)
        identities = [str(value) for value in fields["target_identity"]]
        parameters = fields["parameters"]
        order = self._series_order(parameters)
        tracked: dict[str, dict[str, list[Any]]] = {
            family: {
                "index": [],
                "eigenvalue": [],
                "factor": [],
                "status": [],
                "reason": [],
            }
            for family, _, _ in FAMILIES
        }
        for family, values_name, factors_name in FAMILIES:
            values = fields.get(values_name)
            factors = fields.get(factors_name)
            rule = TRACKING_RULES.get(family, DEFAULT_TRACKING_RULE)
            previous: complex | None = None
            for position in order:
                entry = tracked[family]
                if values is None or factors is None:
                    entry["index"].append(-1)
                    entry["eigenvalue"].append(complex(np.nan, np.nan))
                    entry["factor"].append(np.nan)
                    entry["status"].append(CAMOutcomeStatus.UNIDENTIFIABLE.value)
                    entry["reason"].append(
                        f"the spectrum product has no {values_name!r} column"
                    )
                    continue
                spectrum = np.asarray(values[position], dtype=complex)
                factor_spectrum = np.asarray(factors[position], dtype=float)
                if not np.all(np.isfinite(spectrum)):
                    entry["index"].append(-1)
                    entry["eigenvalue"].append(complex(np.nan, np.nan))
                    entry["factor"].append(np.nan)
                    entry["status"].append(CAMOutcomeStatus.UNIDENTIFIABLE.value)
                    entry["reason"].append("the row carries a non-finite spectrum")
                    previous = None
                    continue
                if rule == "critical":
                    index = track_critical_index(spectrum)
                    previous = None
                else:
                    index = track_soft_index(spectrum, previous)
                    previous = complex(spectrum[index])
                tracked_value = complex(spectrum[index])
                factor = float(factor_spectrum[index])
                if rule == "critical":
                    # A stable critical mode is a valid critical mode: its factor is
                    # the quantity the historical scan reported.
                    status = CAMOutcomeStatus.VALID.value
                    reason = ""
                else:
                    positive = bool(np.real(tracked_value) > 0.0)
                    status = (
                        CAMOutcomeStatus.VALID.value
                        if positive
                        else CAMOutcomeStatus.UNIDENTIFIABLE.value
                    )
                    reason = (
                        ""
                        if positive
                        else "the tracked mode is damped, so it is not a soft mode"
                    )
                entry["index"].append(int(index))
                entry["eigenvalue"].append(tracked_value)
                entry["factor"].append(factor)
                entry["status"].append(status)
                entry["reason"].append(reason)
        fits = self._fits(tracked, parameters, order)
        self.result_metadata = {
            "service": "qphase_cam.postprocessor.petermann_series",
            "families": [family for family, _, _ in FAMILIES],
            "tracking": {
                family: TRACKING_RULES.get(family, DEFAULT_TRACKING_RULE)
                for family, _, _ in FAMILIES
            },
            "tracking_rules": {
                "soft": (
                    "positive-real-part seed with the smallest magnitude, then the "
                    "eigenvalue nearest the previous row; a tracked mode without a "
                    "positive real part is unidentifiable"
                ),
                "critical": (
                    "the eigenvalue with the largest real part of each row; a finite "
                    "one is valid even when its real part is negative"
                ),
            },
            "parameter": self.config.parameter,
            "fit": (
                "log K on log |parameter - critical_value|"
                if self.config.parameter
                else "not requested"
            ),
        }
        fields_out: dict[str, Any] = {
            "reason": np.asarray([""], dtype=str),
            "parameter": np.asarray(
                [[self._parameter_text(parameters[position]) for position in order]],
                dtype=str,
            ),
            "parameter_value": np.asarray(
                [[self._parameter_value(parameters[position]) for position in order]],
                dtype=float,
            ),
        }
        for family, _, _ in FAMILIES:
            entry = tracked[family]
            fields_out[f"{family}_soft_index"] = np.asarray(
                [entry["index"]], dtype=np.int64
            )
            fields_out[f"{family}_soft_eigenvalue"] = np.asarray(
                [entry["eigenvalue"]], dtype=complex
            )
            fields_out[f"{family}_soft_petermann"] = np.asarray(
                [entry["factor"]], dtype=float
            )
            fields_out[f"{family}_status"] = np.asarray([entry["status"]], dtype=str)
        for family, _, _ in FAMILIES:
            fit = fits[family]
            fields_out[f"{family}_fit_exponent"] = np.asarray([fit["exponent"]])
            fields_out[f"{family}_fit_log_rmse"] = np.asarray([fit["log_rmse"]])
            fields_out[f"{family}_fit_r_squared"] = np.asarray([fit["r_squared"]])
            fields_out[f"{family}_fit_point_count"] = np.asarray(
                [fit["point_count"]], dtype=np.int64
            )
        fields_out["families"] = np.asarray(
            [[family for family, _, _ in FAMILIES]], dtype=str
        )
        return build_annotation_product(
            annotation_kind=self.annotation_kind,
            target_kind="case",
            target_identity=[self._series_identity(identities, order)],
            status=[self._series_status(tracked)],
            fields=fields_out,
            field_axes={
                "families": ("family",),
                "parameter": ("point",),
                "parameter_value": ("point",),
                **{
                    f"{family}_{suffix}": ("point",)
                    for family, _, _ in FAMILIES
                    for suffix in (
                        "soft_index",
                        "soft_eigenvalue",
                        "soft_petermann",
                        "status",
                    )
                },
            },
            axes={"point": len(order), "family": len(FAMILIES)},
            method_metadata=[dict(self.result_metadata)],
        )

    # -- inputs -----------------------------------------------------------

    @staticmethod
    def _columns(product: Any) -> dict[str, Any]:
        names = {variable.name for variable in product.variables}
        columns: dict[str, Any] = {
            "target_identity": np.asarray(
                product.handle("target_identity").materialize()
            ),
        }
        for name in (
            "parameters",
            *(values for _, values, _ in FAMILIES),
            *(factors for _, _, factors in FAMILIES),
        ):
            if name in names:
                columns[name] = np.asarray(product.handle(name).materialize())
            elif name == "parameters":
                # Only the parameter column is optional: without it the series is
                # ordered as stored. A missing spectrum family stays missing (its
                # rows report "no such column") instead of being filled with text a
                # numeric cast would choke on.
                columns[name] = ["{}" for _ in range(columns["target_identity"].size)]
        return columns

    def _series_order(self, parameters: Any) -> list[int]:
        count = len(parameters)
        if self.config.parameter is None:
            return list(range(count))
        values = np.asarray(
            [self._parameter_value(item) for item in parameters], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"the series parameter {self.config.parameter!r} is not present and "
                "finite in every row"
            )
        return [int(index) for index in np.argsort(values)]

    @staticmethod
    def _parameter_map(item: Any) -> dict[str, Any]:
        import json

        return json.loads(str(item))

    def _parameter_value(self, item: Any) -> float:
        mapping = self._parameter_map(item)
        name = self.config.parameter
        if name is None or name not in mapping:
            return float("nan")
        return float(mapping[name])

    def _parameter_text(self, item: Any) -> str:
        return str(item)

    # -- fits -------------------------------------------------------------

    def _fits(
        self,
        tracked: dict[str, dict[str, list[Any]]],
        parameters: Any,
        order: list[int],
    ) -> dict[str, dict[str, Any]]:
        empty = {
            "exponent": np.nan,
            "log_rmse": np.nan,
            "r_squared": np.nan,
            "point_count": 0,
        }
        fits = {family: dict(empty) for family, _, _ in FAMILIES}
        if (
            self.config.parameter is None
            or self.config.critical_value is None
            or self.config.inner_max is None
        ):
            return fits
        for family, _, _ in FAMILIES:
            entry = tracked[family]
            x: list[float] = []
            y: list[float] = []
            for slot, position in enumerate(order):
                if entry["status"][slot] != CAMOutcomeStatus.VALID.value:
                    continue
                factor = float(entry["factor"][slot])
                value = self._parameter_value(parameters[position])
                if not np.isfinite(factor) or factor <= 0.0:
                    continue
                distance = abs(value - float(self.config.critical_value))
                if not np.isfinite(distance) or distance <= 0.0:
                    continue
                if distance > float(self.config.inner_max):
                    continue
                x.append(distance)
                y.append(factor)
            if len(x) < 2:
                continue
            pairs = sorted(zip(x, y, strict=True))
            xs = np.asarray([item[0] for item in pairs])
            ys = np.asarray([item[1] for item in pairs])
            fit = log_linear_scaling_fit(xs, ys)
            log_y = np.log(ys)
            residual = log_y - (fit.intercept + fit.exponent * np.log(xs))
            total = float(np.sum((log_y - log_y.mean()) ** 2))
            r_squared = (
                1.0 - float(np.sum(residual**2)) / total if total > 0.0 else np.nan
            )
            fits[family] = {
                "exponent": float(fit.exponent),
                "log_rmse": float(fit.log_rmse),
                "r_squared": float(r_squared),
                "point_count": int(fit.point_count),
            }
        return fits

    # -- summary ----------------------------------------------------------

    @staticmethod
    def _series_identity(identities: list[str], order: list[int]) -> str:
        import hashlib

        from qphase.core.utils import canonical_json

        payload = {
            "kind": "petermann_series",
            "targets": [identities[position] for position in order],
        }
        text = canonical_json(payload)
        return "cam:series:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _series_status(tracked: dict[str, dict[str, list[Any]]]) -> str:
        statuses = [status for entry in tracked.values() for status in entry["status"]]
        if not statuses:
            return CAMOutcomeStatus.UNCHECKED.value
        if all(status == CAMOutcomeStatus.VALID.value for status in statuses):
            return CAMOutcomeStatus.VALID.value
        if any(status == CAMOutcomeStatus.VALID.value for status in statuses):
            return CAMOutcomeStatus.VALID.value
        return CAMOutcomeStatus.UNIDENTIFIABLE.value
