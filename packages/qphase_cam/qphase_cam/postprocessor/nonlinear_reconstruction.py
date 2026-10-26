"""Nonlinear slow-manifold reconstruction annotation for CAM states.

One row per (target, order): the slow manifold to that order, the stationary scalar
problem along it, and the Gaussian terms the reconstruction corrects. Everything
comes from :mod:`qphase_cam.nonlinear` — this plugin only reads targets out of a
typed product and writes the numbers back as typed fields.

The annotation is named ``nonlinear_reconstruction`` rather than ``prediction``: it
reports what the truncated manifold implies at the state it was given (a
reconstruction of the scalar intensity), not a frozen forecast compared against a
measurement. A prediction product, if it is ever needed, would consume these
reconstructions and carry its own provenance — it would not be a second name for
the same annotation.

Target identity follows the product (``solution_identity`` without a case scope,
``candidate_identity`` with the case scope the candidate view carries). The readout
defaults to the Rayleigh frequency; any other readout is a
:class:`~qphase_cam.observable.CAMObservable`.

``config.effective`` is optional and off by default. When it is set, every
(target, order) also carries the full effective-drift curve of the *same*
reconstruction (:func:`~qphase_cam.effective_drift.effective_drift_curve`), spread
over an ``effective_coarse_point`` and an ``effective_dense_point`` axis. The curve
re-uses the reconstruction's grid, so ``config.reconstruction.grid_points`` must
equal ``config.effective.coarse_points_per_side``: both count points per side, the
coarse grid holds ``2 · side - 1`` points and a mismatch is rejected before any
number is computed. With ``effective=None`` no field, axis or metadata entry is
added, so the default annotation is unchanged. A curve that fails is reported
through ``effective_status``/``effective_reason``/``effective_reason_code`` with
unavailable values (NaN, ``False``, ``-1``), and its verdict also caps the row's own
status — the curve never hides an internal failure and never upgrades a source that
was not valid.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.data import build_annotation_product
from qphase_cam.effective_drift import EffectiveDriftConfig, effective_drift_curve
from qphase_cam.nonlinear import (
    DEFAULT_MANIFOLD_ORDERS,
    NonlinearManifold,
    NonlinearManifoldConfig,
    ScalarReconstruction,
    ScalarReconstructionConfig,
    ou_intensities,
    reconstructed_intensity,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.observable import CAMObservable, RayleighObservable
from qphase_cam.outcomes import CAMOutcomeStatus

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .targets import AnnotationTarget, annotation_target_role, annotation_targets

__all__ = ["NonlinearReconstruction", "NonlinearReconstructionConfig"]


class NonlinearReconstructionConfig(CAMPostprocessorConfig):
    """Orders, grid controls, the optional drive direction and the readout.

    ``effective`` enables the full effective-drift curve per (target, order). It is
    ``None`` by default, and then nothing about the annotation changes. Its
    ``coarse_points_per_side`` counts points per side exactly like
    ``reconstruction.grid_points``, because the curve re-uses that grid instead of
    sampling a second one.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    orders: tuple[int, ...] = Field(
        default=DEFAULT_MANIFOLD_ORDERS,
        description="Normal-form orders to reconstruct (the reference set is 3,5,7)",
    )
    manifold: NonlinearManifoldConfig = Field(
        default_factory=NonlinearManifoldConfig,
        description="Order conventions of the slow manifold",
    )
    reconstruction: ScalarReconstructionConfig = Field(
        default_factory=ScalarReconstructionConfig,
        description="Grid and boundary-search controls of the scalar problem",
    )
    perturbation_parameter: str | None = Field(
        default=None,
        description="Parameter whose direction drives the path-information rate; "
        "without it those fields stay unavailable",
    )
    perturbation_scale: float = Field(
        default=1.0, gt=0.0, description="Scale of that parameter direction"
    )
    observable: CAMObservable | None = Field(
        default=None,
        description="Readout to reconstruct; defaults to the Rayleigh frequency",
    )
    effective: EffectiveDriftConfig | None = Field(
        default=None,
        description="Full effective-drift curve per (target, order); off by default, "
        "and its coarse_points_per_side must equal reconstruction.grid_points",
    )

    @model_validator(mode="after")
    def _check_effective_grid(self) -> NonlinearReconstructionConfig:
        """Reject a curve that would not run on the reconstruction's own grid."""
        effective = self.effective
        if effective is None:
            return self
        grid_points = int(self.reconstruction.grid_points)
        per_side = int(effective.coarse_points_per_side)
        if grid_points != per_side:
            raise ValueError(
                "the effective curve re-uses the reconstruction's grid, so "
                f"reconstruction.grid_points ({grid_points} points per side) must "
                f"equal effective.coarse_points_per_side ({per_side}); the curve's "
                f"coarse grid holds {2 * per_side - 1} points"
            )
        return self


#: Per-row numeric fields of the annotation.
_NUMERIC_FIELDS: tuple[str, ...] = (
    "soft_rate",
    "critical_eigenvalue_real",
    "noncritical_spectral_gap",
    "bordered_condition",
    "sigma",
    "x_min",
    "x_max",
    "scalar_intensity",
    "stationary_mean",
    "stationary_response",
    "path_information_rate",
    "minimum_projected_diffusion",
    "endpoint_density_ratio",
    "edge_mass_fraction",
    "density_normalization",
    "soft_intensity",
    "full_intensity",
    "reconstructed_intensity",
)

#: Per-row tri-state boolean fields (-1 unknown).
_TRISTATE_FIELDS: tuple[str, ...] = (
    "confining",
    "negative_physical_boundary",
    "positive_physical_boundary",
)

#: Per-(target, order) summary of the effective curve: (field, summary key). A float
#: field that the curve did not produce stays NaN.
_EFFECTIVE_SUMMARY_FIELDS: tuple[tuple[str, str], ...] = (
    ("effective_stationary_mean", "stationary_mean"),
    ("effective_stationary_mean_x", "stationary_mean_x"),
    ("effective_stationary_response", "stationary_response"),
    ("effective_allan_intensity", "allan_intensity"),
    ("effective_information_rate", "information_rate"),
    ("effective_path_information_rate", "path_information_rate"),
    ("effective_information_over_path", "information_over_path"),
    ("effective_edge_mass_fraction", "edge_mass_fraction"),
    ("effective_endpoint_density_ratio", "endpoint_density_ratio"),
    ("effective_minimum_q_effective", "minimum_q_effective"),
    ("effective_minimum_stable_margin", "minimum_stable_margin"),
    (
        "effective_probability_stable_margin_le_1x_soft_rate",
        "probability_stable_margin_le_1x_soft_rate",
    ),
    (
        "effective_probability_stable_margin_le_3x_soft_rate",
        "probability_stable_margin_le_3x_soft_rate",
    ),
    (
        "effective_probability_stable_margin_le_5x_soft_rate",
        "probability_stable_margin_le_5x_soft_rate",
    ),
    (
        "effective_probability_stable_margin_le_10x_soft_rate",
        "probability_stable_margin_le_10x_soft_rate",
    ),
    ("effective_lyapunov_residual_max", "lyapunov_residual_max"),
)

#: Per-coarse-point fields of the effective curve: (field, curve key).
_EFFECTIVE_COARSE_FIELDS: tuple[tuple[str, str], ...] = (
    ("effective_x", "x"),
    ("effective_f_deterministic", "f_deterministic"),
    ("effective_f_effective", "f_effective"),
    ("effective_q_effective", "q_effective"),
    ("effective_omega_effective", "omega_effective"),
    ("effective_forcing", "forcing"),
    ("effective_stable_covariance_trace", "stable_covariance_trace"),
    ("effective_stable_margin", "stable_margin"),
    ("effective_physical_minimum", "physical_minimum"),
    ("effective_lyapunov_residual", "lyapunov_residual"),
)

#: Per-dense-point fields of the effective curve: (field, curve key).
_EFFECTIVE_DENSE_FIELDS: tuple[tuple[str, str], ...] = (
    ("effective_dense_x", "x"),
    ("effective_density", "density"),
    ("effective_f_effective_dense", "f_effective"),
    ("effective_q_effective_dense", "q_effective"),
    ("effective_omega_effective_dense", "omega_effective"),
    ("effective_forcing_dense", "forcing"),
    ("effective_stable_margin_dense", "stable_margin"),
)


class NonlinearReconstruction(CAMPostprocessor[NonlinearReconstructionConfig]):
    """Reconstruct the stationary scalar problem along the slow manifold."""

    name: ClassVar[str] = "nonlinear_reconstruction"
    description: ClassVar[str] = (
        "O3/O5/O7 slow-manifold scalar reconstruction of one readout"
    )
    config_schema: ClassVar[type[NonlinearReconstructionConfig]] = (
        NonlinearReconstructionConfig
    )

    annotation_kind: ClassVar[str] = "nonlinear_reconstruction"
    accepted_product_roles: ClassVar[tuple[str, ...]] = (
        "state",
        "state_scan",
        "candidates",
    )
    requires_model: ClassVar[bool] = True
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        model = context.model
        self._check_effective_config()
        adapter = QPSGenDynamicsAdapter.from_model(model)
        observable: Any = (
            self.config.observable
            if self.config.observable is not None
            else RayleighObservable(model)
        )
        if not isinstance(observable, CAMObservable):
            raise TypeError(
                "nonlinear_reconstruction needs a CAMObservable readout, got "
                f"{type(observable).__name__}"
            )
        role = annotation_target_role(product, context, consumer=self.name)
        targets = annotation_targets(product, context, model, consumer=self.name)
        rows = self._rows(adapter, model, observable, targets)
        return self._annotation(
            rows, observable, "candidate" if role == "candidates" else "state"
        )

    def _check_effective_config(self) -> None:
        """Reject an effective curve that cannot run on the reconstruction grid.

        The config validator already refuses the mismatch when the options are
        parsed; this repeats the check before any number is computed, because an
        already-validated config can be passed straight through to the constructor.
        """
        effective = self.config.effective
        if effective is None:
            return
        grid_points = int(self.config.reconstruction.grid_points)
        per_side = int(effective.coarse_points_per_side)
        if grid_points != per_side:
            raise ValueError(
                "the effective curve re-uses the reconstruction's grid, so "
                f"reconstruction.grid_points ({grid_points} points per side) must "
                f"equal effective.coarse_points_per_side ({per_side}); the curve's "
                f"coarse grid holds {2 * per_side - 1} points"
            )

    # -- rows -------------------------------------------------------------

    def _rows(
        self,
        adapter: QPSGenDynamicsAdapter,
        model: Any,
        observable: CAMObservable,
        targets: tuple[AnnotationTarget, ...],
    ) -> list[dict[str, Any]]:
        """Return one row per target; the orders live on the row's order axis."""
        rows: list[dict[str, Any]] = []
        direction = None
        if self.config.perturbation_parameter:
            direction = adapter.parameter_direction(
                self.config.perturbation_parameter, self.config.perturbation_scale
            )
        for target in targets:
            entries = [
                self._row(adapter, model, observable, target, int(order), direction)
                for order in self.config.orders
            ]
            rows.append(self._combine(target, entries))
        return rows

    def _row(
        self,
        adapter: Any,
        model: Any,
        observable: CAMObservable,
        target: AnnotationTarget,
        order: int,
        direction: Any | None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "order": order,
            "status": CAMOutcomeStatus.VALID.value,
            "reason": "",
            "reason_code": "",
            "validity_domain": "",
            "confining": None,
            "negative_physical_boundary": None,
            "positive_physical_boundary": None,
            "method": {},
        }
        if not target.success:
            entry.update(
                {
                    "status": CAMOutcomeStatus.NOT_CONVERGED.value,
                    "reason": "the solver did not converge at this target",
                    "reason_code": "not_converged",
                }
            )
            self._without_effective_curve(entry)
            return entry
        config = self.config.manifold.model_copy(update={"order": order})
        try:
            manifold = slow_manifold(
                adapter,
                target.vector,
                target.params,
                order=order,
                config=config,
            )
        except ValueError as exc:
            unstable = "does not decay" in str(exc)
            entry.update(
                {
                    "status": (
                        CAMOutcomeStatus.UNSTABLE.value
                        if unstable
                        else CAMOutcomeStatus.UNIDENTIFIABLE.value
                    ),
                    "reason": str(exc),
                    "reason_code": (
                        "soft_mode_not_decaying" if unstable else "manifold_unresolved"
                    ),
                }
            )
            self._without_effective_curve(entry)
            return entry
        reconstruction = scalar_reconstruction(
            adapter,
            model,
            observable,
            state=target.vector,
            params=target.params,
            order=order,
            parameter_direction=direction,
            manifold=manifold,
            config=self.config.reconstruction,
        )
        entry.update(self._from_reconstruction(reconstruction))
        self._attach_effective_curve(
            entry, adapter, model, observable, target, reconstruction, direction
        )
        try:
            gaussian = ou_intensities(
                adapter,
                observable,
                state=target.vector,
                params=target.params,
                model=model,
            )
        except ValueError as exc:
            entry["method"] = {
                **entry["method"],
                "gaussian_terms": f"unavailable: {exc}",
            }
        else:
            entry["soft_intensity"] = gaussian.soft_intensity
            entry["full_intensity"] = gaussian.full_intensity
            if reconstruction.scalar_intensity is not None:
                entry["reconstructed_intensity"] = reconstructed_intensity(
                    reconstruction.scalar_intensity,
                    full_intensity=gaussian.full_intensity,
                    soft_intensity=gaussian.soft_intensity,
                )
            entry["method"] = {
                **entry["method"],
                "gaussian_terms": {
                    "soft": gaussian.soft_intensity,
                    "full": gaussian.full_intensity,
                    "method": gaussian.method,
                },
            }
        return entry

    @staticmethod
    def _from_reconstruction(reconstruction: ScalarReconstruction) -> dict[str, Any]:
        manifold: NonlinearManifold = reconstruction.manifold
        return {
            "status": reconstruction.status.value,
            "reason": reconstruction.reason or "",
            "reason_code": reconstruction.reason_code or "",
            "validity_domain": reconstruction.validity_domain or "",
            "confining": reconstruction.confining,
            "negative_physical_boundary": reconstruction.negative_physical_boundary,
            "positive_physical_boundary": reconstruction.positive_physical_boundary,
            "soft_rate": manifold.soft_rate,
            "critical_eigenvalue_real": manifold.critical_eigenvalue_real,
            "noncritical_spectral_gap": manifold.noncritical_spectral_gap,
            "bordered_condition": manifold.maximum_bordered_condition,
            "sigma": reconstruction.sigma,
            "x_min": reconstruction.x_min,
            "x_max": reconstruction.x_max,
            "scalar_intensity": reconstruction.scalar_intensity,
            "stationary_mean": reconstruction.stationary_mean,
            "stationary_response": reconstruction.stationary_response,
            "path_information_rate": reconstruction.path_information_rate,
            "minimum_projected_diffusion": (reconstruction.minimum_projected_diffusion),
            "endpoint_density_ratio": reconstruction.endpoint_density_ratio,
            "edge_mass_fraction": reconstruction.edge_mass_fraction,
            "density_normalization": reconstruction.density_normalization,
            "method": {
                "manifold": dict(manifold.method),
                "reconstruction": dict(reconstruction.method),
                "reconstruction_diagnostics": dict(reconstruction.diagnostics),
            },
        }

    # -- effective curve --------------------------------------------------

    def _effective_method(
        self, entry: dict[str, Any], curve: Any | None
    ) -> dict[str, Any]:
        """Return the method block of one row's effective curve."""
        effective = self.config.effective
        assert effective is not None
        block: dict[str, Any] = {
            "config": effective.model_dump(mode="json"),
            "quadrature": effective.quadrature.model_dump(mode="json"),
            "integration_split": effective.quadrature.integration_split,
            "coarse_point_count": 2 * int(effective.coarse_points_per_side) - 1,
            "dense_point_count": int(effective.dense_points),
            "curve": None,
            "source_status": entry["status"],
            "source_reason_code": entry["reason_code"],
            "source_validity_domain": entry["validity_domain"],
            "source_valid": entry["status"] == CAMOutcomeStatus.VALID.value,
        }
        if curve is not None:
            curve_block: dict[str, Any] = {
                "status": curve.status.value,
                "reason_code": curve.reason_code,
                "method": dict(curve.method),
                "has_coarse": bool(curve.coarse),
                "has_dense": bool(curve.dense),
            }
            if not curve.coarse:
                # The failure's diagnostics only live in the summary, so they are
                # carried here rather than silently dropped with the arrays.
                curve_block["diagnostics"] = dict(curve.summary)
            block["curve"] = curve_block
        return block

    def _without_effective_curve(self, entry: dict[str, Any]) -> None:
        """Record that this row has no curve because it failed before the manifold."""
        if self.config.effective is None:
            return
        entry["effective"] = None
        entry["effective_status"] = entry["status"]
        entry["effective_reason"] = (
            entry["reason"] or "the reconstruction failed before the curve"
        )
        entry["effective_reason_code"] = entry["reason_code"] or "no_effective_curve"
        entry["method"]["effective"] = self._effective_method(entry, None)

    def _attach_effective_curve(
        self,
        entry: dict[str, Any],
        adapter: Any,
        model: Any,
        observable: CAMObservable,
        target: AnnotationTarget,
        reconstruction: ScalarReconstruction,
        direction: Any | None,
    ) -> None:
        """Evaluate and record the curve of the reconstruction this row just built."""
        effective = self.config.effective
        if effective is None:
            return
        curve = effective_drift_curve(
            adapter,
            model,
            observable,
            state=target.vector,
            params=target.params,
            reconstruction=reconstruction,
            parameter_direction=direction,
            config=effective,
        )
        base_status = entry["status"]
        base_reason = entry["reason"]
        entry["effective"] = curve
        entry["effective_status"] = curve.status.value
        entry["effective_reason"] = curve.reason or ""
        entry["effective_reason_code"] = curve.reason_code or ""
        entry["method"]["effective"] = self._effective_method(entry, curve)
        if curve.status is CAMOutcomeStatus.VALID:
            return
        # The curve's verdict caps the row: an internal failure is never hidden, and a
        # source that was not valid is never reported as one.
        reason = curve.reason or ""
        if (
            base_status != CAMOutcomeStatus.VALID.value
            and base_status != curve.status.value
        ):
            reason = (
                f"reconstruction: {base_reason or base_status}; "
                f"effective curve: {reason or curve.status.value}"
            )
        entry.update(
            status=curve.status.value,
            reason=reason,
            reason_code=curve.reason_code or entry["reason_code"],
        )

    @staticmethod
    def _combine(
        target: AnnotationTarget, entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Collapse the per-order entries into one row with an order axis."""
        statuses = [entry["status"] for entry in entries]
        failing = [
            entry
            for entry in entries
            if entry["status"] != CAMOutcomeStatus.VALID.value
        ]
        if failing:
            status = failing[0]["status"]
            reason = "; ".join(
                f"order {entry['order']}: {entry['status']}"
                + (f" ({entry['reason']})" if entry["reason"] else "")
                for entry in failing
            )
        else:
            status = CAMOutcomeStatus.VALID.value
            reason = ""
        return {
            "identity": target.identity,
            "case_identity": target.case_identity,
            "status": status,
            "reason": reason,
            "entries": entries,
            "statuses": statuses,
        }

    # -- annotation -------------------------------------------------------

    def _annotation(
        self,
        rows: list[dict[str, Any]],
        observable: CAMObservable,
        target_kind: str,
    ) -> Any:
        width = len(self.config.orders)
        fields: dict[str, Any] = {
            "reason": np.asarray([row["reason"] for row in rows], dtype=str),
            "order": np.asarray(
                [[int(entry["order"]) for entry in row["entries"]] for row in rows],
                dtype=np.int64,
            ),
            "order_status": np.asarray(
                [[entry["status"] for entry in row["entries"]] for row in rows],
                dtype=str,
            ),
            "order_reason": np.asarray(
                [[entry["reason"] for entry in row["entries"]] for row in rows],
                dtype=str,
            ),
            "order_reason_code": np.asarray(
                [[entry["reason_code"] for entry in row["entries"]] for row in rows],
                dtype=str,
            ),
            "order_validity_domain": np.asarray(
                [
                    [entry["validity_domain"] or "" for entry in row["entries"]]
                    for row in rows
                ],
                dtype=str,
            ),
        }
        for name in _NUMERIC_FIELDS:
            fields[name] = np.asarray(
                [
                    [
                        np.nan if entry.get(name) is None else float(entry[name])
                        for entry in row["entries"]
                    ]
                    for row in rows
                ],
                dtype=float,
            )
        for name in _TRISTATE_FIELDS:
            fields[name] = np.asarray(
                [
                    [
                        -1 if entry.get(name) is None else int(bool(entry[name]))
                        for entry in row["entries"]
                    ]
                    for row in rows
                ],
                dtype=np.int8,
            )
        axes_in_use = [name for name in _NUMERIC_FIELDS + _TRISTATE_FIELDS]
        field_axes: dict[str, tuple[str, ...]] = {
            "order": ("order",),
            "order_status": ("order",),
            "order_reason": ("order",),
            "order_reason_code": ("order",),
            "order_validity_domain": ("order",),
            **{name: ("order",) for name in axes_in_use},
        }
        axes: dict[str, int] = {"order": width}
        self.result_metadata = {
            "service": "qphase_cam.nonlinear.scalar_reconstruction",
            "orders": [int(order) for order in self.config.orders],
            "readout": observable.identity.model_dump(mode="json"),
            "grid_points": int(self.config.reconstruction.grid_points),
            "perturbation_parameter": self.config.perturbation_parameter,
        }
        effective_fields = self._effective_fields(rows)
        if effective_fields is not None:
            effective, columns = effective_fields
            fields.update(columns)
            per_side = int(effective.coarse_points_per_side)
            field_axes.update(
                {
                    name: ("order",)
                    for name in (
                        "effective_status",
                        "effective_reason",
                        "effective_reason_code",
                        "effective_valid_coarse_count",
                        "effective_interval_min",
                        "effective_interval_max",
                        *(name for name, _ in _EFFECTIVE_SUMMARY_FIELDS),
                    )
                }
            )
            field_axes.update(
                {
                    name: ("order", "effective_coarse_point")
                    for name, _ in _EFFECTIVE_COARSE_FIELDS
                }
            )
            field_axes["effective_valid"] = ("order", "effective_coarse_point")
            field_axes.update(
                {
                    name: ("order", "effective_dense_point")
                    for name, _ in _EFFECTIVE_DENSE_FIELDS
                }
            )
            axes["effective_coarse_point"] = 2 * per_side - 1
            axes["effective_dense_point"] = int(effective.dense_points)
            # The curve's own record is reported next to the annotation so a reader
            # never has to guess which configuration produced the arrays.
            self.result_metadata["effective"] = {
                "enabled": True,
                "coarse_points_per_side": per_side,
                "coarse_point_count": 2 * per_side - 1,
                "dense_points": int(effective.dense_points),
                "integration_split": effective.quadrature.integration_split,
                "config": effective.model_dump(mode="json"),
            }
        return build_annotation_product(
            annotation_kind=self.annotation_kind,
            target_kind=target_kind,
            target_identity=[row["identity"] for row in rows],
            case_identity=(
                [row["case_identity"] for row in rows]
                if target_kind == "candidate"
                else None
            ),
            status=[row["status"] for row in rows],
            fields=fields,
            field_axes=field_axes,
            axes=axes,
            method_metadata=self._method_metadata(rows, observable),
        )

    def _method_metadata(
        self, rows: list[dict[str, Any]], observable: CAMObservable
    ) -> list[dict[str, Any]]:
        """Return one method block per row.

        The ``effective`` entry is only present when the curve is enabled, so the
        default annotation keeps exactly the metadata it had before the option.
        """
        effective = self.config.effective
        blocks: list[dict[str, Any]] = []
        for row in rows:
            block: dict[str, Any] = {
                "orders": [int(order) for order in self.config.orders],
                "per_order": [dict(entry["method"]) for entry in row["entries"]],
                "readout": observable.identity.model_dump(mode="json"),
            }
            if effective is not None:
                block["effective"] = {
                    "enabled": True,
                    "config": effective.model_dump(mode="json"),
                    "integration_split": effective.quadrature.integration_split,
                    "coarse_point_count": 2 * int(effective.coarse_points_per_side) - 1,
                    "dense_points": int(effective.dense_points),
                }
            blocks.append(block)
        return blocks

    # -- effective curve fields -------------------------------------------

    def _effective_fields(
        self, rows: list[dict[str, Any]]
    ) -> tuple[EffectiveDriftConfig, dict[str, Any]] | None:
        """Return the curve columns and their configuration, or ``None`` when off.

        Every column spans the declared axes in full, so a row without a curve (an
        unconverged target, or a curve that failed) contributes NaN, ``False`` and
        ``-1`` instead of a shorter or ragged array. The row's own
        ``effective_status``/``effective_reason`` say why.
        """
        effective = self.config.effective
        if effective is None:
            return None
        per_side = int(effective.coarse_points_per_side)
        coarse_size = 2 * per_side - 1
        dense_size = int(effective.dense_points)
        entries = [row["entries"] for row in rows]
        columns: dict[str, Any] = {
            "effective_status": np.asarray(
                [[entry["effective_status"] for entry in row] for row in entries],
                dtype=str,
            ),
            "effective_reason": np.asarray(
                [[entry["effective_reason"] for entry in row] for row in entries],
                dtype=str,
            ),
            "effective_reason_code": np.asarray(
                [[entry["effective_reason_code"] for entry in row] for row in entries],
                dtype=str,
            ),
            "effective_valid_coarse_count": np.asarray(
                [[_effective_count(entry) for entry in row] for row in entries],
                dtype=np.int64,
            ),
            "effective_interval_min": np.asarray(
                [[_effective_interval(entry, 0) for entry in row] for row in entries],
                dtype=float,
            ),
            "effective_interval_max": np.asarray(
                [[_effective_interval(entry, 1) for entry in row] for row in entries],
                dtype=float,
            ),
        }
        for name, key in _EFFECTIVE_SUMMARY_FIELDS:
            columns[name] = np.asarray(
                [[_effective_summary(entry, key) for entry in row] for row in entries],
                dtype=float,
            )
        columns["effective_valid"] = _effective_stack(
            entries, "coarse", "valid", coarse_size, dtype=bool
        )
        for name, key in _EFFECTIVE_COARSE_FIELDS:
            columns[name] = _effective_stack(entries, "coarse", key, coarse_size)
        for name, key in _EFFECTIVE_DENSE_FIELDS:
            columns[name] = _effective_stack(entries, "dense", key, dense_size)
        return effective, columns


def _effective_curve(entry: dict[str, Any]) -> Any | None:
    """Return the row's curve when it produced arrays, else ``None``."""
    curve = entry.get("effective")
    if curve is None or not curve.coarse:
        return None
    return curve


def _effective_summary(entry: dict[str, Any], key: str) -> float:
    curve = _effective_curve(entry)
    if curve is None:
        return np.nan
    value = curve.summary.get(key)
    if value is None:
        return np.nan
    number = float(value)
    return number if np.isfinite(number) else np.nan


def _effective_count(entry: dict[str, Any]) -> int:
    curve = _effective_curve(entry)
    if curve is None:
        return -1
    value = curve.summary.get("valid_coarse_count")
    return -1 if value is None else int(value)


def _effective_interval(entry: dict[str, Any], position: int) -> float:
    curve = _effective_curve(entry)
    if curve is None:
        return np.nan
    interval = curve.summary.get("coarse_interval")
    if interval is None:
        return np.nan
    return float(interval[position])


def _effective_stack(
    entries: list[list[dict[str, Any]]],
    block: str,
    key: str,
    size: int,
    *,
    dtype: Any = float,
) -> np.ndarray:
    """Return one ``(row, order, point)`` block of the curve's ``block`` arrays."""
    fill: Any = False if dtype is bool else np.nan
    stacked = np.full((len(entries), len(entries[0]) if entries else 0, size), fill)
    for row_index, row in enumerate(entries):
        for order_index, entry in enumerate(row):
            curve = _effective_curve(entry)
            if curve is None:
                continue
            raw = getattr(curve, block).get(key)
            if raw is None:
                # An unavailable column (no forcing, so no dense forcing vector) stays
                # at its unknown fill instead of collapsing to a scalar array.
                continue
            values = np.asarray(raw, dtype=dtype)
            if values.size != size:
                raise ValueError(
                    f"the effective curve's {block} {key!r} has {values.size} values "
                    f"for the declared {size}-point axis"
                )
            stacked[row_index, order_index] = values
    return stacked.astype(dtype, copy=False)
