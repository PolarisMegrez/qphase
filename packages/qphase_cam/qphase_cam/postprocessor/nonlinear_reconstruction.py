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
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.data import build_annotation_product
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
    """Orders, grid controls, the optional drive direction and the readout."""

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
        self.result_metadata = {
            "service": "qphase_cam.nonlinear.scalar_reconstruction",
            "orders": [int(order) for order in self.config.orders],
            "readout": observable.identity.model_dump(mode="json"),
            "grid_points": int(self.config.reconstruction.grid_points),
            "perturbation_parameter": self.config.perturbation_parameter,
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
            field_axes={
                "order": ("order",),
                "order_status": ("order",),
                "order_reason": ("order",),
                "order_reason_code": ("order",),
                "order_validity_domain": ("order",),
                **{name: ("order",) for name in axes_in_use},
            },
            axes={"order": width},
            method_metadata=[
                {
                    "orders": [int(order) for order in self.config.orders],
                    "per_order": [dict(entry["method"]) for entry in row["entries"]],
                    "readout": observable.identity.model_dump(mode="json"),
                }
                for row in rows
            ],
        )
