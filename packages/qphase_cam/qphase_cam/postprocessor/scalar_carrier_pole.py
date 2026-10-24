"""Algebraic carrier pole of the nonlinear scalar generator for CAM states.

One row per target: the carrier frequency and decay rate of one readout, solved as a
pole of the **backward** (Feynman-Kac) generator the slow manifold induces, together
with the per-grid refinement witness the pole was selected from.

The chain is three existing services and nothing else:
:func:`~qphase_cam.nonlinear.slow_manifold` →
:func:`~qphase_cam.nonlinear.scalar_reconstruction`, whose
:class:`~qphase_cam.nonlinear.ScalarGeneratorTable` carries ``x``, ``f``, ``q`` and the
readout, → :func:`~qphase_cam.carrier_pole.carrier_pole`. This plugin only reads
targets out of a typed product and writes the numbers back as typed fields.

What the annotation claims, and what it does not: the approximation is the
``nonlinear_manifold`` scalar problem at the annotated state. It is not the reference
probe's *effective* table, which averages the noise over the local stable covariance
along a whole readout curve, and it is not an SDE-measured carrier. A pole of an
unbounded or unconfining reconstruction is not a controlled number, so such a target
stops at the reconstruction verdict and no pole is computed at all.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from qphase_cam.carrier_pole import (
    CarrierPoleAssessment,
    CarrierPoleConfig,
    carrier_pole,
)
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.data import build_annotation_product
from qphase_cam.nonlinear import (
    NonlinearManifoldConfig,
    ScalarReconstructionConfig,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.observable import CAMObservable, RayleighObservable
from qphase_cam.outcomes import CAMOutcomeStatus

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .targets import AnnotationTarget, annotation_target_role, annotation_targets

__all__ = ["ScalarCarrierPole", "ScalarCarrierPoleConfig"]

#: The approximation the annotation names: the nonlinear-manifold scalar problem.
APPROXIMATION = "nonlinear_manifold"


class ScalarCarrierPoleConfig(CAMPostprocessorConfig):
    """Manifold order, reconstruction grid, pole grids and the readout."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    order: int = Field(
        default=7,
        ge=2,
        description="Normal-form order of the slow manifold (the reference probe's "
        "seventh order)",
    )
    manifold: NonlinearManifoldConfig = Field(
        default_factory=NonlinearManifoldConfig,
        description="Order conventions of the slow manifold",
    )
    reconstruction: ScalarReconstructionConfig = Field(
        default_factory=ScalarReconstructionConfig,
        description="Grid and boundary-search controls of the scalar problem",
    )
    pole: CarrierPoleConfig = Field(
        default_factory=CarrierPoleConfig,
        description="Refinement grids, continuation and eigensolver controls",
    )
    observable: CAMObservable | None = Field(
        default=None,
        description="Readout to track; defaults to the Rayleigh frequency",
    )


class ScalarCarrierPole(CAMPostprocessor[ScalarCarrierPoleConfig]):
    """Annotate the carrier pole of the nonlinear scalar backward generator."""

    name: ClassVar[str] = "scalar_carrier_pole"
    description: ClassVar[str] = (
        "Algebraic carrier frequency and decay rate of one readout along the slow "
        "manifold"
    )
    config_schema: ClassVar[type[ScalarCarrierPoleConfig]] = ScalarCarrierPoleConfig

    annotation_kind: ClassVar[str] = "scalar_carrier_pole"
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
                "scalar_carrier_pole needs a CAMObservable readout, got "
                f"{type(observable).__name__}"
            )
        role = annotation_target_role(product, context, consumer=self.name)
        targets = annotation_targets(product, context, model, consumer=self.name)
        rows = [self._row(adapter, model, observable, target) for target in targets]
        self.result_metadata = {
            "service": "qphase_cam.carrier_pole.carrier_pole",
            "approximation": APPROXIMATION,
            "order": int(self.config.order),
            "readout": observable.identity.model_dump(mode="json"),
            "configured_grid_sizes": [
                int(size) for size in self.config.pole.grid_sizes
            ],
            "continuation_steps": int(self.config.pole.continuation_steps),
        }
        return self._annotation(rows, "candidate" if role == "candidates" else "state")

    # -- one target -------------------------------------------------------

    def _row(
        self,
        adapter: QPSGenDynamicsAdapter,
        model: Any,
        observable: CAMObservable,
        target: AnnotationTarget,
    ) -> dict[str, Any]:
        width = len(self.config.pole.grid_sizes)
        row: dict[str, Any] = {
            "identity": target.identity,
            "case_identity": target.case_identity,
            "status": CAMOutcomeStatus.VALID.value,
            "reason": "",
            "reason_code": "",
            "source_reconstruction_status": "",
            "order": int(self.config.order),
            "carrier_frequency": math.nan,
            "carrier_decay_rate": math.nan,
            "finest_grid_size": 0,
            "grid_converged": -1,
            "grid_size": np.zeros(width, dtype=np.int64),
            "pole_real": np.full(width, math.nan),
            "pole_imag_shifted": np.full(width, math.nan),
            "grid_carrier_frequency": np.full(width, math.nan),
            "grid_carrier_decay_rate": np.full(width, math.nan),
            "grid_frequency_differences": np.full(max(width - 1, 0), math.nan),
            "grid_decay_differences": np.full(max(width - 1, 0), math.nan),
            "method": {
                "service": "qphase_cam.carrier_pole.carrier_pole",
                "approximation": APPROXIMATION,
                "order": int(self.config.order),
                "readout": observable.identity.model_dump(mode="json"),
                "configured_grid_sizes": [
                    int(size) for size in self.config.pole.grid_sizes
                ],
                "continuation_steps": int(self.config.pole.continuation_steps),
                "manifold": {},
                "reconstruction_status": "",
                "validity_domain": "",
                "generator_table_points": 0,
                "pole": {},
            },
        }
        if not target.success:
            row.update(
                {
                    "status": CAMOutcomeStatus.NOT_CONVERGED.value,
                    "reason": "the solver did not converge at this target",
                    "reason_code": "not_converged",
                }
            )
            return row
        try:
            manifold = slow_manifold(
                adapter,
                target.vector,
                target.params,
                order=self.config.order,
                config=self.config.manifold,
            )
        except ValueError as exc:
            unstable = "does not decay" in str(exc)
            row.update(
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
            return row
        row["method"]["manifold"] = dict(manifold.method)
        reconstruction = scalar_reconstruction(
            adapter,
            model,
            observable,
            state=target.vector,
            params=target.params,
            order=self.config.order,
            manifold=manifold,
            config=self.config.reconstruction,
        )
        row["source_reconstruction_status"] = reconstruction.status.value
        row["method"]["reconstruction_status"] = reconstruction.status.value
        row["method"]["validity_domain"] = reconstruction.validity_domain or ""
        if reconstruction.status is not CAMOutcomeStatus.VALID:
            # An unbounded or non-confining reconstruction is not a controlled pole:
            # its verdict travels as the annotation's status and nothing is solved.
            row.update(
                {
                    "status": reconstruction.status.value,
                    "reason": reconstruction.reason or "",
                    "reason_code": reconstruction.reason_code or "",
                }
            )
            return row
        table = reconstruction.generator_table
        if table is None:
            raise RuntimeError(
                "scalar_carrier_pole needs the generator table of a valid "
                f"reconstruction, but {self.name} received a valid reconstruction "
                "without one"
            )
        row["method"]["generator_table_points"] = len(table.x)
        assessment: CarrierPoleAssessment = carrier_pole(
            table.x,
            table.drift,
            table.diffusion,
            table.readout,
            config=self.config.pole,
        )
        row["method"]["pole"] = dict(assessment.method)
        self._fill_pole(row, assessment)
        row.update(
            {
                "status": assessment.status.value,
                "reason": assessment.reason or "",
                "reason_code": assessment.reason_code or "",
            }
        )
        return row

    @staticmethod
    def _fill_pole(row: dict[str, Any], assessment: CarrierPoleAssessment) -> None:
        """Write the pole and its refinement witness, padding the un-solved slots."""
        for position, grid in enumerate(assessment.grids):
            row["grid_size"][position] = grid.grid_size
            row["pole_real"][position] = grid.pole_real
            row["pole_imag_shifted"][position] = grid.pole_imag_shifted
            row["grid_carrier_frequency"][position] = grid.carrier_frequency
            row["grid_carrier_decay_rate"][position] = grid.carrier_decay_rate
        for position, difference in enumerate(assessment.grid_frequency_differences):
            row["grid_frequency_differences"][position] = difference
        for position, difference in enumerate(assessment.grid_decay_differences):
            row["grid_decay_differences"][position] = difference
        if assessment.carrier_frequency is not None:
            row["carrier_frequency"] = float(assessment.carrier_frequency)
        if assessment.carrier_decay_rate is not None:
            row["carrier_decay_rate"] = float(assessment.carrier_decay_rate)
        if assessment.finest_grid_size is not None:
            row["finest_grid_size"] = int(assessment.finest_grid_size)
        if assessment.grid_converged is not None:
            row["grid_converged"] = 1 if assessment.grid_converged else 0

    # -- annotation -------------------------------------------------------

    def _annotation(self, rows: list[dict[str, Any]], target_kind: str) -> Any:
        fields: dict[str, Any] = {
            "reason": np.asarray([row["reason"] for row in rows], dtype=str),
            "reason_code": np.asarray([row["reason_code"] for row in rows], dtype=str),
            "source_reconstruction_status": np.asarray(
                [row["source_reconstruction_status"] for row in rows], dtype=str
            ),
            "order": np.asarray([row["order"] for row in rows], dtype=np.int64),
            "carrier_frequency": np.asarray(
                [row["carrier_frequency"] for row in rows], dtype=float
            ),
            "carrier_decay_rate": np.asarray(
                [row["carrier_decay_rate"] for row in rows], dtype=float
            ),
            "finest_grid_size": np.asarray(
                [row["finest_grid_size"] for row in rows], dtype=np.int64
            ),
            "grid_converged": np.asarray(
                [row["grid_converged"] for row in rows], dtype=np.int8
            ),
        }
        refinement_fields = (
            "grid_size",
            "pole_real",
            "pole_imag_shifted",
            "grid_carrier_frequency",
            "grid_carrier_decay_rate",
        )
        difference_fields = (
            "grid_frequency_differences",
            "grid_decay_differences",
        )
        for name in refinement_fields + difference_fields:
            fields[name] = np.asarray([row[name] for row in rows])
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
                **{name: ("refinement",) for name in refinement_fields},
                **{name: ("refinement_difference",) for name in difference_fields},
            },
            axes={
                "refinement": len(self.config.pole.grid_sizes),
                "refinement_difference": max(len(self.config.pole.grid_sizes) - 1, 0),
            },
            method_metadata=[dict(row["method"]) for row in rows],
        )
