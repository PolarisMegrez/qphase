"""Closure-risk annotation for CAM states and candidates.

The postprocessor is a thin adapter over :mod:`qphase_cam.closure`: it turns each
filled state (or each candidate) into one closure assessment and writes the
numbers as a typed annotation. It adds no physics of its own — the service owns the
definitions, the tolerances and the status vocabulary.

Target identity follows the product: a state product is annotated over its
``solution_identity`` with no case scope, a candidate product over its
``candidate_identity`` with the case scope the view already carries. Empty products
keep the full field schema with zero rows.

The configuration is numeric only (tolerances, finite-difference steps, whether the
Jensen term and the shape fraction are evaluated) plus an optional ``observable``
for a readout other than the default Rayleigh frequency. Thresholds that would
*screen* a campaign are deliberately absent: this annotation is a diagnostic, and
ranking belongs to whoever consumes it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from qphase_cam.closure import (
    ClosureAssessment,
    ClosureAssessmentConfig,
    assess_closure,
    build_closure_assessment_annotation,
)
from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.observable import CAMObservable, RayleighObservable
from qphase_cam.outcomes import CAMOutcomeStatus

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .candidates import candidate_view
from .state_view import state_rows


class ClosureRiskConfig(CAMPostprocessorConfig):
    """Numeric controls, evaluation switches and an optional readout."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    closure: ClosureAssessmentConfig = Field(
        default_factory=ClosureAssessmentConfig,
        description="Tolerances, steps and switches of the closure assessment",
    )
    observable: CAMObservable | None = Field(
        default=None,
        description="Readout to assess; defaults to the Rayleigh frequency of the "
        "model it is asked about",
    )


class ClosureRisk(CAMPostprocessor[ClosureRiskConfig]):
    """Second-order closure bias of one readout per state or candidate."""

    name: ClassVar[str] = "closure_risk"
    description: ClassVar[str] = (
        "Second-order closure bias (drift contraction, displacement, Jensen)"
    )
    config_schema: ClassVar[type[ClosureRiskConfig]] = ClosureRiskConfig

    annotation_kind: ClassVar[str] = "closure_risk"
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
                "closure_risk needs a CAMObservable readout, got "
                f"{type(observable).__name__}"
            )
        role = self._target_role(product, context)
        if role == "candidates":
            assessments, identities, cases = self._candidate_assessments(
                product, context, adapter, model, observable
            )
            target_kind = "candidate"
        else:
            assessments, identities = self._state_assessments(
                product, adapter, model, observable
            )
            cases = None
            target_kind = "state"
        self.result_metadata = {
            "service": "qphase_cam.closure.assess_closure",
            "target_role": role,
            "observable": observable.identity.name,
            "observable_kind": observable.identity.kind.value,
            "charge_sector": observable.identity.sector.value,
            "kubo_threshold": float(self.config.closure.kubo_threshold),
            "screening": "none: the annotation reports numbers and statuses only",
        }
        return build_closure_assessment_annotation(
            assessments,
            target_identity=identities,
            case_identity=cases,
            annotation_kind=self.annotation_kind,
            target_kind=target_kind,
        )

    @staticmethod
    def _target_role(product: Any, context: CAMAnnotationContext) -> str:
        """Return which product kind this call annotates."""
        role = str(context.source_role or "")
        if not role:
            role = str(getattr(product, "attributes", {}).get("product", ""))
        if role in ("state", "state_scan"):
            return role
        if role == "candidates":
            return role
        raise ValueError(
            f"closure_risk annotates state, state_scan or candidates; got {role!r}"
        )

    def _state_assessments(
        self,
        product: Any,
        adapter: QPSGenDynamicsAdapter,
        model: Any,
        observable: CAMObservable,
    ) -> tuple[list[ClosureAssessment], list[str]]:
        rows = state_rows(product)
        assessments: list[ClosureAssessment] = []
        identities: list[str] = []
        for row in rows:
            state = np.asarray(matrix_to_vector(row.state), dtype=float)
            assessment = assess_closure(
                adapter,
                observable,
                state=state,
                params=dict(row.parameters),
                model=model,
                config=self.config.closure,
            )
            assessments.append(
                assessment
                if row.success
                else assessment.model_copy(
                    update={
                        "status": CAMOutcomeStatus.NOT_CONVERGED,
                        "reason": "the solver did not converge at this slot",
                    }
                )
            )
            identities.append(row.identity)
        return assessments, identities

    def _candidate_assessments(
        self,
        product: Any,
        context: CAMAnnotationContext,
        adapter: QPSGenDynamicsAdapter,
        model: Any,
        observable: CAMObservable,
    ) -> tuple[list[ClosureAssessment], list[str], list[str]]:
        view = candidate_view(product, metadata=dict(context.metadata))
        assessments: list[ClosureAssessment] = []
        identities: list[str] = []
        cases: list[str] = []
        for row in view.rows:
            params = self._candidate_parameters(view, row, model)
            state = np.asarray(row.state_vector, dtype=float)
            assessment = assess_closure(
                adapter,
                observable,
                state=state,
                params=params,
                model=model,
                config=self.config.closure,
            )
            assessments.append(
                assessment
                if row.success
                else assessment.model_copy(
                    update={
                        "status": CAMOutcomeStatus.NOT_CONVERGED,
                        "reason": "the candidate did not converge",
                    }
                )
            )
            identities.append(row.identity)
            cases.append(row.case_identity)
        return assessments, identities, cases

    @staticmethod
    def _candidate_parameters(view: Any, row: Any, model: Any) -> dict[str, Any]:
        """Return the model parameters of one candidate's case."""
        params = dict(model.params)
        params.update(view.meta.get("fixed_params", {}))
        for name, values in view.case_params.items():
            params[name] = float(np.asarray(values).reshape(-1)[row.case])
        params.update(row.controls)
        return params
