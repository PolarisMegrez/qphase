"""Nonlinear prediction product for CAM states.

One prediction per (target, order): the Allan intensity of the truncated slow
manifold, the expected frequency response, the information rates and the validity
domain the numbers hold in. The numbers are the 10F3a services' — this plugin only
collects them into the immutable :class:`~qphase_cam.prediction.CAMPrediction`
contract and publishes the typed ``prediction`` product.

The source artifacts come from the annotation context: the upstream artifact the
states were read from (``input_artifact_id``) or the artifact this job publishes
(``artifact_id``). A target whose provenance cannot be named produces an explicit
``intermediate`` prediction with a stated reason instead of a complete one, so a
downstream validation can never mistake an unprovenanced number for a frozen
prediction.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.data.evidence import SourceArtifactRef
from qphase_cam.nonlinear import (
    DEFAULT_MANIFOLD_ORDERS,
    NonlinearManifoldConfig,
    ScalarReconstructionConfig,
    ou_intensities,
    reconstructed_intensity,
    response_prediction,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.observable import CAMObservable, RayleighObservable
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.prediction import (
    CAMPrediction,
    build_prediction_product,
    prediction_identity,
)

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .candidates import candidate_view
from .state_view import state_rows

__all__ = ["NonlinearPrediction", "NonlinearPredictionConfig"]


class NonlinearPredictionConfig(CAMPostprocessorConfig):
    """Orders, drive parameter, numerical controls and the readout."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    orders: tuple[int, ...] = Field(default=DEFAULT_MANIFOLD_ORDERS)
    perturbation_parameter: str = Field(
        min_length=1,
        description="Control whose direction drives the expected response",
    )
    perturbation_scale: float = Field(default=1.0, gt=0.0)
    response_relative_step: float = Field(
        default=2.0e-6,
        gt=0.0,
        description="Relative step of the explicit parameter derivative",
    )
    approximation: str = Field(
        default="nonlinear_manifold",
        description="soft_ou, full_ou, one_loop or nonlinear_manifold",
    )
    manifold: NonlinearManifoldConfig = Field(default_factory=NonlinearManifoldConfig)
    reconstruction: ScalarReconstructionConfig = Field(
        default_factory=ScalarReconstructionConfig
    )
    observable: CAMObservable | None = Field(
        default=None, description="Readout; defaults to the Rayleigh frequency"
    )


class NonlinearPrediction(CAMPostprocessor[NonlinearPredictionConfig]):
    """Publish immutable predictions from the nonlinear reconstruction."""

    name: ClassVar[str] = "nonlinear_prediction"
    description: ClassVar[str] = (
        "Immutable O3/O5/O7 predictions of one readout's intensity and response"
    )
    config_schema: ClassVar[type[NonlinearPredictionConfig]] = NonlinearPredictionConfig

    annotation_kind: ClassVar[str] = "prediction"
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
                "nonlinear_prediction needs a CAMObservable readout, got "
                f"{type(observable).__name__}"
            )
        sources = self._sources(context)
        role = self._target_role(product, context)
        targets = (
            self._candidate_targets(product, context, model)
            if role == "candidates"
            else self._state_targets(product, model)
        )
        direction = adapter.parameter_direction(
            self.config.perturbation_parameter, self.config.perturbation_scale
        )
        predictions = [
            self._prediction(
                adapter, model, observable, target, int(order), direction, sources
            )
            for target in targets
            for order in self.config.orders
        ]
        self.result_metadata = {
            "service": "qphase_cam.prediction.build_prediction_product",
            "orders": [int(order) for order in self.config.orders],
            "readout": observable.identity.model_dump(mode="json"),
            "approximation": self.config.approximation,
            "source_artifacts": [ref.artifact_id for ref in sources],
            "complete_count": sum(
                1 for item in predictions if item.completeness == "complete"
            ),
            "prediction_count": len(predictions),
        }
        return build_prediction_product(
            predictions,
            provenance={
                "task": "prediction",
                "readout": observable.identity.name,
                "orders": [int(order) for order in self.config.orders],
            },
        )

    # -- sources and targets ----------------------------------------------

    @staticmethod
    def _sources(context: CAMAnnotationContext) -> list[SourceArtifactRef]:
        metadata = dict(context.metadata or {})
        declared = metadata.get("input_artifact_id") or metadata.get("artifact_id")
        if not declared:
            return []
        return [SourceArtifactRef(artifact_id=str(declared))]

    @staticmethod
    def _target_role(product: Any, context: CAMAnnotationContext) -> str:
        role = str(context.source_role or "")
        if not role:
            role = str(getattr(product, "attributes", {}).get("product", ""))
        if role in ("state", "state_scan", "candidates"):
            return role
        raise ValueError(
            f"nonlinear_prediction annotates state, state_scan or candidates; got "
            f"{role!r}"
        )

    @staticmethod
    def _state_targets(product: Any, model: Any) -> list[dict[str, Any]]:
        return [
            {
                "identity": row.identity,
                "vector": np.asarray(matrix_to_vector(row.state), dtype=float),
                "params": {**dict(model.params), **dict(row.parameters)},
                "success": bool(row.success),
            }
            for row in state_rows(product)
        ]

    @staticmethod
    def _candidate_targets(
        product: Any, context: CAMAnnotationContext, model: Any
    ) -> list[dict[str, Any]]:
        view = candidate_view(product, metadata=dict(context.metadata))
        targets: list[dict[str, Any]] = []
        for row in view.rows:
            params = dict(model.params)
            params.update(view.meta.get("fixed_params", {}))
            for name, values in view.case_params.items():
                params[name] = float(np.asarray(values).reshape(-1)[row.case])
            params.update(row.controls)
            targets.append(
                {
                    "identity": row.identity,
                    "vector": np.asarray(row.state_vector, dtype=float),
                    "params": params,
                    "success": bool(row.success),
                }
            )
        return targets

    # -- one prediction ---------------------------------------------------

    def _prediction(
        self,
        adapter: Any,
        model: Any,
        observable: CAMObservable,
        target: dict[str, Any],
        order: int,
        direction: Any,
        sources: list[SourceArtifactRef],
    ) -> CAMPrediction:
        identity = observable.identity
        status = CAMOutcomeStatus.VALID
        reason: str | None = None
        numbers: dict[str, Any] = {}
        validity: dict[str, Any] = {}
        if not target["success"]:
            status = CAMOutcomeStatus.NOT_CONVERGED
            reason = "the solver did not converge at this target"
            return self._build(
                identity, target, order, status, reason, numbers, validity, sources
            )
        try:
            manifold = slow_manifold(
                adapter,
                target["vector"],
                target["params"],
                order=order,
                config=self.config.manifold.model_copy(update={"order": order}),
            )
            reconstruction = scalar_reconstruction(
                adapter,
                model,
                observable,
                state=target["vector"],
                params=target["params"],
                order=order,
                parameter_direction=direction,
                manifold=manifold,
                config=self.config.reconstruction,
            )
            gaussian = ou_intensities(
                adapter,
                observable,
                state=target["vector"],
                params=target["params"],
                model=model,
            )
            response = response_prediction(
                adapter,
                observable,
                state=target["vector"],
                params=target["params"],
                parameter=self.config.perturbation_parameter,
                scale=self.config.perturbation_scale,
                relative_step=self.config.response_relative_step,
            )
        except ValueError as exc:
            status = CAMOutcomeStatus.UNIDENTIFIABLE
            reason = str(exc)
            return self._build(
                identity, target, order, status, reason, numbers, validity, sources
            )
        numbers.update(
            {
                "expected_frequency_response_full": response.full_response,
                "expected_frequency_response_soft": response.soft_response,
                "expected_frequency_response_regular": response.regular_response,
                "allan_intensity_soft": gaussian.soft_intensity,
                "allan_intensity_full": gaussian.full_intensity,
                "regular_response": response.regular_response,
                "bordered_condition": manifold.maximum_bordered_condition,
                "soft_rate": manifold.soft_rate,
                "sigma": reconstruction.sigma,
                "x_min": reconstruction.x_min,
                "x_max": reconstruction.x_max,
                "negative_physical_boundary": (
                    reconstruction.negative_physical_boundary
                ),
                "positive_physical_boundary": (
                    reconstruction.positive_physical_boundary
                ),
                "path_information_rate": reconstruction.path_information_rate,
            }
        )
        reconstructed = None
        if reconstruction.scalar_intensity is not None:
            reconstructed = reconstructed_intensity(
                reconstruction.scalar_intensity,
                full_intensity=gaussian.full_intensity,
                soft_intensity=gaussian.soft_intensity,
            )
            numbers["allan_intensity_reconstructed"] = reconstructed
        if (
            reconstructed is not None
            and np.isfinite(reconstructed)
            and reconstructed > 0.0
        ):
            full_response = response.full_response
            numbers["full_information_rate"] = float(full_response**2 / reconstructed)
        else:
            numbers["full_information_rate"] = None
        validity = {
            "label": reconstruction.validity_domain,
            "order": order,
            "confining": bool(manifold.confining),
            "grid_points": self.config.reconstruction.grid_points,
            "minimum_projected_diffusion": reconstruction.minimum_projected_diffusion,
            "endpoint_density_ratio": reconstruction.endpoint_density_ratio,
            "edge_mass_fraction": reconstruction.edge_mass_fraction,
            "x_min": reconstruction.x_min,
            "x_max": reconstruction.x_max,
            "negative_physical_boundary": (reconstruction.negative_physical_boundary),
            "positive_physical_boundary": (reconstruction.positive_physical_boundary),
        }
        status = reconstruction.status
        reason = reconstruction.reason
        if reconstructed is None and status is CAMOutcomeStatus.VALID:
            status = CAMOutcomeStatus.UNIDENTIFIABLE
            reason = "the reconstructed intensity is not available"
        return self._build(
            identity,
            target,
            order,
            status,
            reason,
            numbers,
            validity,
            sources,
            manifold=manifold,
            response=response,
            reconstruction_method=dict(reconstruction.method),
        )

    def _build(
        self,
        identity: Any,
        target: dict[str, Any],
        order: int,
        status: CAMOutcomeStatus,
        reason: str | None,
        numbers: dict[str, Any],
        validity: dict[str, Any],
        sources: list[SourceArtifactRef],
        *,
        manifold: Any | None = None,
        response: Any | None = None,
        reconstruction_method: dict[str, Any] | None = None,
    ) -> CAMPrediction:
        method: dict[str, Any] = {
            "service": "qphase_cam.nonlinear.scalar_reconstruction",
            "approximation": self.config.approximation,
            "information_rate": "full_response^2 / reconstructed_intensity",
            "regular_combination": "response_regular = response_full - response_soft",
            "readout": identity.model_dump(mode="json"),
            "perturbation_parameter": self.config.perturbation_parameter,
            "orders": [int(order) for order in self.config.orders],
        }
        if manifold is not None:
            method["manifold"] = dict(manifold.method)
        if response is not None:
            method["response"] = dict(response.method)
            method["response_derivative"] = response.derivative.model_dump(mode="json")
        if reconstruction_method is not None:
            method["reconstruction"] = reconstruction_method
        completeness = "complete" if sources else "intermediate"
        if not sources:
            reason = (
                (reason + "; " if reason else "")
                + "the annotation context names no published source artifact, so "
                "this prediction is an intermediate of the current run"
            )
        draft = CAMPrediction(
            prediction_id="pending",
            target_identity=target["identity"],
            observable_name=identity.name,
            observable_kind=identity.kind.value,
            charge_sector=identity.sector.value,
            measurement_mapping=identity.measurement_mapping,
            approximation=self.config.approximation,
            order=int(order),
            status=status,
            reason=reason,
            validity_domain=dict(validity),
            completeness=completeness,
            source_artifacts=tuple(sources),
            method=method,
            **{name: numbers.get(name) for name in numbers},
        )
        return draft.model_copy(update={"prediction_id": prediction_identity(draft)})
