"""qphase_cam: the immutable prediction contract and product builder
---------------------------------------------------------
A :class:`CAMPrediction` is one statement of what an approximation *implies* for a
readout at one target and one order: the expected frequency response, the Allan
intensity, the information rates and the validity domain the numbers hold in. It
is a **product**, not an annotation:

* the identity is a content digest (observable, target, approximation, order,
  validity domain, source artifact ids and the numeric payload) — never a slot
  position, a path or a job name;
* the source artifacts are immutable catalog ids; a prediction that cannot name
  where it came from is either an explicit ``intermediate`` (with a stated reason)
  or it is not built at all;
* the frozen model has no setters, so a validation step adds parents and
  provenance to the artifact and leaves these fields untouched.

Three approximations are distinguished because they mean different things:
``soft_ou`` is the critical-mode-only Gaussian estimate, ``full_ou`` the full
linear response, ``one_loop`` the first closure correction and
``nonlinear_manifold`` the reconstruction along a truncated slow manifold. None of
them is a rigorous bound on the full nonlinear error, and the product says so in
its attributes instead of letting a reader assume otherwise.

``full_information_rate`` is ``response**2 / reconstructed_intensity``: it is only
defined when the reconstructed intensity is positive and finite, and a prediction
whose denominator fails keeps its status and leaves the rate unavailable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qphase.core.utils import canonical_json
from qphase.data import Dataset, StatisticsDataset, VariableSchema

from .data._common import closed_product, json_text_column, string_column
from .data.declarations import CAM_PREDICTION_PRODUCT, PREDICTION_STATUS_SEMANTICS
from .data.evidence import SourceArtifactRef
from .outcomes import CAMOutcomeStatus

__all__ = [
    "COMPLETE_PREDICTION",
    "INTERMEDIATE_PREDICTION",
    "CAMPrediction",
    "build_prediction_product",
    "prediction_identity",
]

#: Completeness marker of a prediction whose sources are all published artifacts.
COMPLETE_PREDICTION = "complete"

#: Completeness marker of a prediction derived from in-memory intermediates.
INTERMEDIATE_PREDICTION = "intermediate"

#: What the prediction's numbers are an approximation of.
PredictionApproximation = Literal[
    "soft_ou", "full_ou", "one_loop", "nonlinear_manifold"
]

_NUMERIC_FIELDS: tuple[str, ...] = (
    "expected_frequency_response_full",
    "expected_frequency_response_soft",
    "expected_frequency_response_regular",
    "allan_intensity_soft",
    "allan_intensity_full",
    "allan_intensity_reconstructed",
    "path_information_rate",
    "full_information_rate",
    "regular_response",
    "bordered_condition",
    "soft_rate",
    "sigma",
    "x_min",
    "x_max",
)


class CAMPrediction(BaseModel):
    """One immutable prediction of one readout at one target and order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str = Field(min_length=1)
    target_identity: str = Field(min_length=1)
    observable_name: str = Field(min_length=1)
    observable_kind: str = Field(min_length=1)
    charge_sector: str = Field(min_length=1)
    measurement_mapping: str = Field(min_length=1)
    approximation: PredictionApproximation
    order: int = Field(ge=0)
    status: CAMOutcomeStatus
    reason: str | None = None
    validity_domain: dict[str, Any] = Field(default_factory=dict)

    expected_frequency_response_full: float | None = None
    expected_frequency_response_soft: float | None = None
    expected_frequency_response_regular: float | None = None
    allan_intensity_soft: float | None = None
    allan_intensity_full: float | None = None
    allan_intensity_reconstructed: float | None = None
    path_information_rate: float | None = None
    full_information_rate: float | None = None
    regular_response: float | None = None
    bordered_condition: float | None = None
    soft_rate: float | None = None
    sigma: float | None = None
    x_min: float | None = None
    x_max: float | None = None
    negative_physical_boundary: bool | None = None
    positive_physical_boundary: bool | None = None

    completeness: Literal["complete", "intermediate"] = "complete"
    source_artifacts: tuple[SourceArtifactRef, ...] = ()
    method: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_sources(self) -> CAMPrediction:
        if self.completeness == "complete" and not self.source_artifacts:
            raise ValueError(
                "a complete prediction needs at least one source artifact id; an "
                "in-memory result is built with completeness='intermediate' and a "
                "reason"
            )
        if (
            self.completeness == INTERMEDIATE_PREDICTION
            and not (self.reason or "").strip()
        ):
            raise ValueError(
                "an intermediate prediction must state its reason so it cannot be "
                "mistaken for a complete one"
            )
        identifiers = [ref.artifact_id for ref in self.source_artifacts]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("source artifacts must be unique")
        if float(self.order) < 0:
            raise ValueError("order must be non-negative")
        return self

    def payload(self) -> dict[str, Any]:
        """Return the content the identity is computed over."""
        return {
            "observable": {
                "name": self.observable_name,
                "kind": self.observable_kind,
                "sector": self.charge_sector,
                "measurement_mapping": self.measurement_mapping,
            },
            "target_identity": self.target_identity,
            "approximation": self.approximation,
            "order": int(self.order),
            "status": self.status.value,
            "validity_domain": dict(self.validity_domain),
            "source_artifacts": sorted(
                ref.artifact_id for ref in self.source_artifacts
            ),
            # A non-finite number is "unavailable" for identity purposes: JSON has
            # no NaN, and NaN never compares equal to itself.
            "numbers": {
                name: _finite_or_none(getattr(self, name)) for name in _NUMERIC_FIELDS
            },
        }


def _finite_or_none(value: Any) -> float | None:
    """Return one number as a JSON-safe value, mapping non-finite to ``None``."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def prediction_identity(prediction: CAMPrediction) -> str:
    """Return the content identity of one prediction.

    ``cam:prediction:<16 hex>`` over the canonical payload: the observable, the
    target, the approximation, the order, the validity domain, the source artifact
    ids and the numbers. Recomputing the same prediction anywhere yields the same
    identity; changing any of those inputs changes it.
    """
    text = canonical_json(prediction.payload())
    return "cam:prediction:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _prediction_rows(
    predictions: Sequence[CAMPrediction],
) -> list[CAMPrediction]:
    cases = list(predictions)
    if not cases:
        raise ValueError("a prediction product needs at least one prediction")
    for case in cases:
        if not isinstance(case, CAMPrediction):
            raise TypeError(
                f"build_prediction_product takes CAMPrediction objects; got "
                f"{type(case).__name__}"
            )
    identifiers = [case.prediction_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        duplicated = sorted(
            {name for name in identifiers if identifiers.count(name) > 1}
        )
        raise ValueError(f"prediction ids must be unique; duplicated: {duplicated}")
    return cases


def _evidence_variables_with_widths(
    widths: Mapping[str, int],
) -> list[VariableSchema]:
    variables: list[VariableSchema] = []
    for variable in CAM_PREDICTION_PRODUCT.variables:
        width = widths.get(variable.name)
        variables.append(
            variable
            if width is None
            else variable.model_copy(
                update={"dtype": np.dtype(f"U{max(width, 1)}").str}
            )
        )
    return variables


def build_prediction_product(
    predictions: Sequence[CAMPrediction],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build the typed prediction product from validated predictions.

    Every prediction carries its own completeness marker and source artifacts, so
    the product mixes complete and intermediate rows without pretending that an
    intermediate one is final. Missing numbers are ``NaN`` and undecided booleans
    are ``-1``: a failed approximation keeps its status and reason instead of being
    dropped or filled in.
    """
    cases = _prediction_rows(predictions)
    arrays: dict[str, np.ndarray] = {
        "prediction_id": string_column([case.prediction_id for case in cases]),
        "target_identity": string_column([case.target_identity for case in cases]),
        "observable_name": string_column([case.observable_name for case in cases]),
        "observable_kind": string_column([case.observable_kind for case in cases]),
        "charge_sector": string_column([case.charge_sector for case in cases]),
        "measurement_mapping": string_column(
            [case.measurement_mapping for case in cases]
        ),
        "approximation": string_column([case.approximation for case in cases]),
        "order": np.asarray([int(case.order) for case in cases], dtype=np.int64),
        "status": string_column([case.status.value for case in cases]),
        "reason": string_column([case.reason or "" for case in cases]),
        "validity_domain": json_text_column(
            [dict(case.validity_domain) for case in cases]
        ),
        "completeness": string_column([case.completeness for case in cases]),
        "source_artifacts": json_text_column(
            [
                [ref.model_dump(mode="json") for ref in case.source_artifacts]
                for case in cases
            ]
        ),
        "method": json_text_column([dict(case.method) for case in cases]),
    }
    for name in _NUMERIC_FIELDS:
        arrays[name] = np.asarray(
            [
                np.nan if getattr(case, name) is None else float(getattr(case, name))
                for case in cases
            ],
            dtype=float,
        )
    for name in ("negative_physical_boundary", "positive_physical_boundary"):
        arrays[name] = np.asarray(
            [
                -1 if getattr(case, name) is None else int(bool(getattr(case, name)))
                for case in cases
            ],
            dtype=np.int8,
        )
    widths = {
        name: int(array.dtype.itemsize // 4)
        for name, array in arrays.items()
        if array.dtype.kind == "U"
    }
    attributes = {
        **CAM_PREDICTION_PRODUCT.attributes,
        "prediction_count": len(cases),
        "complete_count": sum(
            1 for case in cases if case.completeness == COMPLETE_PREDICTION
        ),
        "intermediate_count": sum(
            1 for case in cases if case.completeness == INTERMEDIATE_PREDICTION
        ),
        "source_artifacts": sorted(
            {ref.artifact_id for case in cases for ref in case.source_artifacts}
        ),
        "status_vocabulary": [member.value for member in CAMOutcomeStatus],
        "status_semantics": PREDICTION_STATUS_SEMANTICS,
    }
    schema = closed_product(
        CAM_PREDICTION_PRODUCT,
        sizes={"prediction": len(cases)},
        variables=_evidence_variables_with_widths(widths),
        coordinates=[],
        attributes=attributes,
    )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.cam",
        provenance={
            **dict(provenance or {}),
            "prediction_ids": [case.prediction_id for case in cases],
            "source_artifacts": attributes["source_artifacts"],
        },
    )
