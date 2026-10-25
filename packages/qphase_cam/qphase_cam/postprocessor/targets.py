"""Reading annotation targets out of a typed CAM product.

Every CAM annotation is written over *targets*: the filled solution slots of a state
or state_scan product, or the candidates of a candidate product with their case scope
and their parameters. This module is the one place that knows how to read them, so
several postprocessors cannot drift apart in how they resolve a parameter map or a
target identity. It reads the product and the annotation context only — no engine
internals, no solver result.

The parameter map of a target is the model parameters overridden by whatever the
product says about that target: the scan parameters of a state slot, and for a
candidate the view's own per-case resolution
(:meth:`~qphase_cam.postprocessor.candidates.CandidateView.params_at`), so this
module never maintains a second merge order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qphase_cam.core.coordinates import matrix_to_vector

from .base import CAMAnnotationContext
from .candidates import candidate_parameters, candidate_view
from .state_view import state_rows

__all__ = [
    "ANNOTATION_TARGET_ROLES",
    "AnnotationTarget",
    "annotation_target_role",
    "annotation_targets",
]

#: Product roles a target can be read from.
ANNOTATION_TARGET_ROLES: tuple[str, ...] = ("state", "state_scan", "candidates")


@dataclass(frozen=True)
class AnnotationTarget:
    """One thing an annotation is written for, with the parameters it holds.

    ``case_identity`` is empty unless the target came from a candidate product, where
    one physical candidate can appear once per scan case.
    """

    identity: str
    case_identity: str
    vector: np.ndarray
    params: dict[str, float]
    success: bool


def annotation_target_role(
    product: Any, context: CAMAnnotationContext, *, consumer: str
) -> str:
    """Return the product role the annotation targets come from.

    The context names the role the engine selected the product for; a product built
    outside the engine falls back to its own ``product`` attribute. ``consumer`` names
    the postprocessor in the error, so an unsupported role says who rejected it.
    """
    role = str(context.source_role or "")
    if not role:
        role = str(getattr(product, "attributes", {}).get("product", ""))
    if role in ANNOTATION_TARGET_ROLES:
        return role
    raise ValueError(
        f"{consumer} annotates state, state_scan or candidates; got {role!r}"
    )


def annotation_targets(
    product: Any,
    context: CAMAnnotationContext,
    model: Any,
    *,
    consumer: str = "this annotation",
) -> tuple[AnnotationTarget, ...]:
    """Return the targets of one product, in product order.

    A state or state_scan product contributes one target per filled solution slot; a
    candidate product contributes one target per candidate row of the view, so one
    physical candidate yields one target per case it was found in.
    """
    role = annotation_target_role(product, context, consumer=consumer)
    if role == "candidates":
        return _candidate_targets(product, context, model)
    return _state_targets(product, model)


def _state_targets(product: Any, model: Any) -> tuple[AnnotationTarget, ...]:
    return tuple(
        AnnotationTarget(
            identity=row.identity,
            case_identity="",
            vector=np.asarray(matrix_to_vector(row.state), dtype=float),
            params={**dict(model.params), **dict(row.parameters)},
            success=bool(row.success),
        )
        for row in state_rows(product)
    )


def _candidate_targets(
    product: Any, context: CAMAnnotationContext, model: Any
) -> tuple[AnnotationTarget, ...]:
    view = candidate_view(product, metadata=dict(context.metadata))
    targets: list[AnnotationTarget] = []
    for row in view.rows:
        targets.append(
            AnnotationTarget(
                identity=row.identity,
                case_identity=row.case_identity,
                vector=np.asarray(row.state_vector, dtype=float),
                params=candidate_parameters(view, row.index, model),
                success=bool(row.success),
            )
        )
    return tuple(targets)
