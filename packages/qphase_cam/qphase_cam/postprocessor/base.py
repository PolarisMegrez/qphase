"""Public base class for CAM postprocessor plugins.

A postprocessor is an *annotation producer*: it reads one typed product and
writes one annotation product attached to that product's identities. Phase 10D2
replaces the old ``accepted_result_kinds`` / ``process(result, model, backend) ->
dict`` contract with this typed one, so no result object or postprocess dictionary
survives between the solver and an annotation.

The declarations below are static: the engine selects a postprocessor by the
product role it annotates and by the model/backend it requires, and the tests
freeze them against the real behaviour of each postprocessor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase
from qphase.data import Dataset, ProductDeclaration

from qphase_cam.data import CAM_ANNOTATION_DECLARATION

__all__ = [
    "CAMAnnotationContext",
    "CAMPostprocessor",
    "CAMPostprocessorConfig",
    "POSTPROCESSOR_TARGET_ROLES",
]


class CAMPostprocessorConfig(PluginConfigBase):
    """Strict postprocessor configuration base."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


PostprocessorConfigT = TypeVar("PostprocessorConfigT", bound=CAMPostprocessorConfig)

#: Product roles an annotation may be attached to.
POSTPROCESSOR_TARGET_ROLES: tuple[str, ...] = (
    "state",
    "state_scan",
    "candidates",
    "continuation",
)


@dataclass(frozen=True)
class CAMAnnotationContext:
    """What an annotation needs besides the product it annotates.

    ``products`` holds the sibling products of the same bundle by role, so an
    annotation that needs a second view (for example a candidate annotation that
    cross-checks the state product) reads it through a declared requirement
    instead of reaching into engine internals. ``metadata`` is execution metadata
    (solver identity, search design); no scientific payload travels there.
    """

    model: Any = None
    backend: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    products: Mapping[str, Dataset] = field(default_factory=dict)
    source_role: str = ""

    def require_product(self, role: str) -> Dataset:
        """Return one sibling product by role."""
        try:
            return self.products[role]
        except KeyError:
            raise KeyError(
                f"annotation context has no product {role!r}; available: "
                f"{sorted(self.products)}"
            ) from None


class CAMPostprocessor(ABC, Generic[PostprocessorConfigT]):
    """Base class for computations applied to a typed CAM product."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[PostprocessorConfigT]]

    #: Annotation product kind produced by this postprocessor.
    annotation_kind: ClassVar[str] = ""
    #: Product roles this annotation may be attached to.
    accepted_product_roles: ClassVar[tuple[str, ...]] = POSTPROCESSOR_TARGET_ROLES
    #: Whether the numeric work needs the CAM model.
    requires_model: ClassVar[bool] = False
    #: Whether the numeric work needs a backend.
    requires_backend: ClassVar[bool] = False
    #: The product declaration this postprocessor emits.
    output_products: ClassVar[tuple[ProductDeclaration, ...]] = (
        CAM_ANNOTATION_DECLARATION,
    )

    config: PostprocessorConfigT

    def __init__(
        self, config: PostprocessorConfigT | None = None, **kwargs: Any
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword options, not both")
        if config is None:
            self.config = self.config_schema.model_validate(kwargs)
        elif isinstance(config, self.config_schema):
            # An already-validated config is passed through: dumping and
            # re-validating it would lose an injected object (an observable, for
            # example) that only lives in memory.
            self.config = config
        else:
            self.config = self.config_schema.model_validate(config.model_dump())
        self.result_metadata: dict[str, Any] = {}

    @abstractmethod
    def process(self, product: Dataset, context: CAMAnnotationContext) -> Dataset:
        """Return the annotation product for the given typed product.

        This is the only postprocessor entry point: no result object, no
        ``postprocess`` dictionary and no analysis sidecar survive Phase 10D2.
        """
