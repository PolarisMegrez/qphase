"""qphase_sde: SDE Data Bundle Contract (2.0)
---------------------------------------------------------
Freezes the ``SDEDataBundle`` contract: a logical SDE job returns a bundle of
named data products plus provenance. The bundle is a catalog — it never copies
arrays into itself. Downstream jobs select products by kind/quantity/fields,
not by internal labels.

Also freezes the SDE trajectory semantics every time-series product must
declare: channel definitions, independent-realization axis, RNG mapping,
warm-up and mask provenance.

Public API
----------
SDEProvenance
    JSON-serializable provenance of an SDE job.
TRAJECTORY_PRODUCT
    Open schema template of the trajectory product.
SDEDataBundleProtocol
    Structural contract of the data bundle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from qphase.data import AxisSchema, DataKind, ProductSchema, VariableSchema

from .quantities import DEFAULT_FREQUENCY_ORIENTATION, SDEQuantity

__all__ = [
    "TRAJECTORY_PRODUCT",
    "SDEDataBundleProtocol",
    "SDEProvenance",
]

#: Canonical product catalog labels of an SDE data bundle. Labels are
#: job-local conveniences; dependency selection never treats them as contract.
BUNDLE_PRODUCT_LABELS = (
    "trajectories",
    "spectrum",
    "spectral_peaks",
    "coherence_frequency",
    "allan_variance",
    "moments",
    "distributions",
    "first_passage",
    "fit",
)


class SDEProvenance(BaseModel):
    """Provenance every SDE data product carries.

    The RNG mapping rule (master seed + scan index + trajectory id) is frozen
    so re-tiling or re-batching never changes realizations.
    """

    model_config = ConfigDict(extra="forbid")

    t0: float = 0.0
    dt: float | None = None
    saved_samples: int | None = None
    warmup_samples: int = 0
    master_seed: int | None = None
    rng_mapping: str = "master_seed+scan_index+trajectory_id"
    frequency_orientation: str = DEFAULT_FREQUENCY_ORIENTATION
    model_fingerprint: str = ""
    integrator_fingerprint: str = ""
    scan_grid: dict[str, list[float]] = Field(default_factory=dict)


#: Open schema template of the SDE trajectory product.
TRAJECTORY_PRODUCT = ProductSchema(
    kind=DataKind.TIME_SERIES,
    axes=[
        AxisSchema(name="scan", independent=True),
        AxisSchema(name="trajectory", independent=True),
        AxisSchema(name="time", coordinate="regular", units="inverse_rate"),
        AxisSchema(name="channel"),
    ],
    variables=[
        VariableSchema(
            name="alpha",
            dtype="complex128",
            value_domain="complex",
            dims=("scan", "trajectory", "time", "channel"),
            quantity=SDEQuantity.FIELD_AMPLITUDE.value,
        ),
        VariableSchema(
            name="valid_length",
            dtype="int64",
            value_domain="real",
            dims=("scan", "trajectory"),
            quantity="valid_length",
        ),
    ],
    attributes={
        "frequency_orientation": DEFAULT_FREQUENCY_ORIENTATION,
        "independent_realizations": "trajectory",
        "channel_definition": "mode",
    },
)


@runtime_checkable
class SDEDataBundleProtocol(Protocol):
    """Structural contract of a logical SDE job's output bundle."""

    @property
    def products(self) -> Mapping[str, Any]:
        """Named data products (values satisfy the core DataProduct protocol)."""
        ...

    @property
    def provenance(self) -> SDEProvenance:
        """Job-level provenance shared by all products."""
        ...

    def require(
        self,
        *,
        kind: DataKind | None = None,
        quantity: str | None = None,
        fields: tuple[str, ...] = (),
    ) -> Mapping[str, Any]:
        """Select products by kind/quantity/fields, never by label alone."""
        ...
