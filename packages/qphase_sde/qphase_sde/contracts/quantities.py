"""qphase_sde: SDE Quantity Contracts (2.0)
---------------------------------------------------------
Freezes the public quantity identifiers and product schema templates of the
SDE resource package, plus the single canonical frequency-orientation
convention. These contracts are pure declarations: they must never import
concrete plugins, models or backend implementations.

Axis roles follow the core contract: ``scan`` is a *parameter* axis (it is
swept, not sampled), and statistical uncertainties always count over a
*realization* axis (``trajectory``) — never over the scan axis.

The frequency orientation convention is *defined* here; the 1.x helper module
``qphase_sde.analyser.frequency_orientation`` remains the runtime
implementation until Phase 2 re-points it at this canonical definition.

Moment families are SDE-private: the core schema has no ``moment_family``
field, so the versioned :class:`SDEMomentFamilySchema` descriptor is embedded
into the product's ``attributes``. Only moments with a single explicit
``order`` index axis and fixed remaining dims are frozen; arbitrary
mixed-rank moment tensors are deliberately *not* claimed by this schema.

Public API
----------
SDEQuantity
    SDE-specific quantity identifiers (extending core spectral quantities).
FrequencyOrientation
    Canonical orientation values.
DEFAULT_FREQUENCY_ORIENTATION
    Default orientation for 2.x products.
SDEMomentFamilySchema
    SDE-private, versioned moment family descriptor.
SPECTRUM_PRODUCT, ALLAN_PRODUCT, MOMENT_FAMILY_PRODUCT
    Open product schema templates referenced by the resource manifest.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    SpectralQuantity,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
)

__all__ = [
    "ALLAN_PRODUCT",
    "DEFAULT_FREQUENCY_ORIENTATION",
    "LEGACY_FREQUENCY_ORIENTATION",
    "MOMENT_FAMILY_PRODUCT",
    "SPECTRUM_PRODUCT",
    "FrequencyOrientation",
    "SDEMomentFamilySchema",
    "SDEQuantity",
    "SpectralQuantity",
]


#: Canonical frequency orientation values (mirrored by the 1.x runtime module
#: ``qphase_sde.analyser.frequency_orientation`` until Phase 2).
FrequencyOrientation = Literal["phase_decreasing", "phase_increasing"]

DEFAULT_FREQUENCY_ORIENTATION: FrequencyOrientation = "phase_decreasing"
LEGACY_FREQUENCY_ORIENTATION: FrequencyOrientation = "phase_increasing"


class SDEQuantity(str, Enum):
    """SDE-specific quantity identifiers for time-domain and statistics data.

    Spectral quantities reuse the core ``SpectralQuantity`` enumeration; this
    enum only adds identifiers the core contract deliberately does not own.
    """

    FIELD_AMPLITUDE = "field_amplitude"
    INTENSITY = "intensity"
    PHASE = "phase"
    ALLAN_VARIANCE = "allan_variance"
    MOMENTS = "moments"
    DISTRIBUTION = "distribution"
    FIRST_PASSAGE = "first_passage"
    PEAK_LOCATION = "peak_location"
    COHERENCE_FREQUENCY = "coherence_frequency"
    FIT_PARAMETERS = "fit_parameters"


class SDEMomentFamilySchema(BaseModel):
    """SDE-private descriptor of one moment family.

    Only moments with a single explicit ``order`` index axis are frozen: all
    orders share the remaining dims (e.g. ``moment[scan, order, channel]`` for
    per-channel occupation moments). ``orders`` lists the explicit positive
    integer order coordinates. Arbitrary mixed-rank moment tensors (where the
    tensor rank grows with the order) are *not* representable and must wait
    for a dedicated design with a real consumer.
    """

    model_config = ConfigDict(extra="forbid")

    family_id: str = Field(min_length=1)
    moment_kind: Literal["raw", "central", "cumulant", "factorial"]
    ordering: Literal["c_number", "normal", "symmetric"]
    order_axis: Literal["order"] = "order"
    orders: list[int] = Field(
        min_length=1,
        description="Explicit positive integer order coordinates.",
    )

    @field_validator("orders")
    @classmethod
    def _check_orders(cls, value: list[int]) -> list[int]:
        for order in value:
            if order <= 0:
                raise ValueError("moment orders must be positive integers")
        if len(set(value)) != len(value):
            raise ValueError("moment orders must be unique")
        return sorted(value)


#: Open (plan-time) schema of the SDE spectral product. Axis sizes close when
#: an estimator is planned against a concrete time grid. Uncertainties count
#: over the ``trajectory`` realization axis, never over ``scan``.
SPECTRUM_PRODUCT = ProductSchema(
    kind=DataKind.SPECTRAL,
    axes=[
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="trajectory", role=AxisRole.REALIZATION),
        AxisSchema(
            name="frequency", role=AxisRole.COORDINATE, units="inverse_time"
        ),
        AxisSchema(name="channel", role=AxisRole.COMPONENT),
    ],
    variables=[
        VariableSchema(
            name="amplitude",
            dtype="complex128",
            value_domain="complex",
            dims=("scan", "frequency", "channel"),
            quantity=SpectralQuantity.FOURIER_AMPLITUDE.value,
        ),
        VariableSchema(
            name="power",
            dtype="float64",
            value_domain="real",
            dims=("scan", "frequency", "channel"),
            quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
            constraints=VariableConstraints(nonnegative=True),
        ),
    ],
    uncertainties=[
        UncertaintySchema(
            target="power",
            kind="sample_std",
            independent_unit="trajectory",
            covariance="real",
            scope="sampling",
        )
    ],
    attributes={
        "frequency_units": "inverse_time",
        "orientation": DEFAULT_FREQUENCY_ORIENTATION,
        "sidedness": "one_sided",
        "normalization": "density",
        "window": "hann",
        "estimator": "periodogram",
    },
)

#: Open schema of the Allan variance product. The sampling uncertainty counts
#: over independent trajectories — the scan axis is a swept parameter and must
#: never be treated as an ensemble of independent samples.
ALLAN_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="trajectory", role=AxisRole.REALIZATION),
        AxisSchema(name="tau", role=AxisRole.COORDINATE, units="inverse_time"),
        AxisSchema(name="channel", role=AxisRole.COMPONENT),
    ],
    variables=[
        VariableSchema(
            name="allan_variance",
            dtype="float64",
            value_domain="real",
            dims=("scan", "tau", "channel"),
            quantity=SDEQuantity.ALLAN_VARIANCE.value,
            constraints=VariableConstraints(nonnegative=True),
        )
    ],
    uncertainties=[
        UncertaintySchema(
            target="allan_variance",
            kind="sample_std",
            independent_unit="trajectory",
            covariance="real",
            scope="sampling",
        )
    ],
    attributes={"estimator": "non_overlapping_windows"},
)

#: Default SDE moment family descriptor, embedded into the product attributes.
DEFAULT_MOMENT_FAMILY = SDEMomentFamilySchema(
    family_id="sde-channel-moments",
    moment_kind="raw",
    ordering="c_number",
    orders=[1, 2, 3, 4],
)

#: Open schema of a moment-family statistics product: per-channel moments with
#: one explicit ``order`` index axis. The SDE-private family descriptor travels
#: in ``attributes``; the core schema has no ``moment_family`` field.
MOMENT_FAMILY_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", role=AxisRole.PARAMETER),
        AxisSchema(name="trajectory", role=AxisRole.REALIZATION),
        AxisSchema(name="order", role=AxisRole.INDEX),
        AxisSchema(name="channel", role=AxisRole.COMPONENT),
    ],
    variables=[
        VariableSchema(
            name="moment",
            dtype="complex128",
            value_domain="complex",
            dims=("scan", "order", "channel"),
            quantity=SDEQuantity.MOMENTS.value,
        )
    ],
    uncertainties=[
        UncertaintySchema(
            target="moment",
            kind="sample_std",
            independent_unit="trajectory",
            covariance="real_imag",
            scope="sampling",
        )
    ],
    attributes={
        "moment_family": DEFAULT_MOMENT_FAMILY.model_dump(mode="json"),
    },
)
