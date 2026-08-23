"""qphase_sde: SDE Quantity Contracts (2.0)
---------------------------------------------------------
Freezes the public quantity identifiers and product schema templates of the
SDE resource package, plus the single canonical frequency-orientation
convention. These contracts are pure declarations: they must never import
concrete plugins, models or backend implementations.

The frequency orientation convention is *defined* here; the 1.x helper module
``qphase_sde.analyser.frequency_orientation`` remains the runtime
implementation until Phase 2 re-points it at this canonical definition.

Public API
----------
SDEQuantity
    SDE-specific quantity identifiers (extending core spectral quantities).
FrequencyOrientation
    Canonical orientation values.
DEFAULT_FREQUENCY_ORIENTATION
    Default orientation for 2.x products.
SPECTRUM_PRODUCT, ALLAN_PRODUCT, MOMENT_FAMILY_PRODUCT
    Open product schema templates referenced by the resource manifest.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from qphase.data import (
    AxisSchema,
    DataKind,
    MomentFamilySchema,
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


#: Open (plan-time) schema of the SDE spectral product. Axis sizes close when
#: an estimator is planned against a concrete time grid.
SPECTRUM_PRODUCT = ProductSchema(
    kind=DataKind.SPECTRAL,
    axes=[
        AxisSchema(name="scan", independent=True),
        AxisSchema(name="statistic", independent=True),
        AxisSchema(name="frequency", units="inverse_time"),
        AxisSchema(name="channel"),
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
            independent_unit="statistic",
            covariance="real",
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

#: Open schema of the Allan variance product.
ALLAN_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", independent=True),
        AxisSchema(name="tau", units="inverse_time"),
        AxisSchema(name="channel"),
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
            independent_unit="scan",
            covariance="real",
        )
    ],
    attributes={"estimator": "non_overlapping_windows"},
)

#: Open schema of a moment-family statistics product.
MOMENT_FAMILY_PRODUCT = ProductSchema(
    kind=DataKind.STATISTICS,
    axes=[
        AxisSchema(name="scan", independent=True),
        AxisSchema(name="order"),
        AxisSchema(name="channel_i"),
        AxisSchema(name="channel_j"),
    ],
    variables=[
        VariableSchema(
            name="moment",
            dtype="complex128",
            value_domain="complex",
            dims=("scan", "order", "channel_i", "channel_j"),
            quantity=SDEQuantity.MOMENTS.value,
            constraints=VariableConstraints(symmetry="symmetric"),
        )
    ],
    uncertainties=[
        UncertaintySchema(
            target="moment",
            kind="sample_std",
            independent_unit="scan",
            covariance="real_imag",
        )
    ],
    moment_family=MomentFamilySchema(
        family_id="sde-moments",
        moment_kind="raw",
        ordering="c_number",
        maximum_order=4,
        symmetry="symmetric",
    ),
)
