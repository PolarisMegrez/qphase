"""qphase: Data Product Schemas
---------------------------------------------------------
Defines the machine-readable schema language of typed data products:
``AxisSchema``, ``VariableSchema``, ``UncertaintySchema``, ``ProductSchema``
and ``MomentFamilySchema``. Schemas are JSON-serializable, strictly
extra-forbid, and carry a stable fingerprint. Shapes may be partially unknown
at plan time (``AxisSchema.size is None``) but must be closed before
materialization.

The *variable* — not the dataset class — decides real vs. complex; spectral
products are never split into incompatible real/complex result classes.
Uncertainties of complex variables must declare an explicit covariance
representation. Matrix/tensor variables use named dimensions plus a
symmetry/layout descriptor; object dtypes are forbidden.

Public API
----------
AxisSchema
    Named axis with optional size and coordinate description.
VariableSchema
    Named variable with dtype, value domain, dims and constraints.
VariableConstraints
    Nonnegativity and tensor symmetry/layout constraints.
UncertaintySchema
    Uncertainty attached to a variable.
SpectralAttributes
    Mandatory attribute set of spectral products.
MomentFamilySchema
    Grouping of related moments in one statistics product.
ProductSchema
    Complete product schema with cross-validation and fingerprint.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.utils import canonical_json
from .kinds import DataKind

__all__ = [
    "PRODUCT_SCHEMA_VERSION",
    "AxisSchema",
    "MomentFamilySchema",
    "ProductSchema",
    "SpectralAttributes",
    "UncertaintySchema",
    "VariableConstraints",
    "VariableSchema",
]

#: Schema identifier frozen for the qphase 2.0 product schema contract.
PRODUCT_SCHEMA_VERSION = "qphase.product/1"


class AxisSchema(BaseModel):
    """One named axis of a data product.

    ``size`` may be unknown at plan time; the axis is *closed* once its size is
    known. ``independent`` marks realization axes (scan points, trajectories)
    that uncertainty merging counts over.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    size: int | None = Field(
        default=None, description="Axis length; None while unknown at plan time."
    )
    coordinate: Literal["regular", "explicit"] = Field(
        default="explicit",
        description="'regular' axes are described by start/step; 'explicit' "
        "axes carry materialized coordinates.",
    )
    units: str = ""
    monotonic: bool = True
    independent: bool = Field(
        default=False,
        description="True for realization axes (scan/trajectory).",
    )
    start: float | None = None
    step: float | None = None

    @model_validator(mode="after")
    def _check_regular_coordinate(self) -> AxisSchema:
        if self.coordinate == "regular" and self.step is None:
            raise ValueError(
                f"regular axis {self.name!r} must declare a step"
            )
        return self

    @property
    def is_closed(self) -> bool:
        """Return True when the axis size is known."""
        return self.size is not None


class VariableConstraints(BaseModel):
    """Value and layout constraints of a variable."""

    model_config = ConfigDict(extra="forbid")

    nonnegative: bool = False
    symmetry: Literal["symmetric", "hermitian"] | None = None
    layout: Literal["dense", "upper_triangular", "lower_triangular", "diagonal"] = (
        "dense"
    )


class VariableSchema(BaseModel):
    """One named variable of a data product.

    The variable decides the real/complex value domain; matrices and tensors
    use named ``dims`` plus a symmetry/layout constraint. Object dtypes are
    forbidden — structured tables belong to statistics products with typed
    columns.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str = Field(description="NumPy-compatible dtype name.")
    value_domain: Literal["real", "complex"]
    dims: tuple[str, ...] = Field(
        default_factory=tuple, description="Axis names this variable spans."
    )
    quantity: str = Field(
        default="",
        description="Physical quantity identifier (resource-defined), e.g. "
        "'power_spectral_density'.",
    )
    units: str = ""
    constraints: VariableConstraints = Field(default_factory=VariableConstraints)

    @field_validator("dtype")
    @classmethod
    def _check_dtype(cls, value: str) -> str:
        try:
            dtype = np.dtype(value)
        except TypeError as exc:
            raise ValueError(f"unparseable dtype {value!r}") from exc
        if dtype.hasobject:
            raise ValueError("object dtype is not allowed in data products")
        return dtype.str if dtype.metadata is None else value

    @model_validator(mode="after")
    def _check_domain_dtype_consistency(self) -> VariableSchema:
        dtype = np.dtype(self.dtype)
        if self.value_domain == "real" and dtype.kind == "c":
            raise ValueError(
                f"variable {self.name!r} declares value_domain 'real' with a "
                "complex dtype"
            )
        return self


class UncertaintySchema(BaseModel):
    """Uncertainty attached to one variable of the same product.

    Complex variables must declare an explicit covariance representation; a
    bare complex "std" is not a valid uncertainty. ``independent_unit`` names
    the realization axis the estimate counts over.
    """

    model_config = ConfigDict(extra="forbid")

    target: str = Field(description="Name of the variable being described.")
    kind: Literal[
        "sample_std", "sem", "confidence_interval", "covariance", "other"
    ]
    independent_unit: str = Field(
        default="",
        description="Realization axis the uncertainty counts over.",
    )
    covariance: Literal["real", "real_imag", "magnitude_phase", "custom"] | None = (
        None
    )
    confidence: float | None = None
    count: int | None = Field(
        default=None, description="Number of independent realizations."
    )


class SpectralAttributes(BaseModel):
    """Mandatory attribute set of spectral products.

    Extra keys are allowed so resource packages can extend the set, but the
    fields declared here must always be present for ``spectral`` products.
    """

    model_config = ConfigDict(extra="allow")

    frequency_units: str = ""
    orientation: str = Field(
        default="",
        description="Frequency orientation convention identifier; values are "
        "defined by the owning resource package.",
    )
    sidedness: Literal["one_sided", "two_sided"]
    normalization: str
    window: str = ""
    estimator: str
    effective_degrees_of_freedom: float | None = None


class MomentFamilySchema(BaseModel):
    """Grouping of related moments stored as one statistics product.

    Moments of one family share independent counts and joint covariance; they
    split into separate products only when orders come from different
    populations, estimators or provenance. Moment order is an axis/variable
    attribute, never a new dataset class.
    """

    model_config = ConfigDict(extra="forbid")

    family_id: str
    moment_kind: Literal["raw", "central", "cumulant", "factorial"]
    ordering: Literal["c_number", "normal", "symmetric"]
    maximum_order: int = Field(ge=1)
    symmetry: Literal["symmetric", "hermitian"] | None = None
    layout: Literal["dense", "upper_triangular", "lower_triangular", "diagonal"] = (
        "dense"
    )


class ProductSchema(BaseModel):
    """Complete, fingerprintable schema of one data product."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["qphase.product/1"] = PRODUCT_SCHEMA_VERSION
    kind: DataKind
    axes: list[AxisSchema] = Field(default_factory=list)
    variables: list[VariableSchema] = Field(min_length=1)
    uncertainties: list[UncertaintySchema] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    moment_family: MomentFamilySchema | None = None

    @field_validator("attributes")
    @classmethod
    def _check_attributes_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "product attributes must be JSON-serializable"
            ) from exc
        return value

    @model_validator(mode="after")
    def _check_references(self) -> ProductSchema:
        axis_names = {axis.name for axis in self.axes}
        if len(axis_names) != len(self.axes):
            raise ValueError("axis names must be unique")
        variable_names = {variable.name for variable in self.variables}
        if len(variable_names) != len(self.variables):
            raise ValueError("variable names must be unique")

        for variable in self.variables:
            unknown_dims = set(variable.dims) - axis_names
            if unknown_dims:
                raise ValueError(
                    f"variable {variable.name!r} references unknown axes "
                    f"{sorted(unknown_dims)}"
                )
        domains = {v.name: v.value_domain for v in self.variables}
        for uncertainty in self.uncertainties:
            if uncertainty.target not in variable_names:
                raise ValueError(
                    f"uncertainty targets unknown variable "
                    f"{uncertainty.target!r}"
                )
            if (
                domains[uncertainty.target] == "complex"
                and uncertainty.covariance is None
            ):
                raise ValueError(
                    f"uncertainty of complex variable {uncertainty.target!r} "
                    "must declare a covariance representation"
                )
            if (
                uncertainty.independent_unit
                and uncertainty.independent_unit not in axis_names
            ):
                raise ValueError(
                    f"uncertainty of {uncertainty.target!r} counts over "
                    f"unknown axis {uncertainty.independent_unit!r}"
                )

        if self.kind == DataKind.SPECTRAL:
            # Raises if the mandatory spectral attributes are absent.
            SpectralAttributes.model_validate(self.attributes)
        if self.moment_family is not None and self.kind != DataKind.STATISTICS:
            raise ValueError(
                "moment families are only valid for statistics products"
            )
        return self

    def axis(self, name: str) -> AxisSchema:
        """Return the axis with the given name."""
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(f"unknown axis {name!r}")

    def variable(self, name: str) -> VariableSchema:
        """Return the variable with the given name."""
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(f"unknown variable {name!r}")

    @property
    def is_closed(self) -> bool:
        """Return True when every axis size is known (materializable)."""
        return all(axis.is_closed for axis in self.axes)

    def fingerprint(self) -> str:
        """Return the stable content hash of this schema."""
        payload = self.model_dump(mode="json")
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
