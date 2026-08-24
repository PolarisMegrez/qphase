"""qphase: Data Product Schemas
---------------------------------------------------------
Defines the machine-readable schema language of typed data products:
``AxisSchema``, ``VariableSchema``, ``UncertaintySchema`` and
``SamplingBasisSchema`` and ``ProductSchema``. Schemas are JSON-serializable,
strictly extra-forbid, and
carry a stable fingerprint. Shapes may be partially unknown at plan time
(``AxisSchema.size is None``) but must be closed before materialization.

The *variable* — not the dataset class — decides real vs. complex; spectral
products are never split into incompatible real/complex result classes.
Uncertainties of complex variables must declare an explicit covariance
representation. Matrix/tensor variables use named dimensions plus a
symmetry/layout descriptor; object dtypes are forbidden.

Public API
----------
AxisRole
    Statistical role of an axis (parameter/realization/coordinate/...).
AxisSchema
    Named axis with optional size and coordinate description.
SamplingBasisSchema
    Statistical source reduced out of the retained payload axes.
VariableSchema
    Named variable with dtype, value domain, dims and constraints.
VariableConstraints
    Nonnegativity and tensor symmetry/layout constraints.
UncertaintySchema
    Uncertainty attached to a variable.
SpectralAttributes
    Mandatory attribute set of spectral products.
ProductSchema
    Complete product schema with cross-validation and fingerprint.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.utils import canonical_json
from .kinds import DataKind

__all__ = [
    "PRODUCT_SCHEMA_VERSION",
    "AxisRole",
    "AxisSchema",
    "ProductSchema",
    "SamplingBasisSchema",
    "SpectralAttributes",
    "UncertaintySchema",
    "VariableConstraints",
    "VariableSchema",
]

#: Schema identifier frozen by the approved qphase 2.0 Phase 0 contract.
PRODUCT_SCHEMA_VERSION: Literal["qphase.product/1"] = "qphase.product/1"


class AxisRole(str, Enum):
    """Statistical role of an axis.

    ``parameter`` axes are scan/control coordinates: distinct points never
    merge into one estimator's SEM. ``realization`` axes are independent
    stochastic realizations (trajectory, seed, trajectory group, time block)
    that uncertainty merging counts over. ``coordinate`` axes are sampling
    coordinates (time, frequency, tau). ``component`` axes index channels,
    modes or tensor components. ``index`` axes enumerate discrete rows such as
    candidates, paths or moment orders.
    """

    PARAMETER = "parameter"
    REALIZATION = "realization"
    COORDINATE = "coordinate"
    COMPONENT = "component"
    INDEX = "index"


class AxisSchema(BaseModel):
    """One named axis of a data product.

    ``size`` may be unknown at plan time; the axis is *closed* once its size
    is known and — for regular coordinates — a finite, non-zero step is set.
    Explicit coordinate arrays never enter the schema; their length and
    monotonicity are validated at materialization time.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    role: AxisRole = AxisRole.COORDINATE
    size: int | None = Field(
        default=None,
        ge=0,
        description="Axis length; None while unknown at plan time.",
    )
    coordinate: Literal["regular", "explicit"] = Field(
        default="explicit",
        description="'regular' axes are described by start/step; 'explicit' "
        "axes carry materialized coordinates.",
    )
    units: str = ""
    monotonic: bool = True
    start: float | None = None
    step: float | None = None

    @property
    def is_closed(self) -> bool:
        """Return True when the axis is ready for materialization.

        An axis is closed once its size is known and — for regular
        coordinates — its step is finite and non-zero.
        """
        if self.size is None:
            return False
        if self.coordinate == "regular":
            if self.step is None or not np.isfinite(self.step) or self.step == 0:
                return False
        return True


class SamplingBasisSchema(BaseModel):
    """Statistical source used by an uncertainty estimator.

    Product axes describe dimensions retained by at least one payload variable.
    A sampling basis instead describes realizations that have already been
    reduced away, such as trajectories contributing to a mean PSD. If the
    realizations remain in the payload, ``source_axis`` references their
    ``role=realization`` axis. Otherwise a concrete product closes the basis
    with either a fixed ``count`` or an integer ``count_variable``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source_axis: str | None = None
    count: int | None = Field(default=None, gt=0)
    count_variable: str | None = None

    @model_validator(mode="after")
    def _check_count_sources(self) -> SamplingBasisSchema:
        if self.count is not None and self.count_variable is not None:
            raise ValueError(
                "sampling basis must use either count or count_variable, not both"
            )
        return self

    @property
    def is_closed(self) -> bool:
        """Return whether the realized sample count can be determined."""
        return (
            self.source_axis is not None
            or self.count is not None
            or self.count_variable is not None
        )


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
    columns. ``real`` covers any non-complex numeric dtype, including integer
    counts and status codes.
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
            dtype: np.dtype[Any] = np.dtype(value)
        except TypeError as exc:
            raise ValueError(f"unparseable dtype {value!r}") from exc
        if dtype.hasobject:
            raise ValueError("object dtype is not allowed in data products")
        return dtype.str if dtype.metadata is None else value

    @field_validator("dims")
    @classmethod
    def _check_dims_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("a variable must not repeat an axis name")
        return value

    @model_validator(mode="after")
    def _check_domain_dtype_consistency(self) -> VariableSchema:
        dtype: np.dtype[Any] = np.dtype(self.dtype)
        if self.value_domain == "real" and dtype.kind == "c":
            raise ValueError(
                f"variable {self.name!r} declares value_domain 'real' with a "
                "complex dtype"
            )
        if self.value_domain == "complex" and dtype.kind != "c":
            raise ValueError(
                f"variable {self.name!r} declares value_domain 'complex' with "
                "a non-complex dtype"
            )
        if self.constraints.nonnegative and (
            self.value_domain != "real" or dtype.kind not in "fiu"
        ):
            raise ValueError(
                f"variable {self.name!r}: nonnegative constraints only apply "
                "to real numeric variables"
            )
        return self


class UncertaintySchema(BaseModel):
    """Uncertainty attached to one variable of the same product.

    Complex variables must declare an explicit covariance representation; a
    bare complex "std" is not a valid uncertainty. ``sampling_basis`` names
    the retained or reduced realization source the estimate counts over.
    Covariance payloads are
    themselves typed variables (or a separate covariance product) referenced
    by ``data_variable`` — never metadata dicts.
    """

    model_config = ConfigDict(extra="forbid")

    target: str = Field(description="Name of the variable being described.")
    kind: Literal[
        "sample_std", "sem", "confidence_interval", "covariance", "other"
    ]
    sampling_basis: str = Field(
        default="",
        description="Sampling basis the uncertainty counts over.",
    )
    covariance: Literal["real", "real_imag", "magnitude_phase"] | None = None
    scope: str | None = Field(
        default=None,
        description="Resource-defined uncertainty scope identifier, e.g. "
        "'conditional' or 'sampling'.",
    )
    data_variable: str | None = Field(
        default=None,
        description="Typed variable carrying the covariance/interval payload.",
    )
    confidence: float | None = None
    count: int | None = Field(
        default=None, description="Number of independent realizations."
    )

    @field_validator("confidence")
    @classmethod
    def _check_confidence(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError("confidence must satisfy 0 < confidence < 1")
        return value

    @field_validator("count")
    @classmethod
    def _check_count(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("count must be a positive integer")
        return value


class SpectralAttributes(BaseModel):
    """Mandatory attribute set of spectral products.

    All declared fields are required and must be non-empty; empty-string
    defaults must not bypass validation. Resource packages may add extra keys
    (JSON-serializable); attributes participate in the product fingerprint.
    """

    model_config = ConfigDict(extra="allow")

    frequency_units: str = Field(min_length=1)
    orientation: str = Field(
        min_length=1,
        description="Frequency orientation convention identifier; values are "
        "defined by the owning resource package.",
    )
    sidedness: Literal["one_sided", "two_sided"]
    normalization: str = Field(min_length=1)
    window: str = Field(
        min_length=1,
        description="Window identifier; use 'rectangular' when no window is "
        "applied.",
    )
    estimator: str = Field(min_length=1)
    effective_degrees_of_freedom: float | None = None


class ProductSchema(BaseModel):
    """Complete, fingerprintable schema of one data product."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["qphase.product/1"] = PRODUCT_SCHEMA_VERSION
    kind: DataKind
    axes: list[AxisSchema] = Field(default_factory=list)
    sampling_bases: list[SamplingBasisSchema] = Field(default_factory=list)
    variables: list[VariableSchema] = Field(min_length=1)
    uncertainties: list[UncertaintySchema] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

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
        axis_roles = {axis.name: axis.role for axis in self.axes}
        sampling_basis_names = {basis.name for basis in self.sampling_bases}
        if len(sampling_basis_names) != len(self.sampling_bases):
            raise ValueError("sampling basis names must be unique")
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
            if variable.constraints.symmetry == "hermitian":
                tensor_dims = [
                    dim
                    for dim in variable.dims
                    if axis_roles[dim]
                    in {AxisRole.COMPONENT, AxisRole.INDEX}
                ]
                if len(tensor_dims) < 2:
                    raise ValueError(
                        f"variable {variable.name!r}: a Hermitian layout "
                        "requires at least two component/index dims"
                    )
        variable_by_name = {variable.name: variable for variable in self.variables}
        retained_dims = {dim for variable in self.variables for dim in variable.dims}
        unused_realization_axes = {
            name
            for name, role in axis_roles.items()
            if role == AxisRole.REALIZATION and name not in retained_dims
        }
        if unused_realization_axes:
            raise ValueError(
                "realization axes must be retained by at least one variable; "
                f"reduced realizations belong in sampling_bases: "
                f"{sorted(unused_realization_axes)}"
            )

        for basis in self.sampling_bases:
            if basis.source_axis is not None:
                role = axis_roles.get(basis.source_axis)
                if role is None:
                    raise ValueError(
                        f"sampling basis {basis.name!r} references unknown source "
                        f"axis {basis.source_axis!r}"
                    )
                if role != AxisRole.REALIZATION:
                    raise ValueError(
                        f"sampling basis {basis.name!r} source axis "
                        f"{basis.source_axis!r} is not a realization axis"
                    )
            if basis.count_variable is not None:
                count_variable = variable_by_name.get(basis.count_variable)
                if count_variable is None:
                    raise ValueError(
                        f"sampling basis {basis.name!r} references unknown count "
                        f"variable {basis.count_variable!r}"
                    )
                if np.dtype(count_variable.dtype).kind not in "iu":
                    raise ValueError(
                        f"sampling basis {basis.name!r} count variable must have "
                        "an integer dtype"
                    )
        domains = {v.name: v.value_domain for v in self.variables}
        for uncertainty in self.uncertainties:
            if uncertainty.target not in variable_names:
                raise ValueError(
                    f"uncertainty targets unknown variable "
                    f"{uncertainty.target!r}"
                )
            if domains[uncertainty.target] == "complex":
                if uncertainty.covariance not in {"real_imag", "magnitude_phase"}:
                    raise ValueError(
                        f"uncertainty of complex variable "
                        f"{uncertainty.target!r} must use the 'real_imag' or "
                        "'magnitude_phase' covariance representation"
                    )
            elif uncertainty.covariance not in {None, "real"}:
                raise ValueError(
                    f"uncertainty of real variable {uncertainty.target!r} "
                    "must use the 'real' covariance representation"
                )
            if uncertainty.sampling_basis:
                if uncertainty.sampling_basis not in sampling_basis_names:
                    raise ValueError(
                        f"uncertainty of {uncertainty.target!r} references "
                        f"unknown sampling basis {uncertainty.sampling_basis!r}"
                    )
            elif uncertainty.scope == "sampling":
                raise ValueError(
                    f"sampling uncertainty of {uncertainty.target!r} must "
                    "reference a sampling basis"
                )
            if (
                uncertainty.data_variable is not None
                and uncertainty.data_variable not in variable_names
            ):
                raise ValueError(
                    f"uncertainty of {uncertainty.target!r} references "
                    f"unknown data variable {uncertainty.data_variable!r}"
                )

        if self.kind == DataKind.SPECTRAL:
            # Raises if the mandatory spectral attributes are absent.
            SpectralAttributes.model_validate(self.attributes)
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
        """Return True when every axis is closed (materializable)."""
        return all(axis.is_closed for axis in self.axes) and all(
            basis.is_closed for basis in self.sampling_bases
        )

    def fingerprint(self) -> str:
        """Return the stable content hash of this schema."""
        payload = self.model_dump(mode="json")
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
