"""Closure tests for uncertainty references of the core product schema.

Every ``UncertaintySchema`` attached to a product must reference variables
and sampling bases that actually exist: the target variable, the optional
payload variable (``data_variable``) and the sampling basis. In particular,
a producer that drops an optional uncertainty payload variable from a
concrete product must drop the uncertainty entry too — the schema validator
enforces that closure.
"""

import pytest
from pydantic import ValidationError
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    SamplingBasisSchema,
    UncertaintySchema,
    VariableSchema,
)


def _statistics_schema(
    *,
    variables: list[VariableSchema] | None = None,
    uncertainties: list[UncertaintySchema] | None = None,
) -> ProductSchema:
    """Minimal statistics product with one estimate per (scan, channel)."""
    return ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=3),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=2),
        ],
        sampling_bases=[SamplingBasisSchema(name="trajectory", count=64)],
        variables=(
            variables
            if variables is not None
            else [
                VariableSchema(
                    name="estimate",
                    dtype="float64",
                    value_domain="real",
                    dims=("scan", "channel"),
                ),
                VariableSchema(
                    name="estimate_std",
                    dtype="float64",
                    value_domain="real",
                    dims=("scan", "channel"),
                ),
            ]
        ),
        uncertainties=(
            uncertainties
            if uncertainties is not None
            else [
                UncertaintySchema(
                    target="estimate",
                    kind="sem",
                    sampling_basis="trajectory",
                    covariance="real",
                    scope="sampling",
                    data_variable="estimate_std",
                )
            ]
        ),
    )


def test_closed_uncertainty_references_validate():
    """The positive control: all uncertainty references resolve."""
    schema = _statistics_schema()
    uncertainty = schema.uncertainties[0]
    assert uncertainty.target == "estimate"
    assert uncertainty.data_variable == "estimate_std"
    assert uncertainty.sampling_basis == "trajectory"


def test_uncertainty_target_must_reference_an_existing_variable():
    """An uncertainty describing an absent variable fails the schema."""
    with pytest.raises(ValidationError, match="targets unknown variable"):
        _statistics_schema(
            uncertainties=[
                UncertaintySchema(
                    target="dropped_estimate",
                    kind="sem",
                    sampling_basis="trajectory",
                    covariance="real",
                )
            ]
        )


def test_uncertainty_data_variable_must_reference_an_existing_variable():
    """The uncertainty payload variable must exist in the same product."""
    with pytest.raises(ValidationError, match="unknown data variable"):
        _statistics_schema(
            uncertainties=[
                UncertaintySchema(
                    target="estimate",
                    kind="sem",
                    sampling_basis="trajectory",
                    covariance="real",
                    data_variable="dropped_std",
                )
            ]
        )


def test_uncertainty_sampling_basis_must_exist():
    """The sampling basis an uncertainty counts over must be declared."""
    with pytest.raises(ValidationError, match="unknown sampling basis"):
        _statistics_schema(
            uncertainties=[
                UncertaintySchema(
                    target="estimate",
                    kind="sem",
                    sampling_basis="scan",
                    covariance="real",
                )
            ]
        )


def test_sampling_scope_requires_a_sampling_basis():
    """A sampling-scope uncertainty must name its realization source."""
    with pytest.raises(ValidationError, match="must reference a sampling basis"):
        _statistics_schema(
            uncertainties=[
                UncertaintySchema(
                    target="estimate",
                    kind="sem",
                    covariance="real",
                    scope="sampling",
                )
            ]
        )


def test_dropping_the_payload_variable_requires_dropping_the_uncertainty():
    """Producer closure: dropping an optional uncertainty payload variable
    while keeping the ``UncertaintySchema`` entry must fail; dropping both
    validates.
    """
    estimate_only = [
        VariableSchema(
            name="estimate",
            dtype="float64",
            value_domain="real",
            dims=("scan", "channel"),
        )
    ]
    # Variable dropped, entry kept: rejected.
    with pytest.raises(ValidationError, match="unknown data variable"):
        _statistics_schema(variables=estimate_only)
    # Both dropped: the reduced product is a valid closed schema.
    schema = _statistics_schema(variables=estimate_only, uncertainties=[])
    assert schema.uncertainties == []
    assert schema.is_closed
