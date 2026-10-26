"""The public polynomial reduction, on the established one- and two-mode toys.

The one-mode toy (``test_nonlinear``) has ``J(v) = -1 - 3v²``, a unit diffusion and a
physical half-line ``v >= 0``; the two-mode toy (``test_effective_drift``) has four
canonical coordinates and, along its real off-diagonal coordinate, a *bounded* physical
interval ``|x| <= sqrt(R00 R11)``. Both give a projected diffusion and a readout
that are known in closed form, so the fits, the interval and the quadrature can be
checked against the formulas instead of against a stored run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qphase_cam.nonlinear import (
    NonlinearManifold,
    ScalarReconstructionConfig,
    evaluate_generator,
    slow_manifold,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.polynomial_reconstruction import (
    LOCAL_EXTENT_FACTOR,
    LOCAL_INTERVAL_FRACTION,
    MINIMUM_LOCAL_POINTS,
    PolynomialReconstruction,
    PolynomialReconstructionConfig,
    polynomial_reconstruction,
)

from tests.qphase_cam import test_effective_drift as curve_toy
from tests.qphase_cam import test_nonlinear as one_mode

SMALL = {
    "physical_half_width": 3.0,
    "physical_scan_points": 601,
    "fit_points": 21,
    "dense_points": 401,
}


def _one_mode_sigma(manifold: NonlinearManifold, diffusion: float = 1.0) -> float:
    """Return the soft-mode width ``sqrt(q0 / (2 rate))`` of the one-mode toy."""
    projected = 2.0 * diffusion * one_mode.X0
    return float(np.sqrt(projected / (2.0 * manifold.soft_rate)))


def _one_mode() -> tuple[Any, Any, NonlinearManifold, float]:
    adapter = one_mode._QuadraticAdapter()
    manifold = slow_manifold(adapter, [one_mode.X0], {}, order=3)
    return adapter, one_mode._UnitModel(), manifold, _one_mode_sigma(manifold)


def _off_diagonal_manifold() -> NonlinearManifold:
    """Return a manifold along the real off-diagonal coordinate of the two-mode toy."""
    return NonlinearManifold(
        order=3,
        soft_rate=1.0,
        critical_eigenvalue_real=-1.0,
        critical_eigenvalue_imag=0.0,
        noncritical_spectral_gap=None,
        right=(0.0, 0.0, 1.0, 0.0),
        left=(0.0, 0.0, 1.0, 0.0),
        embeddings=((0.0, 0.0, 1.0, 0.0),),
        drift=(0.0, -1.0, 0.0, 0.0),
        maximum_bordered_condition=1.0,
        confining=True,
        method={"service": "test fixture"},
    )


def test_the_one_mode_reduction_reproduces_the_closed_forms():
    adapter, model, manifold, sigma = _one_mode()
    config = PolynomialReconstructionConfig(**SMALL)
    result = polynomial_reconstruction(
        adapter,
        model,
        one_mode._observable(),
        state=[one_mode.X0],
        params={},
        manifold=manifold,
        sigma=sigma,
        config=config,
    )
    assert isinstance(result, PolynomialReconstruction)
    assert result.status is CAMOutcomeStatus.VALID
    assert result.reason is None and result.reason_code is None
    assert result.order == 3
    assert result.sigma == pytest.approx(sigma)
    # The physical half-line v >= 0 is reached on the left and cut by the scan on the
    # right, which the result reports instead of pretending the interval is bounded.
    spacing = (
        2.0 * config.physical_half_width * sigma / (config.physical_scan_points - 1)
    )
    assert result.lower == pytest.approx(-one_mode.X0, abs=spacing)
    assert result.upper == pytest.approx(config.physical_half_width * sigma, abs=1e-12)
    assert result.domain_clipped is True
    assert 0.0 < result.physical_grid_fraction < 1.0
    expected_fraction = (config.physical_half_width * sigma + one_mode.X0) / (
        2.0 * config.physical_half_width * sigma
    )
    assert result.physical_grid_fraction == pytest.approx(expected_fraction, abs=1e-2)

    # q(x) = 2 D v = 2 (0.5 + x) and omega(x) = 0.5 + x are linear, so the fits are
    # exact and their coefficients are the scaled closed forms.
    assert result.diffusion_coefficients[0] == pytest.approx(
        2.0 * one_mode.X0, rel=1e-12
    )
    assert result.diffusion_coefficients[1] == pytest.approx(2.0 * sigma, rel=1e-12)
    assert all(abs(value) < 1e-12 for value in result.diffusion_coefficients[2:])
    assert result.diffusion_error < 1e-12
    assert result.frequency_coefficients[0] == pytest.approx(one_mode.X0, rel=1e-12)
    assert result.frequency_coefficients[1] == pytest.approx(sigma, rel=1e-12)
    assert result.frequency_error < 1e-12
    assert result.diffusion_minimum == pytest.approx(
        result.diffusion_coefficients[0] + 2.0 * sigma * result.lower / sigma, rel=1e-9
    )

    table = result.generator_table
    assert table is not None
    assert len(table.x) == config.dense_points
    assert len(table.drift) == len(table.x)
    x = np.asarray(table.x, dtype=float)
    drift = np.asarray(table.drift, dtype=float)
    # The drift polynomial has no constant term, so it crosses zero at x = 0.
    assert float(np.polynomial.polynomial.polyval(0.0, manifold.drift)) == 0.0
    assert float(np.min(np.abs(drift))) < 1e-2
    np.testing.assert_allclose(
        drift, np.polynomial.polynomial.polyval(x, manifold.drift), rtol=0.0, atol=0.0
    )
    assert min(table.diffusion) > 0.0

    evaluation = result.evaluation
    assert evaluation is not None
    assert evaluation.normalizable is True
    shared = evaluate_generator(table, config=config.quadrature)
    assert evaluation.scalar_intensity == shared.scalar_intensity
    assert evaluation.core_point_count == shared.core_point_count
    assert evaluation.edge_point_count == shared.edge_point_count
    summary = result.summary
    assert summary["scalar"] == evaluation.scalar_intensity
    assert summary["drift_highest"] == pytest.approx(manifold.drift[3])
    assert summary["bordered_condition"] == pytest.approx(
        manifold.maximum_bordered_condition
    )
    assert summary["edge_mass_fraction"] == evaluation.edge_mass_fraction
    assert summary["edge_density_ratio"] == evaluation.endpoint_density_ratio
    assert (
        summary["minimum_projected_diffusion"] == evaluation.minimum_projected_diffusion
    )
    assert summary["core_point_count"] == evaluation.core_point_count
    assert summary["edge_point_count"] == max(
        2, int(np.ceil(0.02 * config.dense_points))
    )
    assert summary["dense_point_count"] == config.dense_points
    assert summary["fit_point_count"] == config.fit_points
    # The intensity is the shared centred-CDF formula on the polynomial table.
    from scipy.integrate import cumulative_trapezoid

    q = np.asarray(table.diffusion, dtype=float)
    omega = np.asarray(table.readout, dtype=float)
    potential = cumulative_trapezoid(2.0 * drift / q, x, initial=0.0)
    log_density = potential - np.log(q)
    log_density -= float(np.max(log_density))
    density = np.exp(np.clip(log_density, -745.0, 0.0))
    density /= float(np.trapezoid(density, x))
    mean = float(np.trapezoid(omega * density, x))
    cumulative = cumulative_trapezoid((omega - mean) * density, x, initial=0.0)
    distribution = cumulative_trapezoid(density, x, initial=0.0)
    cumulative = cumulative - distribution * cumulative[-1]
    retained = density > float(np.max(density)) * 1.0e-14
    expected_intensity = float(
        4.0
        * np.trapezoid(
            cumulative[retained] ** 2 / (q[retained] * density[retained]), x[retained]
        )
    )
    assert summary["scalar"] == pytest.approx(expected_intensity, rel=1e-13)
    # The method block states the configuration and the fit conventions.
    method = result.method
    assert method["quadrature_config"]["intensity_integration"] == "centered_cdf"
    assert method["quadrature_config"]["core_density_comparison"] == "greater"
    assert method["quadrature_config"]["edge_count_rule"] == "ceil_min_two"
    assert method["quadrature_config"]["density_floor_relative"] == pytest.approx(1e-14)
    assert method["local_extent_factor"] == pytest.approx(LOCAL_EXTENT_FACTOR)
    assert method["local_interval_fraction"] == pytest.approx(LOCAL_INTERVAL_FRACTION)
    assert method["minimum_local_points"] == MINIMUM_LOCAL_POINTS
    assert method["order"] == 3


def test_a_bounded_physical_interval_is_not_clipped():
    adapter = curve_toy._CurveAdapter()
    model = curve_toy._CurveModel()
    manifold = _off_diagonal_manifold()
    sigma = 0.2
    config = PolynomialReconstructionConfig(
        physical_half_width=30.0,
        physical_scan_points=601,
        fit_points=21,
        dense_points=401,
    )
    result = polynomial_reconstruction(
        adapter,
        model,
        curve_toy._observable(),
        state=curve_toy.STATE,
        params={},
        manifold=manifold,
        sigma=sigma,
        config=config,
    )
    assert result.status is CAMOutcomeStatus.VALID
    # R(x) = [[1, x], [x, 0.5]] is physical exactly while |x| <= sqrt(0.5).
    bound = float(np.sqrt(0.5))
    spacing = (
        2.0 * config.physical_half_width * sigma / (config.physical_scan_points - 1)
    )
    assert result.lower == pytest.approx(-bound, abs=spacing)
    assert result.upper == pytest.approx(bound, abs=spacing)
    assert result.domain_clipped is False
    expected_fraction = 2.0 * bound / (2.0 * config.physical_half_width * sigma)
    assert result.physical_grid_fraction == pytest.approx(expected_fraction, abs=1e-2)
    # Both the diffusion and the readout are constant along this coordinate.
    r11 = 0.5
    d00 = 0.5 * (1.0 - 1.0 / 2.0) * (1.0 + 0.2 * r11**2)
    d11 = 0.05
    expected_q = 0.5 * (d00 * r11 + d11 * 1.0)
    assert result.diffusion_coefficients[0] == pytest.approx(expected_q, rel=1e-9)
    assert all(abs(value) < 1e-9 for value in result.diffusion_coefficients[1:])
    expected_frequency = 1.0 + 0.5 * r11**2
    assert result.frequency_coefficients[0] == pytest.approx(
        expected_frequency, rel=1e-12
    )
    assert all(abs(value) < 1e-9 for value in result.frequency_coefficients[1:])


def test_the_physical_tolerance_decides_what_counts_as_physical():
    adapter = curve_toy._CurveAdapter()
    model = curve_toy._CurveModel()
    manifold = _off_diagonal_manifold()
    sigma = 0.2
    config = PolynomialReconstructionConfig(
        physical_half_width=30.0,
        physical_scan_points=601,
        fit_points=11,
        dense_points=201,
        physical_tolerance=6.0,
    )
    result = polynomial_reconstruction(
        adapter,
        model,
        curve_toy._observable(),
        state=curve_toy.STATE,
        params={},
        manifold=manifold,
        sigma=sigma,
        config=config,
    )
    assert result.status is CAMOutcomeStatus.VALID
    # A tolerance loose enough to accept every scanned state makes the interval the scan
    # itself, so the clipping is pure scan truncation and no state is called unphysical.
    half_width = config.physical_half_width * sigma
    assert result.lower == pytest.approx(-half_width, rel=1e-12)
    assert result.upper == pytest.approx(half_width, rel=1e-12)
    assert result.domain_clipped is True
    assert result.physical_grid_fraction == pytest.approx(1.0)


def test_a_non_physical_reference_state_is_reported():
    class _NegativeAdapter:
        """A state matrix that is not physical at the reference."""

        moment_layout = "normal"

        @staticmethod
        def state_matrix(vector: Any, params: Any = None) -> np.ndarray:
            del params
            return np.asarray(vector, dtype=float).reshape(1, 1).astype(complex) - 1.0

    manifold = NonlinearManifold(
        order=3,
        soft_rate=1.0,
        critical_eigenvalue_real=-1.0,
        critical_eigenvalue_imag=0.0,
        noncritical_spectral_gap=None,
        right=(1.0,),
        left=(1.0,),
        embeddings=((1.0,),),
        drift=(0.0, -1.0, 0.0, 0.0),
        maximum_bordered_condition=1.0,
        confining=True,
        method={"service": "test fixture"},
    )
    result = polynomial_reconstruction(
        _NegativeAdapter(),
        one_mode._UnitModel(),
        one_mode._observable(),
        state=[0.5],
        params={},
        manifold=manifold,
        sigma=0.5,
        config=PolynomialReconstructionConfig(
            physical_half_width=2.0,
            physical_scan_points=101,
            fit_points=11,
            dense_points=101,
        ),
    )
    assert result.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert result.reason_code == "center_not_physical"
    assert "not physical" in result.reason
    assert result.lower is None and result.upper is None
    assert result.generator_table is None
    assert result.evaluation is None


def test_a_non_positive_sampled_diffusion_is_not_clamped():
    adapter = one_mode._QuadraticAdapter()
    manifold = slow_manifold(adapter, [one_mode.X0], {}, order=3)
    sigma = _one_mode_sigma(manifold)
    result = polynomial_reconstruction(
        adapter,
        one_mode._BoundedModel(),
        one_mode._observable(),
        state=[one_mode.X0],
        params={},
        manifold=manifold,
        sigma=sigma,
        config=PolynomialReconstructionConfig(
            physical_half_width=3.0,
            physical_scan_points=601,
            fit_points=21,
            dense_points=201,
        ),
    )
    assert result.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert result.reason_code == "diffusion_not_positive"
    assert result.diffusion_minimum is not None and result.diffusion_minimum < 0.0
    assert result.generator_table is None
    assert result.evaluation is None


def test_the_configuration_defaults_and_validation():
    settings = PolynomialReconstructionConfig()
    assert settings.physical_half_width == pytest.approx(30.0)
    assert settings.physical_scan_points == 12001
    assert settings.physical_tolerance == pytest.approx(1.0e-10)
    assert settings.fit_points == 81
    assert settings.dense_points == 100001
    assert settings.quadrature.intensity_integration == "centered_cdf"
    assert settings.quadrature.core_density_comparison == "greater"
    assert settings.quadrature.edge_count_rule == "ceil_min_two"
    assert settings.quadrature.density_floor_relative == pytest.approx(1.0e-14)
    # ``integration_split`` is irrelevant to the centred assembly and stays its default.
    assert settings.quadrature.integration_split == "grid_zero"
    for updates in (
        {"physical_half_width": 0.0},
        {"physical_scan_points": 2},
        {"fit_points": 2},
        {"dense_points": 10},
        {"physical_tolerance": -1.0},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            PolynomialReconstructionConfig(**updates)


def test_the_integration_split_does_not_touch_the_centred_assembly():
    adapter, model, manifold, sigma = _one_mode()
    base = PolynomialReconstructionConfig(
        physical_half_width=2.0,
        physical_scan_points=401,
        fit_points=11,
        dense_points=201,
    )
    alternate = PolynomialReconstructionConfig(
        physical_half_width=2.0,
        physical_scan_points=401,
        fit_points=11,
        dense_points=201,
        quadrature=ScalarReconstructionConfig(
            intensity_integration="centered_cdf",
            core_density_comparison="greater",
            edge_count_rule="ceil_min_two",
            density_floor_relative=1.0e-14,
            integration_split="nearest_zero",
        ),
    )
    first = polynomial_reconstruction(
        adapter,
        model,
        one_mode._observable(),
        state=[one_mode.X0],
        params={},
        manifold=manifold,
        sigma=sigma,
        config=base,
    )
    second = polynomial_reconstruction(
        adapter,
        model,
        one_mode._observable(),
        state=[one_mode.X0],
        params={},
        manifold=manifold,
        sigma=sigma,
        config=alternate,
    )
    assert first.summary["scalar"] == second.summary["scalar"]
    assert second.method["quadrature_config"]["integration_split"] == "nearest_zero"


def test_the_structural_inputs_are_validated():
    adapter, model, manifold, sigma = _one_mode()
    config = PolynomialReconstructionConfig(
        physical_half_width=2.0,
        physical_scan_points=101,
        fit_points=11,
        dense_points=101,
    )
    with pytest.raises(ValueError, match="width must be finite and positive"):
        polynomial_reconstruction(
            adapter,
            model,
            one_mode._observable(),
            state=[one_mode.X0],
            params={},
            manifold=manifold,
            sigma=0.0,
            config=config,
        )
    with pytest.raises(ValueError, match="width must be finite and positive"):
        polynomial_reconstruction(
            adapter,
            model,
            one_mode._observable(),
            state=[one_mode.X0],
            params={},
            manifold=manifold,
            sigma=float("nan"),
            config=config,
        )
    with pytest.raises(ValueError, match="state must be finite"):
        polynomial_reconstruction(
            adapter,
            model,
            one_mode._observable(),
            state=[float("nan")],
            params={},
            manifold=manifold,
            sigma=sigma,
            config=config,
        )
    two_mode_manifold = _off_diagonal_manifold()
    with pytest.raises(ValueError, match="left vector has"):
        polynomial_reconstruction(
            adapter,
            model,
            one_mode._observable(),
            state=[one_mode.X0],
            params={},
            manifold=two_mode_manifold,
            sigma=sigma,
            config=config,
        )
