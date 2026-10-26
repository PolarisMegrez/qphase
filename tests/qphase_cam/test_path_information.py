"""Path-information bounds of the full CAM covariance, on the two established toys.

The one-mode toy is ``test_nonlinear``'s: a bounded domain, ``J(v) = -1 - 3v²`` and unit
diffusion scaled by ``1 - v``, so the canonical covariance is the scalar
``Σ = 2(1 - v)v`` and the full bound must collapse onto the reduced one.

The two-mode toy is ``test_effective_drift``'s: four canonical coordinates with
``J = diag(-1, -3 - 3R11, -4, -5)`` and ``D = diag(½(1 - R00/2)(1 + 0.2R11²), 0.05)``,
so the canonical covariance is the **diagonal** matrix known in closed form and the full
bound can be compared with the one-dimensional formula direction by direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qphase_cam.nonlinear import (
    ScalarReconstructionConfig,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.path_information import (
    RELATIVE_PSD_TOLERANCE,
    PathInformationBounds,
    path_information_bounds,
)

from tests.qphase_cam import test_effective_drift as curve_toy
from tests.qphase_cam import test_nonlinear as one_mode

#: One quarter of every canonical coordinate, so the full forcing is not aligned with
#: the critical direction: that is the case where the full bound must exceed the
#: reduced one.
TWO_MODE_DIRECTION = np.asarray([0.25, 0.25, 0.25, 0.25])
ONE_MODE_GRID = 51
TWO_MODE_GRID = curve_toy.CURVE_SIDE


class _SingularModel(curve_toy._CurveModel):
    """``d11 = 0``: the projection stays positive while the full noise does not.

    ``Σ[1,1] = 2·d11·R11`` vanishes, so the full canonical covariance is singular even
    though ``q = Σ[0,0] = 2·d00·R00`` is positive and the reconstruction succeeds.
    """

    @classmethod
    def cam_diffusion(cls, matrix: Any, params: Any = None) -> np.ndarray:
        del params
        array = np.asarray(matrix)
        r00 = float(np.real(array[..., 0, 0]))
        r11 = float(np.real(array[..., 1, 1]))
        return np.asarray(
            [[0.5 * (1.0 - r00 / 2.0) * (1.0 + 0.2 * r11**2), 0.0], [0.0, 0.0]],
            dtype=complex,
        )


class _IndefiniteModel(curve_toy._CurveModel):
    """An indefinite diffusion: ``det D < 0``, so the covariance is indefinite."""

    @classmethod
    def cam_diffusion(cls, matrix: Any, params: Any = None) -> np.ndarray:
        del matrix, params
        return np.asarray([[0.5, 0.3], [0.3, 0.05]], dtype=complex)


def _one_mode_reconstruction() -> tuple[Any, Any, np.ndarray, Any]:
    adapter = one_mode._QuadraticAdapter()
    model = one_mode._BoundedModel()
    direction = np.asarray([1.0])
    reconstruction = scalar_reconstruction(
        adapter,
        model,
        one_mode._observable(),
        state=[one_mode.X0],
        params={},
        order=3,
        parameter_direction=direction,
        manifold=one_mode._confining_manifold(),
        config=ScalarReconstructionConfig(grid_points=ONE_MODE_GRID),
    )
    assert reconstruction.status is CAMOutcomeStatus.VALID
    assert reconstruction.generator_table is not None
    return adapter, model, direction, reconstruction


def _two_mode_reconstruction(
    model: Any = None,
    *,
    manifold: Any = None,
    direction: np.ndarray | None = TWO_MODE_DIRECTION,
) -> tuple[Any, Any, Any]:
    adapter = curve_toy._CurveAdapter()
    resolved_model = curve_toy._CurveModel() if model is None else model
    reconstruction = scalar_reconstruction(
        adapter,
        resolved_model,
        curve_toy._observable(),
        state=curve_toy.STATE,
        params={},
        manifold=curve_toy._curve_manifold() if manifold is None else manifold,
        parameter_direction=direction,
        config=ScalarReconstructionConfig(grid_points=TWO_MODE_GRID),
    )
    return adapter, resolved_model, reconstruction


def _one_mode_sigma(x: np.ndarray) -> np.ndarray:
    """``Σ = 2(1 - v)v`` of the one-mode toy with ``v = 0.5 + x``."""
    v = one_mode.X0 + x
    return 2.0 * (1.0 - v) * v


def _two_mode_sigma(x: np.ndarray) -> np.ndarray:
    """Return the diagonal ``Σ`` of the two-mode toy, one column per coordinate."""
    r00 = 1.0 + x
    r11 = 0.5
    d00 = 0.5 * (1.0 - r00 / 2.0) * (1.0 + 0.2 * r11**2)
    d11 = 0.05
    symmetric = 0.5 * (d00 * r11 + d11 * r00)
    return np.stack(
        [
            2.0 * d00 * r00,
            np.full(r00.shape, 2.0 * d11 * r11),
            symmetric,
            symmetric,
        ],
        axis=1,
    )


def test_the_reduced_bound_is_the_reconstruction_rate():
    adapter, model, direction, reconstruction = _one_mode_reconstruction()
    bounds = path_information_bounds(
        adapter,
        model,
        state=[one_mode.X0],
        params={},
        reconstruction=reconstruction,
        parameter_direction=direction,
    )
    assert isinstance(bounds, PathInformationBounds)
    assert bounds.status is CAMOutcomeStatus.VALID
    assert bounds.source_status is CAMOutcomeStatus.VALID
    assert bounds.reason is None and bounds.reason_code is None
    # The reduced bound *is* the shared evaluation's path-information rate, and the
    # source reported the same integral: both differences are real zeros here.
    assert bounds.evaluation is not None
    assert bounds.reduced == bounds.evaluation.path_information_rate
    assert bounds.reduced == reconstruction.path_information_rate
    assert bounds.reduced_absolute_difference == 0.0
    assert bounds.reduced_relative_difference == 0.0
    # A one-coordinate covariance makes the full bound the reduced one.
    assert bounds.full == pytest.approx(bounds.reduced, rel=1e-15)
    assert bounds.full_at_center == pytest.approx(
        1.7**2 / (2.0 * (1.0 - one_mode.X0) * one_mode.X0), rel=1e-14
    )
    assert bounds.minimum_relative_eigenvalue == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert bounds.method["integration_split"] == "grid_zero"
    assert bounds.method["batch_noise"]["adapter_supports_batched_states"] is False
    assert bounds.method["parameter_direction"] == [1.0]
    assert bounds.method["config"]["integration_split"] == "grid_zero"
    assert bounds.evaluation.normalizable is True


def test_the_full_bound_is_the_one_dimensional_covariance_formula():
    adapter, model, direction, reconstruction = _one_mode_reconstruction()
    bounds = path_information_bounds(
        adapter,
        model,
        state=[one_mode.X0],
        params={},
        reconstruction=reconstruction,
        parameter_direction=direction,
    )
    assert bounds.evaluation is not None
    table = reconstruction.generator_table
    assert table is not None
    x = np.asarray(table.x, dtype=float)
    density = np.asarray(bounds.evaluation.density, dtype=float)
    sigma = _one_mode_sigma(x)
    forcing = 1.7  # parameter_jacobian [[1.7]] @ direction [1.0]
    expected_full = float(np.trapezoid(forcing**2 / sigma * density, x))
    expected_reduced = float(np.trapezoid(forcing**2 * density / sigma, x))
    assert bounds.full == pytest.approx(expected_full, rel=1e-13)
    assert bounds.reduced == pytest.approx(expected_reduced, rel=1e-13)
    # The density is the shared evaluation's, normalized on the table's own grid.
    assert float(np.trapezoid(density, x)) == pytest.approx(1.0, rel=1e-12)
    # Σ is positive along the whole table, and its smallest eigenvalue is the
    # projected diffusion the reconstruction uses.
    assert float(np.min(sigma)) > 0.0
    assert np.allclose(np.asarray(table.diffusion, dtype=float), sigma, rtol=1e-12)


def test_a_two_mode_noise_makes_the_full_bound_strictly_larger():
    adapter, model, reconstruction = _two_mode_reconstruction()
    bounds = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=TWO_MODE_DIRECTION,
    )
    assert bounds.status is CAMOutcomeStatus.VALID
    assert bounds.source_status is CAMOutcomeStatus.VALID
    assert bounds.evaluation is not None
    assert bounds.reduced == reconstruction.path_information_rate
    table = reconstruction.generator_table
    assert table is not None
    x = np.asarray(table.x, dtype=float)
    density = np.asarray(bounds.evaluation.density, dtype=float)
    sigma = _two_mode_sigma(x)
    forcing = TWO_MODE_DIRECTION
    expected_full = float(np.trapezoid(np.sum(forcing**2 / sigma, axis=1) * density, x))
    expected_reduced = float(np.trapezoid(forcing[0] ** 2 / sigma[:, 0] * density, x))
    assert bounds.full == pytest.approx(expected_full, rel=1e-13)
    assert bounds.reduced == pytest.approx(expected_reduced, rel=1e-13)
    # The reduced bound keeps only the critical noise, so the full one is larger by
    # more than an order of magnitude on this toy.
    assert bounds.full > 10.0 * bounds.reduced
    # The centre bound is F₀ᵀ Σ(0)⁻¹ F₀, with Σ(0) = diag(0.525, 0.05, 0.090625, s).
    centre_sigma = _two_mode_sigma(np.asarray([0.0]))[0]
    assert bounds.full_at_center == pytest.approx(
        float(np.sum(forcing**2 / centre_sigma)), rel=1e-13
    )
    # The conditioning reported is the smallest eigenvalue ratio of the diagonal Σ.
    expected_relative = float(np.min(np.min(sigma, axis=1) / np.max(sigma, axis=1)))
    assert bounds.minimum_relative_eigenvalue == pytest.approx(
        expected_relative, rel=1e-9
    )
    assert bounds.minimum_relative_eigenvalue > 0.0
    assert bounds.reduced_absolute_difference == 0.0


def test_a_source_that_is_not_valid_is_not_upgraded():
    manifold = slow_manifold(curve_toy._CurveAdapter(), curve_toy.STATE, {}, order=3)
    # No parameter direction on the source: it reports no path-information rate, which
    # is the case where a difference must stay ``None`` instead of becoming zero.
    adapter, model, reconstruction = _two_mode_reconstruction(
        manifold=manifold, direction=None
    )
    assert reconstruction.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    bounds = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=TWO_MODE_DIRECTION,
    )
    # The table is complete, so the numbers are real...
    assert bounds.reduced is not None and np.isfinite(bounds.reduced)
    assert bounds.full is not None and np.isfinite(bounds.full)
    assert bounds.full_at_center is not None and np.isfinite(bounds.full_at_center)
    # ... and the verdict is the source's, not a fresh one.
    assert bounds.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert bounds.source_status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert bounds.reason_code == "manifold_not_confining"
    assert "not valid" in bounds.reason
    assert bounds.method["source_valid"] is False
    assert bounds.method["source_reason_code"] == "manifold_not_confining"
    # The source reported no path-information rate, so no difference is invented.
    assert reconstruction.path_information_rate is None
    assert bounds.reduced_reconstruction is None
    assert bounds.reduced_absolute_difference is None
    assert bounds.reduced_relative_difference is None


def test_a_source_without_a_generator_table_cannot_be_bounded():
    adapter = curve_toy._CurveAdapter()
    model = curve_toy._ZeroDiffusionModel()
    reconstruction = scalar_reconstruction(
        adapter,
        model,
        curve_toy._observable(),
        state=curve_toy.STATE,
        params={},
        manifold=curve_toy._curve_manifold(),
        config=ScalarReconstructionConfig(grid_points=TWO_MODE_GRID),
    )
    assert reconstruction.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert reconstruction.generator_table is None
    bounds = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=TWO_MODE_DIRECTION,
    )
    assert bounds.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert bounds.source_status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert bounds.reason_code == "no_generator_table"
    assert reconstruction.reason in bounds.reason
    assert bounds.reduced is None
    assert bounds.full is None
    assert bounds.full_at_center is None
    assert bounds.minimum_relative_eigenvalue is None
    assert bounds.evaluation is None
    assert bounds.diagnostics["missing_source_fields"] == ["generator_table"]


def test_a_valid_source_without_a_table_is_a_caller_error():
    adapter, model, reconstruction = _two_mode_reconstruction()
    broken = reconstruction.model_copy(update={"generator_table": None})
    with pytest.raises(ValueError, match="generator table"):
        path_information_bounds(
            adapter,
            model,
            state=curve_toy.STATE,
            params={},
            reconstruction=broken,
            parameter_direction=TWO_MODE_DIRECTION,
        )


def test_a_singular_full_covariance_is_reported_not_regularized():
    adapter, model, reconstruction = _two_mode_reconstruction(model=_SingularModel())
    assert reconstruction.status is CAMOutcomeStatus.VALID
    assert reconstruction.generator_table is not None
    bounds = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=TWO_MODE_DIRECTION,
    )
    assert bounds.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert bounds.source_status is CAMOutcomeStatus.VALID
    assert bounds.reason_code in {"covariance_singular", "covariance_solve_not_finite"}
    assert bounds.diagnostics["minimum_relative_eigenvalue"] == pytest.approx(
        0.0, abs=RELATIVE_PSD_TOLERANCE
    )
    # Nothing is reported as a number: the reduced bound is available from the
    # evaluation, but the full and centre ones have no value.
    assert bounds.full is None
    assert bounds.full_at_center is None


def test_an_indefinite_covariance_is_reported_not_clamped():
    adapter, model, reconstruction = _two_mode_reconstruction(model=_IndefiniteModel())
    assert reconstruction.generator_table is not None
    bounds = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=TWO_MODE_DIRECTION,
    )
    assert bounds.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert bounds.reason_code == "covariance_not_positive_semidefinite"
    assert bounds.diagnostics["minimum_relative_eigenvalue"] < -RELATIVE_PSD_TOLERANCE
    assert bounds.full is None
    assert bounds.full_at_center is None
    assert bounds.minimum_relative_eigenvalue is None


def test_the_parameter_direction_must_match_the_jacobian():
    adapter, model, reconstruction = _two_mode_reconstruction()
    with pytest.raises(ValueError, match="parameter direction has"):
        path_information_bounds(
            adapter,
            model,
            state=curve_toy.STATE,
            params={},
            reconstruction=reconstruction,
            parameter_direction=np.asarray([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="must be finite"):
        path_information_bounds(
            adapter,
            model,
            state=curve_toy.STATE,
            params={},
            reconstruction=reconstruction,
            parameter_direction=np.asarray([1.0, 0.0, 0.0, float("nan")]),
        )


class _WideParameterAdapter(curve_toy._CurveAdapter):
    """A parameter Jacobian with more parameters than canonical coordinates.

    ``parameter_direction`` lives in the adapter's parameter space (one entry per model
    parameter), which the Jacobian maps onto the coordinates, so the two counts are
    independent: the direction has five entries and the state four coordinates.
    """

    def parameter_jacobian(self, vector: Any, params: Any = None) -> np.ndarray:
        del params
        value = np.asarray(vector, dtype=float).reshape(-1)
        base = np.eye(4) + 0.1 * value[:, None] * np.ones((1, 4))
        return np.hstack([base, 0.05 * value[:, None]])


def test_a_parameter_space_direction_is_not_a_coordinate_vector():
    adapter = _WideParameterAdapter()
    model = curve_toy._CurveModel()
    direction = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625])
    reconstruction = scalar_reconstruction(
        adapter,
        model,
        curve_toy._observable(),
        state=curve_toy.STATE,
        params={},
        manifold=curve_toy._curve_manifold(),
        parameter_direction=direction,
        config=ScalarReconstructionConfig(grid_points=TWO_MODE_GRID),
    )
    assert reconstruction.status is CAMOutcomeStatus.VALID
    bounds = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=direction,
    )
    assert bounds.status is CAMOutcomeStatus.VALID
    assert bounds.method["parameter_direction"] == [float(value) for value in direction]
    assert bounds.reduced == reconstruction.path_information_rate
    assert bounds.full is not None and bounds.full > bounds.reduced


def test_the_default_integration_split_stays_grid_zero():
    adapter, model, reconstruction = _two_mode_reconstruction()
    default = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=TWO_MODE_DIRECTION,
    )
    assert ScalarReconstructionConfig().integration_split == "grid_zero"
    assert default.method["integration_split"] == "grid_zero"
    # The table's grid contains zero, so the two split policies agree here: the
    # nearest-zero policy is an opt-in, not the default.
    nearest = path_information_bounds(
        adapter,
        model,
        state=curve_toy.STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=TWO_MODE_DIRECTION,
        config=ScalarReconstructionConfig(integration_split="nearest_zero"),
    )
    assert nearest.method["integration_split"] == "nearest_zero"
    assert nearest.reduced == default.reduced
    assert nearest.full == default.full
