"""10G2a probe 3a: the algebraic carrier pole of the scalar backward generator.

The reference algorithm is the one of
``scripts/tools/probe_kerr_full_3mode_stochastic_carrier_pole.py::_generator_pole``
(not imported here): PCHIP resampling of ``f``, ``q`` and ``omega`` onto a uniform
grid, the upwind backward generator with reflecting ends, the frequency operator
switched on by a continuation coupling, and the eigenvalue nearest to the previous
pole.

The discretization is exactly singular when the source coefficients are constant
(every row sums to zero, which is the discrete mass-conservation mode), so the
constant-source test uses grids whose ``q / (2 dx²)`` stays far above the rounding
level of the reference's ``-1e-12`` starting shift; larger constant grids are a
genuine eigensolver failure, which the mocked tests pin down.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from qphase_cam.carrier_pole import (
    CarrierPoleConfig,
    carrier_pole,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from scipy.interpolate import PchipInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import ArpackNoConvergence, eigs

REAL_PCHIP = PchipInterpolator


def _constant_source(size: int = 8):
    """Uneven source grid with constant drift, diffusion and readout."""
    x = np.asarray([-1.0, -0.73, -0.41, -0.12, 0.0, 0.28, 0.64, 1.0])
    return (
        x,
        np.full(size, 0.5),
        np.ones(size),
        np.full(size, 2.0),
    )


def _curved_source():
    """Uneven source grid with a restoring drift, a curved diffusion and a tilt."""
    x = np.asarray([-2.0, -1.2, -0.6, -0.2, 0.0, 0.3, 0.9, 1.5, 2.0])
    return x, -1.5 * x, 1.0 + 0.4 * x**2, 1.0 + 0.5 * x


def _reference_pole(x_source, size, steps):
    """Compute the pole with the reference discretization, written out here."""
    x, f_source, q_source, omega_source = x_source
    grid = np.linspace(float(x[0]), float(x[-1]), size)
    f = PchipInterpolator(x, f_source)(grid)
    q = PchipInterpolator(x, q_source)(grid)
    omega = PchipInterpolator(x, omega_source)(grid)
    dx = float(grid[1] - grid[0])
    rate = q / (2.0 * dx**2)
    lower = rate + np.maximum(-f, 0.0) / dx
    upper = rate + np.maximum(f, 0.0) / dx
    lower[0] = 0.0
    upper[-1] = 0.0
    generator = diags(
        (lower[1:], -(lower + upper), upper[:-1]), offsets=(-1, 0, 1), format="csc"
    )
    reference = float(np.median(omega))
    tilt = diags(omega - reference, offsets=0, format="csc")
    pole = 0.0j
    for coupling in np.linspace(1.0 / steps, 1.0, steps):
        tilted = generator.astype(complex) - 1j * coupling * tilt
        sigma = pole if pole != 0.0j else -1.0e-12
        values = eigs(
            tilted,
            k=min(6, size - 2),
            sigma=sigma,
            which="LM",
            return_eigenvectors=False,
            tol=1.0e-10,
            maxiter=200_000,
        )
        pole = values[int(np.argmin(np.abs(values - pole)))]
    return reference - pole.imag, -pole.real


# --- the reference algorithm ----------------------------------------------


def test_a_constant_source_has_the_mass_conservation_pole():
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17, 33))
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.reason is None
    assert assessment.grid_converged is True
    assert assessment.finest_grid_size == 33
    # A constant readout zeroes the frequency operator, so the continuation tracks
    # the mass-conservation mode at exactly zero decay.
    assert assessment.carrier_frequency == pytest.approx(2.0, abs=1e-9)
    assert assessment.carrier_decay_rate == pytest.approx(0.0, abs=1e-9)
    for grid in assessment.grids:
        assert grid.carrier_frequency == pytest.approx(2.0, abs=1e-9)
        assert grid.carrier_decay_rate == pytest.approx(0.0, abs=1e-9)
        assert grid.omega_reference == pytest.approx(2.0, rel=0.0, abs=0.0)
        assert grid.continuation_steps == 20
    assert [grid.grid_size for grid in assessment.grids] == [9, 17, 33]
    assert len(assessment.grid_frequency_differences) == 2
    assert len(assessment.grid_decay_differences) == 2
    assert max(assessment.grid_decay_differences) < 1e-9
    assert "backward Feynman-Kac generator" in assessment.method["discretization"]
    assert "PCHIP" in assessment.method["interpolation"]


def test_the_pole_matches_an_independent_reference_discretization():
    source = _curved_source()
    steps = 3
    assessment = carrier_pole(
        *source,
        config=CarrierPoleConfig(
            grid_sizes=(21, 41),
            continuation_steps=steps,
            grid_absolute_tolerance=1e-2,
        ),
    )
    assert [grid.grid_size for grid in assessment.grids] == [21, 41]
    # The pole is not the trivial readout value, so the comparison is not vacuous.
    assert all(grid.carrier_decay_rate > 1e-2 for grid in assessment.grids)
    assert assessment.grids[-1].omega_reference == pytest.approx(1.0, rel=1e-12)
    for grid in assessment.grids:
        frequency, decay = _reference_pole(source, grid.grid_size, steps)
        assert grid.carrier_frequency == pytest.approx(frequency, abs=1e-12)
        assert grid.carrier_decay_rate == pytest.approx(decay, abs=1e-12)
        # ARPACK starts from an internal random vector, so the two runs agree to
        # rounding rather than bit for bit; the definitions must still line up.
        assert grid.pole_real == pytest.approx(-decay, abs=1e-12)
        assert grid.pole_imag_shifted == pytest.approx(
            grid.omega_reference - grid.carrier_frequency, abs=1e-12
        )
    # The finer source sampling is the convergence witness the status reports on.
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.finest_grid_size == 41
    assert assessment.grid_converged is True


def test_the_configuration_defaults_are_the_reference_probe():
    config = CarrierPoleConfig()
    assert config.grid_sizes == (401, 801, 1201)
    assert config.continuation_steps == 20
    assert config.k == 6
    assert config.eigensolver_tolerance == pytest.approx(1.0e-10)
    assert config.eigensolver_maxiter == 200_000
    assert config.grid_absolute_tolerance == pytest.approx(1.0e-3)
    assert config.grid_relative_tolerance == pytest.approx(1.0e-3)
    assert config.shift_floor == pytest.approx(1.0e-12)


# --- source-table and resampling failures ---------------------------------


def test_a_non_positive_source_diffusion_is_rejected():
    x = np.linspace(-1.0, 1.0, 5)
    assessment = carrier_pole(
        x, np.zeros(5), np.asarray([1.0, 1.0, 0.0, 1.0, 1.0]), np.zeros(5)
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "invalid_source_table"
    assert "diffusion coefficients must be strictly positive" in assessment.reason
    assert assessment.grids == ()
    assert assessment.carrier_frequency is None
    assert assessment.grid_converged is None


def test_a_malformed_source_table_is_rejected():
    good = np.linspace(-1.0, 1.0, 5)
    cases = {
        "too few points": (good[:2], np.zeros(2), np.ones(2), np.zeros(2)),
        "duplicated grid": (
            np.asarray([-1.0, 0.0, 0.0, 0.5, 1.0]),
            np.zeros(5),
            np.ones(5),
            np.zeros(5),
        ),
        "decreasing grid": (
            np.asarray([-1.0, 0.0, 0.2, 0.1, 1.0]),
            np.zeros(5),
            np.ones(5),
            np.zeros(5),
        ),
        "length mismatch": (good, np.zeros(4), np.ones(5), np.zeros(5)),
        "non-finite grid": (
            np.asarray([-1.0, 0.0, np.inf, 0.5, 1.0]),
            np.zeros(5),
            np.ones(5),
            np.zeros(5),
        ),
        "non-finite drift": (good, np.full(5, np.nan), np.ones(5), np.zeros(5)),
        "non-finite readout": (good, np.zeros(5), np.ones(5), np.full(5, np.inf)),
        "not numeric": (good, ["a"] * 5, np.ones(5), np.zeros(5)),
    }
    for name, arguments in cases.items():
        assessment = carrier_pole(*arguments)
        assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID, name
        assert assessment.reason_code == "invalid_source_table", name
        assert assessment.reason, name
        assert assessment.grids == (), name


def test_a_non_positive_resampled_diffusion_is_rejected(monkeypatch):
    class NegatingPchip:
        """Resample with the reference interpolant and flip the sign."""

        def __init__(self, x, y):
            self.interpolant = REAL_PCHIP(x, y)

        def __call__(self, values):
            return -np.asarray(self.interpolant(values), dtype=float)

    monkeypatch.setattr("qphase_cam.carrier_pole.PchipInterpolator", NegatingPchip)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "resampled_grid_invalid"
    assert "non-positive" in assessment.reason
    assert assessment.grids == ()


# --- eigensolver failures -------------------------------------------------


def test_an_arpack_failure_is_a_not_converged_status(monkeypatch):
    def failing(*args, **kwargs):
        del args, kwargs
        raise ArpackNoConvergence(
            "ARPACK error 3: No shifts could be applied",
            np.zeros(0, dtype=complex),
            np.zeros((0, 0), dtype=complex),
        )

    monkeypatch.setattr("qphase_cam.carrier_pole.eigs", failing)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.NOT_CONVERGED
    assert assessment.reason_code == "eigensolver_failed"
    assert "ARPACK" in assessment.reason
    assert assessment.grids == ()
    assert assessment.finest_grid_size is None
    assert assessment.carrier_frequency is None
    assert assessment.carrier_decay_rate is None


def test_a_non_finite_eigenvalue_is_not_reported_as_a_pole(monkeypatch):
    def non_finite(*args, **kwargs):
        del args, kwargs
        return np.asarray([np.nan + 0.0j, -1.0 + 0.0j, -2.0 + 0.0j])

    monkeypatch.setattr("qphase_cam.carrier_pole.eigs", non_finite)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.NOT_CONVERGED
    assert assessment.reason_code == "eigensolver_failed"
    assert "non-finite eigenvalue" in assessment.reason
    assert assessment.carrier_frequency is None
    assert assessment.carrier_decay_rate is None


def test_a_growing_pole_is_reported_unstable_and_not_clamped(monkeypatch):
    def growing(*args, **kwargs):
        del args, kwargs
        return np.asarray([0.1 + 0.0j, -1.0 + 0.0j, -2.0 + 0.0j])

    monkeypatch.setattr("qphase_cam.carrier_pole.eigs", growing)
    assessment = carrier_pole(
        *_constant_source(), config=CarrierPoleConfig(grid_sizes=(9, 17))
    )
    assert assessment.status is CAMOutcomeStatus.UNSTABLE
    assert assessment.reason_code == "negative_decay_rate"
    assert assessment.carrier_decay_rate == pytest.approx(-0.1, rel=1e-12)
    assert assessment.carrier_decay_rate < 0.0
    assert assessment.grid_converged is True
    assert [grid.grid_size for grid in assessment.grids] == [9, 17]


# --- grid convergence -----------------------------------------------------


def test_a_grid_disagreement_is_not_converged():
    assessment = carrier_pole(
        *_curved_source(),
        config=CarrierPoleConfig(
            grid_sizes=(21, 41),
            continuation_steps=3,
            grid_absolute_tolerance=0.0,
            grid_relative_tolerance=1.0e-14,
        ),
    )
    assert assessment.status is CAMOutcomeStatus.NOT_CONVERGED
    assert assessment.reason_code == "grid_not_converged"
    assert assessment.grid_converged is False
    assert [grid.grid_size for grid in assessment.grids] == [21, 41]
    assert assessment.finest_grid_size == 41
    assert assessment.carrier_frequency is not None
    assert assessment.carrier_decay_rate is not None
    assert assessment.grid_decay_differences[0] > 1.0e-5


@pytest.mark.parametrize(
    "overrides",
    [
        {"grid_sizes": (41,)},
        {"grid_sizes": (41, 21)},
        {"grid_sizes": (41, 41)},
        {"grid_sizes": (2, 41)},
        {"grid_sizes": (41, float("nan"))},
        {"continuation_steps": 0},
        {"k": 0},
        {"k": float("nan")},
        {"eigensolver_tolerance": 0.0},
        {"eigensolver_tolerance": float("inf")},
        {"eigensolver_maxiter": 0},
        {"grid_absolute_tolerance": -1.0},
        {"grid_absolute_tolerance": 0.0, "grid_relative_tolerance": 0.0},
        {"grid_relative_tolerance": float("nan")},
        {"decay_tolerance": -1.0},
        {"shift_floor": 0.0},
        {"shift_floor": float("inf")},
    ],
)
def test_an_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValidationError):
        CarrierPoleConfig(**overrides)
