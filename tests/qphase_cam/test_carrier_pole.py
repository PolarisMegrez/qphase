"""10G2a probe 3a: the algebraic carrier pole of the scalar backward generator.

The reference algorithm is the one of
``scripts/tools/probe_kerr_full_3mode_stochastic_carrier_pole.py::_generator_pole``
(not imported here): PCHIP resampling of ``f``, ``q`` and ``omega`` onto a uniform
grid, the upwind backward generator with reflecting ends, the frequency operator
switched on by a continuation coupling, and the eigenvalue nearest to the previous
pole.

The discretization is exactly singular when the source coefficients are constant
(every row sums to zero, which is the discrete mass-conservation mode), so the
shift-invert shift has to stay above the rounding level of ``q / (2 dx²)``; the
constant source is therefore exercised on the reference grids and not only on small
ones, since a large grid is exactly where the reference's fixed ``-1e-12`` shift
becomes unrepresentable.
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
    """Compute the pole with the reference discretization, written out here.

    The shift is kept literally as the reference probe writes it (the previous pole,
    ``-1e-12`` at the start). A resolvable pole is far above either shift, so the
    parity test also shows that the service's scale-aware shift does not move it.
    """
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


@pytest.mark.parametrize(
    ("drift", "diffusion", "readout"),
    [
        (0.0, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (-0.5, 1.0, 2.0),
        (0.0, 2.5, -3.0),
    ],
)
def test_the_constant_source_converges_on_the_reference_grids(
    drift, diffusion, readout
):
    """The large-grid regression for the constant (exactly singular) source.

    The tolerance is the shift-invert accuracy of the tracked zero mode: ARPACK's
    ``tol`` is relative, so the absolute residual is bounded by
    ``tol * max|diagonal|`` (about ``1e-10 * 2 q / (2 dx^2)``, i.e. ~4e-5 on the
    finest grid), while the observed decay is ~2e-11. ``1e-8`` sits about three
    orders inside that bound and about three above the observation. The carrier
    frequency is not affected in the same way (``Im(pole)`` is ~0 when the frequency
    operator is zeroed), so it is held to ``1e-9``.
    """
    x = np.linspace(-1.0, 1.0, 9)
    assessment = carrier_pole(
        x,
        np.full(x.size, drift),
        np.full(x.size, diffusion),
        np.full(x.size, readout),
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.reason_code is None
    assert assessment.grid_converged is True
    assert [grid.grid_size for grid in assessment.grids] == [401, 801, 1201]
    assert assessment.finest_grid_size == 1201
    assert assessment.carrier_frequency == pytest.approx(readout, abs=1e-9)
    assert abs(assessment.carrier_decay_rate) < 1e-8
    for grid in assessment.grids:
        assert grid.carrier_frequency == pytest.approx(readout, abs=1e-9)
        assert abs(grid.carrier_decay_rate) < 1e-8
        # The readout is constant, so the frequency operator is exactly zero and the
        # tracked pole is the mass-conservation mode at zero decay on every grid.
        assert grid.omega_reference == pytest.approx(readout, rel=0.0, abs=0.0)
        assert abs(grid.pole_imag_shifted) < 1e-8
    assert max(assessment.grid_decay_differences) < 1e-8
    # On these grids the scale-aware shift is what makes the solve possible: the
    # reference's fixed 1e-12 is below ulp(max|diagonal|), which is what turns the
    # shift-invert factor exactly singular.
    diagonal_scale = 2.0 * diffusion / (2.0 * (2.0 / 1200.0) ** 2)
    assert np.finfo(float).eps * diagonal_scale * 1201 > 1e-12
    assert "eps*max|diagonal|*size" in assessment.method["pole_selection"]


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


def test_a_degenerate_diffusion_rate_is_a_resampled_grid_failure():
    """Underflow and overflow of ``q / (2 dx^2)`` must not pose as a pole.

    A zero rate would zero whole rows of the generator (an all-zero matrix still
    returns an eigenvalue) and an infinite rate poisons the diagonal, so both are
    typed failures rather than a plausible-looking carrier. The PCHIP resampling stays
    finite in both cases, so this is the rate guard and not the finiteness guard.
    """
    wide = np.linspace(-1.0e50, 1.0e50, 5)
    underflow = carrier_pole(
        wide,
        np.zeros(5),
        np.full(5, 1.0e-300),
        np.zeros(5),
        config=CarrierPoleConfig(grid_sizes=(5, 7)),
    )
    assert underflow.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert underflow.reason_code == "resampled_grid_invalid"
    assert "q / (2 dx^2)" in underflow.reason
    assert underflow.grids == ()

    narrow = np.linspace(0.0, 2.0e-9, 5)
    overflow = carrier_pole(
        narrow,
        np.zeros(5),
        np.full(5, 1.0e300),
        np.zeros(5),
        config=CarrierPoleConfig(grid_sizes=(5, 7)),
    )
    assert overflow.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert overflow.reason_code == "resampled_grid_invalid"
    assert "q / (2 dx^2)" in overflow.reason
    assert overflow.carrier_frequency is None


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
