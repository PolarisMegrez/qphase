"""Frozen math baselines of the sampled-scalar measurement path (Wave 6B).

Covers the Phase 6 math layer before the analyser wiring lands:

- strict one-sided folding of two-sided DFT powers (DC/Nyquist counted once,
  interior Hermitian pairs doubled), the non-negative frequency axis and the
  canonical sampled-scalar PSD units;
- the strict ``none``/``constant`` detrend modes;
- the sampled-scalar Allan variance (non-overlapping primary and overlapping
  variant) against hand-computed and analytic constant/sinusoid/white-noise/
  Ornstein-Uhlenbeck fixtures;
- the identity that the overlapping scalar Allan variance of phase
  increments ``y_k = (phi_{k+1} - phi_k) / dt`` reproduces the existing
  overlapping phase-frequency Allan variance of ``phi``.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.allan_statistics import (
    calculate_allan_variance,
    calculate_sampled_scalar_allan,
)
from qphase_sde.math.spectral import (
    detrend_series,
    fold_one_sided_powers,
    one_sided_bin_count,
    one_sided_frequency_axis,
    sampled_scalar_psd_units,
)
from qphase_sde.spectral_estimator.builtin import (
    PeriodogramEstimator,
    PeriodogramEstimatorConfig,
)

BACKEND = NumpyBackend()


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# one-sided folding
# ---------------------------------------------------------------------------


def test_fold_one_sided_even_length_hand_computed():
    """Even N: DC and Nyquist bins are counted once, the interior doubles."""
    powers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    folded = fold_one_sided_powers(powers)

    np.testing.assert_array_equal(folded, [1.0, 4.0, 6.0, 8.0, 5.0])


def test_fold_one_sided_odd_length_hand_computed():
    """Odd N: no Nyquist bin exists; every non-DC bin doubles."""
    powers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    folded = fold_one_sided_powers(powers)

    np.testing.assert_array_equal(folded, [1.0, 4.0, 6.0, 8.0])


def test_fold_one_sided_degenerate_lengths():
    """N=1 is DC only; N=2 is DC+Nyquist with nothing to double."""
    np.testing.assert_array_equal(fold_one_sided_powers(np.array([3.0])), [3.0])
    np.testing.assert_array_equal(
        fold_one_sided_powers(np.array([3.0, 4.0])), [3.0, 4.0]
    )
    with pytest.raises(ValueError, match="positive"):
        one_sided_bin_count(0)


def test_fold_one_sided_preserves_batch_dimensions():
    rng = np.random.default_rng(20261010)
    powers = rng.uniform(0.5, 1.5, size=(5, 3, 8))

    folded = fold_one_sided_powers(powers)

    assert folded.shape == (5, 3, 5)
    np.testing.assert_allclose(folded[..., 0], powers[..., 0])
    np.testing.assert_allclose(folded[..., -1], powers[..., 4])
    np.testing.assert_allclose(folded[..., 1:4], 2.0 * powers[..., 1:4])


def test_fold_preserves_total_power_for_hermitian_spectrum():
    """Folding a real signal's DFT powers keeps the total sum exactly."""
    rng = np.random.default_rng(20261011)
    for n_time in (511, 512):
        x = rng.standard_normal((4, n_time))
        powers = np.abs(np.fft.fft(x, axis=-1)) ** 2

        folded = fold_one_sided_powers(powers)

        np.testing.assert_allclose(
            folded.sum(axis=-1), powers.sum(axis=-1), rtol=1e-14
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_fold_one_sided_is_backend_duck_typed():
    """The fold must not force a host copy of device-resident powers."""
    import cupy as cp

    rng = np.random.default_rng(20261012)
    powers = rng.uniform(0.5, 1.5, size=(3, 64))
    device_powers = cp.asarray(powers)

    folded = fold_one_sided_powers(device_powers)

    assert isinstance(folded, cp.ndarray)
    np.testing.assert_allclose(
        cp.asnumpy(folded), fold_one_sided_powers(powers), rtol=0, atol=0
    )


def test_one_sided_frequency_axis_matches_rfftfreq():
    """The folded coordinate is the non-negative (rfft) frequency grid."""
    for n_time in (7, 8):
        for dt in (0.1, 2.0):
            axis = one_sided_frequency_axis(n_time, dt, convention="pragmatic")
            np.testing.assert_allclose(
                axis, np.fft.rfftfreq(n_time, d=dt), rtol=1e-14, atol=0
            )
            assert np.all(axis >= 0.0)
            assert np.all(np.diff(axis) > 0.0)


def test_one_sided_frequency_axis_angular_conventions_scale_by_two_pi():
    for convention in ("symmetric", "unitary"):
        axis = one_sided_frequency_axis(16, 0.25, convention=convention)
        np.testing.assert_allclose(
            axis, 2.0 * np.pi * np.fft.rfftfreq(16, d=0.25), rtol=1e-14, atol=0
        )
    with pytest.raises(ValueError, match="convention"):
        one_sided_frequency_axis(16, 0.25, convention="bogus")
    with pytest.raises(ValueError, match="positive"):
        one_sided_frequency_axis(16, 0.0, convention="pragmatic")


def test_detrend_series_modes():
    y = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])

    untouched = detrend_series(y, "none")
    np.testing.assert_array_equal(untouched, y)

    detrended = detrend_series(y, "constant")
    np.testing.assert_allclose(detrended.mean(axis=-1), 0.0, atol=1e-15)
    np.testing.assert_allclose(detrended[0], [-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_allclose(detrended[1], [-3.0, -1.0, 1.0, 3.0])
    # The input is not mutated.
    np.testing.assert_array_equal(y[0], [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="detrend"):
        detrend_series(y, "linear")


def test_sampled_scalar_psd_units_are_canonical():
    assert (
        sampled_scalar_psd_units("occupation", convention="symmetric")
        == "occupation^2/inverse_time"
    )
    assert (
        sampled_scalar_psd_units("occupation", convention="pragmatic")
        == "occupation^2/inverse_time"
    )
    assert sampled_scalar_psd_units("", convention="unitary") == "1/inverse_time"
    with pytest.raises(ValueError, match="convention"):
        sampled_scalar_psd_units("occupation", convention="bogus")


# ---------------------------------------------------------------------------
# estimator composition: folding + normalization (Parseval) against rfft
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["symmetric", "unitary", "pragmatic"])
@pytest.mark.parametrize("window", [None, "hamming"])
def test_periodogram_fold_matches_rfft_reference(convention, window):
    """Folded estimator output equals the independent one-sided rfft reference.

    The reference applies the documented density scaling to the one-sided
    rfft bins and merges the Hermitian pairs explicitly; the production path
    under test computes the full two-sided periodogram first and folds it.
    """
    rng = np.random.default_rng(20261013)
    n_traj, n_time, dt = 3, 512, 0.2
    x = rng.standard_normal((n_traj, n_time))

    norm = "ortho" if convention in ("symmetric", "unitary") else None
    window_values = (
        np.ones(n_time) if window is None else getattr(np, window)(n_time)
    )
    energy = float(np.sum(window_values * window_values))
    scale = (
        n_time * dt / (2.0 * np.pi * energy)
        if convention in ("symmetric", "unitary")
        else dt / energy
    )
    tapered = x * window_values

    reference = np.abs(np.fft.rfft(tapered, norm=norm, axis=-1)) ** 2 * scale
    reference[:, 1:-1] *= 2.0
    reference_mean = reference.mean(axis=0)

    estimator = PeriodogramEstimator(PeriodogramEstimatorConfig(window=window))
    estimate = estimator.estimate(x, dt, convention, BACKEND)
    # The estimator reports fftshifted two-sided statistics; undo the shift
    # to recover unshifted DFT bin order, then fold.
    unshifted_mean = np.fft.ifftshift(estimate.mean)
    unshifted_axis = np.fft.ifftshift(estimate.axis)
    folded_mean = fold_one_sided_powers(unshifted_mean)

    np.testing.assert_allclose(folded_mean, reference_mean, rtol=1e-12, atol=1e-30)
    # The two-sided fftfreq grid reports the even-N Nyquist bin at its
    # negative alias; the one-sided product reports the positive value.
    np.testing.assert_allclose(
        np.abs(unshifted_axis[: one_sided_bin_count(n_time)]),
        one_sided_frequency_axis(n_time, dt, convention=convention),
        rtol=1e-14,
        atol=0,
    )

    # Parseval: the folded density integrates to the tapered mean-square.
    folded_axis = one_sided_frequency_axis(n_time, dt, convention=convention)
    step = float(folded_axis[1] - folded_axis[0])
    np.testing.assert_allclose(
        folded_mean.sum() * step,
        np.mean(np.sum(tapered * tapered, axis=-1) / energy),
        rtol=1e-12,
    )


def test_periodogram_fold_of_constant_is_dc_only():
    """A constant signal folds all of its power into the DC bin."""
    n_traj, n_time, dt = 2, 256, 0.5
    level = 1.7
    x = np.full((n_traj, n_time), level)

    estimate = PeriodogramEstimator(PeriodogramEstimatorConfig()).estimate(
        x, dt, "pragmatic", BACKEND
    )
    folded = fold_one_sided_powers(np.fft.ifftshift(estimate.mean))
    axis = one_sided_frequency_axis(n_time, dt, convention="pragmatic")

    # The DC bin holds the full density: |sum(x)|^2 * dt / N = level^2*N*dt.
    assert folded[0] == pytest.approx(level**2 * n_time * dt, rel=1e-12)
    np.testing.assert_allclose(folded[1:], 0.0, atol=1e-24)
    # Parseval: the folded density integrates to the mean-square level^2.
    step = float(axis[1] - axis[0])
    assert folded.sum() * step == pytest.approx(level**2, rel=1e-12)
    # With constant detrending the DC bin empties as well.
    detrended = detrend_series(x, "constant")
    estimate = PeriodogramEstimator(PeriodogramEstimatorConfig()).estimate(
        detrended, dt, "pragmatic", BACKEND
    )
    np.testing.assert_allclose(
        fold_one_sided_powers(np.fft.ifftshift(estimate.mean)), 0.0, atol=1e-24
    )


def test_periodogram_fold_recovers_sinusoid_frequency():
    """A real sinusoid peaks at its own bin on the one-sided axis."""
    n_time, dt = 512, 0.1
    times = np.arange(n_time) * dt
    omega = 2.0 * np.pi * 3.0 / (n_time * dt)  # exactly bin 3
    x = np.sin(omega * times)[None, :]

    estimate = PeriodogramEstimator(PeriodogramEstimatorConfig()).estimate(
        x, dt, "symmetric", BACKEND
    )
    folded = fold_one_sided_powers(np.fft.ifftshift(estimate.mean))
    axis = one_sided_frequency_axis(n_time, dt, convention="symmetric")

    peak = int(np.argmax(folded))
    assert peak == 3
    assert axis[peak] == pytest.approx(omega)
    # Parseval: a unit-amplitude sinusoid has mean-square 1/2.
    step = float(axis[1] - axis[0])
    assert folded.sum() * step == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# sampled-scalar Allan variance
# ---------------------------------------------------------------------------


def _scalar_allan(y, dt, *, taus, min_windows=8, min_nonoverlap_pairs=4):
    return calculate_sampled_scalar_allan(
        y,
        dt,
        taus=taus,
        points=40,
        min_windows=min_windows,
        min_nonoverlap_pairs=min_nonoverlap_pairs,
    )


def test_scalar_allan_constant_series_vanishes():
    y = np.full((3, 1000), 2.5)

    result = _scalar_allan(y, 0.1, taus=[1.0, 2.0, 4.0])

    np.testing.assert_array_equal(result["tau"], [1.0, 2.0, 4.0])
    for key in ("sampled_scalar_variance", "overlap_sampled_scalar_variance"):
        np.testing.assert_array_equal(result[key], np.zeros(3))
    # Every window pair is valid for a constant series.
    np.testing.assert_array_equal(
        result["valid_window_pairs"],
        [[1000 // m - 1 for m in (10, 20, 40)]] * 3,
    )
    np.testing.assert_array_equal(
        result["overlap_valid_window_pairs"],
        [[1000 - 2 * m + 1 for m in (10, 20, 40)]] * 3,
    )
    np.testing.assert_array_equal(
        result["nominal_windows_per_trajectory"], [100, 50, 25]
    )


def test_scalar_allan_sinusoid_matches_brute_force():
    """A small case against a literal nested-loop evaluation of the formulas."""
    n_time, dt = 64, 0.5
    times = np.arange(n_time) * dt
    y = (1.3 * np.sin(2.0 * np.pi * 3.0 * times / (n_time * dt) + 0.7))[None, :]
    m = 4

    result = _scalar_allan(y, dt, taus=[m * dt])

    # Non-overlapping primary: disjoint block means, consecutive pairs.
    blocks = y[0][: (n_time // m) * m].reshape(-1, m).mean(axis=1)
    expected_primary = 0.5 * np.mean(np.diff(blocks) ** 2)
    # Overlapping: stride-1 block means, pairs separated by m.
    cumsum = np.concatenate([[0.0], np.cumsum(y[0])])
    means = (cumsum[m:] - cumsum[:-m]) / m
    expected_overlap = 0.5 * np.mean((means[m:] - means[:-m]) ** 2)

    assert result["sampled_scalar_variance"][0] == pytest.approx(
        expected_primary, rel=1e-14
    )
    assert result["overlap_sampled_scalar_variance"][0] == pytest.approx(
        expected_overlap, rel=1e-14
    )
    np.testing.assert_allclose(
        result["per_trajectory"][:, 0], [expected_primary], rtol=1e-14
    )
    np.testing.assert_allclose(
        result["overlap_per_trajectory"][:, 0], [expected_overlap], rtol=1e-14
    )


def test_scalar_allan_white_noise_matches_analytic():
    """White noise of variance s^2 has Allan variance s^2 / m on both paths."""
    rng = np.random.default_rng(20261014)
    n_traj, n_time, dt = 256, 8192, 1.0
    sigma = 0.7
    y = sigma * rng.standard_normal((n_traj, n_time))
    taus = [4.0, 16.0, 64.0]

    result = _scalar_allan(y, dt, taus=taus)

    for column, tau in enumerate(taus):
        m = int(tau / dt)
        expected = sigma**2 / m
        for mean_key, sem_key in (
            ("sampled_scalar_variance", "sampled_scalar_variance_sem"),
            ("overlap_sampled_scalar_variance", "overlap_sampled_scalar_variance_sem"),
        ):
            mean = result[mean_key][column]
            sem = result[sem_key][column]
            assert mean == pytest.approx(expected, rel=0.05)
            # The ensemble mean must agree with the analytic value within a
            # few standard errors of the per-trajectory scatter.
            assert abs(mean - expected) < 5.0 * sem


def test_scalar_allan_ou_process_matches_analytic():
    """OU (AR(1)) Allan variance against the exact autocovariance sums.

    For stationary autocovariance C(d) = s^2 rho^|d|, a length-m block mean
    has variance ``(s^2/m^2) * (m + 2 * sum_{d=1}^{m-1} (m-d) rho^d)`` and
    adjacent disjoint blocks have covariance
    ``(s^2/m^2) * sum_{d=1}^{2m-1} (m - |d-m|) rho^d``; the Allan variance
    ``1/2 E[(y_bar_1 - y_bar_0)^2]`` equals their difference. Both the
    non-overlapping and overlapping estimators share this expectation.
    """
    rng = np.random.default_rng(20261015)
    n_traj, n_time, dt = 256, 8192, 1.0
    theta = 5.0 * dt
    rho = float(np.exp(-dt / theta))
    variance = 2.0
    innovation = np.sqrt(variance * (1.0 - rho**2))
    y = np.empty((n_traj, n_time))
    y[:, 0] = np.sqrt(variance) * rng.standard_normal(n_traj)
    for k in range(1, n_time):
        y[:, k] = rho * y[:, k - 1] + innovation * rng.standard_normal(n_traj)
    taus = [4.0, 16.0, 64.0]

    result = _scalar_allan(y, dt, taus=taus)

    for column, tau in enumerate(taus):
        m = int(tau / dt)
        interior = np.arange(1, m)
        block_variance = (variance / m**2) * (
            m + 2.0 * np.sum((m - interior) * rho**interior)
        )
        lags = np.arange(1, 2 * m)
        block_covariance = (variance / m**2) * np.sum(
            (m - np.abs(lags - m)) * rho**lags
        )
        expected = block_variance - block_covariance
        for mean_key, sem_key in (
            ("sampled_scalar_variance", "sampled_scalar_variance_sem"),
            ("overlap_sampled_scalar_variance", "overlap_sampled_scalar_variance_sem"),
        ):
            mean = result[mean_key][column]
            sem = result[sem_key][column]
            assert mean == pytest.approx(expected, rel=0.1)
            assert abs(mean - expected) < 6.0 * sem


def test_scalar_allan_overlap_matches_phase_frequency_path():
    """The overlapping scalar Allan of phase increments is the phase Allan.

    For ``y_k = (phi_{k+1} - phi_k) / dt`` the overlapping windowed
    difference ``y_bar_{j+m} - y_bar_j`` equals the phase second difference
    ``(phi_{j+2m} - 2*phi_{j+m} + phi_j) / tau`` exactly, so the frozen
    sampled-scalar definition reproduces the existing golden path.
    """
    rng = np.random.default_rng(20261016)
    n_traj, n_time, dt = 4, 2000, 0.1
    y = 0.3 * rng.standard_normal((n_traj, n_time))
    phase = np.concatenate(
        [np.zeros((n_traj, 1)), np.cumsum(y * dt, axis=1)], axis=1
    )
    field = np.exp(1j * phase)
    taus = [1.0, 2.0, 5.0]

    phase_result = calculate_allan_variance(
        field,
        dt,
        taus=taus,
        points=40,
        min_windows=8,
        min_independent_windows=4,
        amplitude_floor=0.0,
    )
    scalar_result = _scalar_allan(y, dt, taus=taus)

    np.testing.assert_array_equal(phase_result["tau"], scalar_result["tau"])
    np.testing.assert_allclose(
        scalar_result["overlap_per_trajectory"],
        phase_result["per_trajectory"],
        rtol=1e-12,
        atol=1e-30,
    )


def test_scalar_allan_tau_grid_constraints():
    y = np.random.default_rng(20261017).standard_normal((2, 512))

    # Unaligned taus are rejected.
    with pytest.raises(ValueError, match="not aligned"):
        _scalar_allan(y, 0.3, taus=[1.0])
    # A tau beyond the window constraints is rejected with the actual scalar
    # API parameter names (the phase path keeps its legacy error text).
    with pytest.raises(
        ValueError,
        match="an Allan tau violates min_windows or min_nonoverlap_pairs",
    ):
        _scalar_allan(y, 1.0, taus=[400.0], min_windows=8, min_nonoverlap_pairs=4)
    # The default logarithmic grid respects the same bounds: largest m keeps
    # at least min_windows overlapping pairs and min_nonoverlap_pairs+1
    # disjoint windows.
    result = calculate_sampled_scalar_allan(
        y,
        1.0,
        taus=None,
        points=6,
        min_windows=8,
        min_nonoverlap_pairs=4,
    )
    n_time = 512
    m_max = int(result["tau"][-1])
    assert n_time - 2 * m_max + 1 >= 8
    assert n_time // m_max - 1 >= 4


def test_scalar_allan_marks_nan_windows_invalid():
    """A non-finite sample censors the remainder of that trajectory.

    Block sums are cumulative, so the first non-finite sample truncates the
    usable record — the same censoring model ``valid_length`` expresses. With
    the NaN at sample 35 and ``m = 10``, non-overlapping pairs need block
    boundaries ``<= 35`` (only pairs 0 and 1 survive); overlapping pairs need
    their closing boundary ``j + 2*m <= 35``.
    """
    n_time, dt, m = 100, 1.0, 10
    y = np.random.default_rng(20261018).standard_normal((2, n_time))
    y[0, 35] = np.nan

    result = _scalar_allan(y, dt, taus=[m * dt])

    assert result["valid_window_pairs"][0, 0] == 2
    assert result["valid_window_pairs"][1, 0] == 9
    assert result["overlap_valid_window_pairs"][0, 0] == 16
    assert result["overlap_valid_window_pairs"][1, 0] == 81
    assert np.isfinite(result["per_trajectory"][0, 0])
    np.testing.assert_array_equal(result["trajectory_sample_count"], [2])


def test_scalar_allan_rejects_non_real_or_misshaped_input():
    with pytest.raises(ValueError, match="real"):
        calculate_sampled_scalar_allan(
            np.ones((2, 128), dtype=np.complex128),
            1.0,
            taus=[4.0],
            points=4,
            min_windows=2,
            min_nonoverlap_pairs=1,
        )
    with pytest.raises(ValueError, match="n_traj, n_time"):
        calculate_sampled_scalar_allan(
            np.ones(128),
            1.0,
            taus=[4.0],
            points=4,
            min_windows=2,
            min_nonoverlap_pairs=1,
        )
    with pytest.raises(ValueError, match="positive"):
        calculate_sampled_scalar_allan(
            np.ones((2, 128)),
            0.0,
            taus=[4.0],
            points=4,
            min_windows=2,
            min_nonoverlap_pairs=1,
        )


def test_scalar_allan_payload_is_frozen():
    """The scalar Allan table leaf set and definition strings are frozen.

    These dotted leaves become the concrete product variables once the
    analyser wiring lands (Wave 6D); the non-overlapping primary estimator
    carries the unsuffixed leaves.
    """
    result = _scalar_allan(
        np.random.default_rng(20261019).standard_normal((2, 512)), 1.0, taus=[4.0]
    )

    assert set(result) == {
        "quantity",
        "variable",
        "measurement_model",
        "tau_unit",
        "tau",
        "sampled_scalar_variance",
        "sampled_scalar_variance_sem",
        "per_trajectory",
        "valid_window_pairs",
        "trajectory_sample_count",
        "nominal_windows_per_trajectory",
        "total_valid_window_pairs",
        "overlap_sampled_scalar_variance",
        "overlap_sampled_scalar_variance_sem",
        "overlap_per_trajectory",
        "overlap_valid_window_pairs",
        "overlap_trajectory_sample_count",
        "overlap_total_valid_window_pairs",
        "observation_duration",
        "min_nonoverlap_pairs",
        "definition",
        "overlapping_definition",
    }
    assert result["quantity"] == "allan_variance"
    assert result["variable"] == "sampled_scalar"
    assert result["measurement_model"] == "sampled_scalar"
    # SDE time may be normalized; the canonical token is ``time`` and the
    # typed tau axis schema stays the unit source of truth once wired (6D).
    assert result["tau_unit"] == "time"
    assert result["definition"] == "nonoverlapping_window_mean_difference"
    assert (
        result["overlapping_definition"] == "overlapping_window_mean_difference"
    )
