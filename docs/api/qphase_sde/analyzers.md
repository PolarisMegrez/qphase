---
layout: default
title: Analyzers
parent: qphase_sde
grand_parent: API Reference
nav_order: 4
---

# Analyzers

Analyzers run after the SDE integration loop and produce payloads stored in `SDEResult.analysis`. They can also be invoked in `mode: analyze` for post-processing.

## `psd`

Estimates the power spectral density (PSD) of selected modes.

### Configuration

```yaml
analyser:
  psd:
    modes: [0]
    kind: complex
    expected_freq_max: 0.34
    find_peaks: true
    estimator:
      periodogram:
        window: null
```

| Key | Type | Description |
| :-- | :-- | :-- |
| `modes` | `list[int]` | Mode indices to analyze. |
| `kind` | `str` | PSD variant, e.g. `complex`, `real`, `imag`. |
| `expected_freq_max` | `float \| None` | Optional largest expected physical frequency in output-axis units. Analysis fails when it reaches the Nyquist limit. |
| `find_peaks` | `bool` | Whether to report peak locations. |
| `estimator` | child selection | Exactly one of `periodogram`, `welch`, or `multitaper`. |

Estimator comparison:

| Estimator | Child fields | Resolution and cost |
| :-- | :-- | :-- |
| `periodogram` | `window` | Uses the full saved duration; highest frequency resolution and largest per-trajectory FFT. |
| `welch` | `window`, `nperseg`, `noverlap`, `nfft` | Lower variance and bounded segment FFT memory; physical resolution is set by `nperseg * sample_dt`. `nfft > nperseg` only interpolates bins. |
| `multitaper` | `nw`, `k_tapers` | Full-duration resolution with taper averaging; additional FFT work per taper. |

All three support online aggregation across trajectory batches. The selected
estimator is a child plugin of `analyser.psd`, so it is discoverable with
`qphase list --parent analyser.psd` and queryable with
`qphase config schema analyser.psd/estimator.welch`.

### Frequency axis

The frequency grid depends on the saved trajectory:

```text
f = np.fft.fftfreq(n_saved, dt * save_stride) * 2 * pi
```

For a narrow peak, choose `save_stride` so the Nyquist frequency is well above the peak.
For angular-frequency conventions, `omega_Nyquist = pi / (dt * save_stride)`.
Increasing `t1` improves resolution `2*pi/t1` but does not increase this bandwidth.
Set `expected_freq_max` to turn an otherwise silent aliasing error into an explicit
configuration failure.

### Output payload

The analyzer exports:

*   `axis` — frequency or angular-frequency axis.
*   `psd` — mean PSD for each requested mode.
*   `psd_std` — sample standard deviation across trajectories (`ddof=1`).
*   `psd_sem` — standard error of the mean, `psd_std / sqrt(n_traj)`.
*   `uncertainty` — metadata identifying `psd_sem`, the independent unit, and sample count.

For Welch and multitaper estimates, segments or tapers are averaged within each
trajectory first. The uncertainty is then computed across trajectories, so
correlated segments are not counted as independent samples. With one trajectory,
`psd_std` and `psd_sem` contain `NaN` and uncertainty is marked unavailable.

When `find_peaks: true`, metadata also includes detected peak positions and heights.

## `coherence_matrix`

Computes the ensemble first-order coherence matrix

```text
R_ij = mean_trajectory,time(alpha_i * conj(alpha_j))
rho_R = R / Tr(R)
P_R = Tr(rho_R^2).
```

`P_R` is the purity of the normalized modal coherence matrix. It is not the
purity of the complete many-body quantum density operator.

```yaml
analyser:
  coherence_matrix:
    modes: [0, 1, 2]
    time_blocks: 8
    min_block_samples: 32
    confidence_level: 0.95
```

The output includes `matrix`, `normalized_matrix`, eigenvalues, `purity`,
effective rank, spectral entropy, principal-mode fraction, connected
covariance, and normalized first-order coherence. Matrix-element SEM is
computed across independent trajectory-level time averages. Purity uncertainty
uses a leave-one-trajectory-out jackknife and reports `purity_sem` and
`purity_ci`.

Contiguous `time_blocks` report matrices, purities, traces, and drift distances
for stationarity diagnostics. They are explicitly not treated as independent
samples. The analyser supports trajectory batching and can be used with
`keep_traj: false`; only compact matrix statistics are retained.

No phase-space ordering correction is applied. For Wigner trajectories the
payload therefore follows the raw amplitude convention used by the configured
model and CAM equations. A normal-order correction must be introduced through
an explicit model-aware analysis, not by silently subtracting `1/2` here.

All modes needed in `R` must be present in `engine.sde.record_modes`. Omit
`modes` to analyze every recorded mode.

## `moment_statistics`

Computes c-number occupation moments from recorded complex amplitudes:

```text
n_i = |alpha_i|^2
G2_ij = mean_trajectory,time(n_i n_j)
g2_ij = G2_ij / (mean(n_i) mean(n_j)).
```

```yaml
analyser:
  moment_statistics:
    modes: [0, 1, 2]
    time_blocks: 16
    time_chunk_samples: 8192
```

The result contains modal occupations, fourth moments, all cross-mode
occupation products, covariances, normalized `g2`, trajectory-level SEM, and
contiguous-block stationarity diagnostics. `g2_sem` uses a
leave-one-trajectory-out jackknife. The analyser supports trajectory batching
and recomputes nonlinear ratios after merging all batches.

`time_chunk_samples` bounds temporary backend memory. The analyser never
materializes the complete `|alpha|^2` trajectory, so its workspace is
independent of the total observation length once that chunk size is reached.
No Wigner-to-normal-order correction is applied; the payload explicitly uses
the configured model's raw c-number convention.

## `dist`

Computes marginal distributions of selected modes.

## `pdist`

Computes pairwise or higher-dimensional distributions for selected observables.

## `trajectory_diagnostics`

Computes time-domain diagnostics without assuming a Lorentzian or any other
spectral line shape. This is useful for separating nonstationarity,
trajectory-to-trajectory heterogeneity, loss of coherence, and phase-frequency
noise before interpreting a fitted PSD linewidth.

```yaml
analyser:
  trajectory_diagnostics:
    modes: [0]
    block_durations: [100.0, 1000.0]
    coherence: true
    coherence_max_lag: 500.0
    allan: true
    allan_taus: null
    allan_points: 24
    allan_min_windows: 8
    amplitude_floor: 0.0
```

The payload contains `mode_results[mode]` with:

*   `block_statistics`: per-trajectory, non-overlapping block means of the
    complex amplitude, amplitude, power, and angular frequency.
*   `phase_increment`: per-trajectory mean angular frequency, largest saved
    phase step, and fraction of steps within 10% of the Nyquist phase boundary.
*   `coherence`: complex `g1`, normalized `g1`, and cross-trajectory SEM
    magnitude versus lag.
*   `allan`: overlapping phase-second-difference angular-frequency Allan
    variance, per-trajectory values, ensemble mean, and cross-trajectory SEM.

All configured durations are physical times and must align with the saved
sample interval `dt * save_stride`. Auto-selected Allan averaging times are
logarithmically spaced integer sample counts. The SEM treats trajectories as
the independent units; overlapping time windows do not inflate the sample
count.

A Lorentzian core with HWHM close to a soft-mode decay rate describes the
stationary recovery after stochastic perturbations. It is not, by itself, the
one-time relaxation from the configured initial condition. A nonzero engine
`t0` removes that initial transient before this analyzer and the PSD run.

The first implementation materializes each supplied trajectory ensemble on the
host and does not provide online trajectory-batch aggregation. Use it for
reduced diagnostic jobs; streaming accumulation is reserved for a later phase.

## `allan_variance`

Computes focused angular-frequency Allan statistics when coherence and the
other trajectory diagnostics are not needed. Unlike `trajectory_diagnostics`,
this analyser supports trajectory batching and can be used with
`keep_traj: false`:

```yaml
analyser:
  allan_variance:
    modes: [0]
    points: 40
    min_windows: 8
    min_independent_windows: 4
```

For each tau it reports both the established overlapping estimate and a
non-overlapping phase-second-difference estimate. The payload includes the
per-trajectory values, cross-trajectory SEM, actual valid non-overlapping
window counts, and the nominal windows per trajectory. "Independent" here
means that the time blocks do not overlap; colored dynamics may still correlate
adjacent blocks. Trajectory bootstrap is therefore preferred over treating all
blocks as exchangeable samples.

## `allan_scaling`

Consumes a logical SDE scan dataset in `mode: analyze`. It detects a long-time
white-FM region at every scan point, intersects those regions across contiguous
perturbation points, computes
`N_A = tau * angular_frequency_allan_variance`, and fits the background-free
law `N_A = C * abs(epsilon) ** (-q)`. Per-trajectory bootstrap supplies the
exponent interval. The phase-increment mean frequency is fitted separately as
`omega = omega0 + A * abs(epsilon) ** p`. A linear model is retained only as a
null hypothesis for resolving nonlinearity; no linear correction is included
in the power-law model.

```yaml
analyser:
  allan_scaling:
    scan_param: omega_c
    critical_value: 1.0
    mode: 0
    min_scaling_points: 5
    target_scaling_decades: 1.0
    normal_form: {n: 3, k: 1, m: 0, observable_order: 2}
```

For the normal form `epsilon**k * x**m + x**n = 0`, the expected frequency
exponent is `observable_order * k / (n - m)`. With regular projected white
noise, the expected Allan-intensity exponent is
`2 * (n - observable_order) * k / (n - m)`. Final `status: ok` requires
sufficient epsilon span, an acceptable Allan power-law fit, resolved frequency
nonlinearity, and agreement with configured normal-form exponents. The analyser exports
`allan_points.csv` and `allan_scaling.json`. Legacy
`trajectory_diagnostics` payloads remain readable, but their independent
window counts are explicitly marked as estimated rather than measured.

## `lorentz_fitter`

Fits Lorentzians to PSD point views from a logical SDE scan dataset. It is a
downstream analyzer intended for `mode: analyze`.

### Configuration

```yaml
analyser:
  lorentz_fitter:
    scan_param: omega_a
    mode: 0
    uncertainty: auto
    fit_window: [0.1, 0.2]
    freq_min: -0.1
    freq_max: 0.1
    clip_by_std: true
    clip_sigma: 10.0
    min_r2: 0.5
    export:
      - fit_results.csv
      - psd_merged.csv
```

| Key | Type | Description |
| :-- | :-- | :-- |
| `scan_param` | `str` | Sweep parameter used to merge PSDs. |
| `mode` | `int` | Mode index to fit. |
| `uncertainty` | `auto \| required \| off` | `auto` propagates `psd_sem` to parameter covariance and falls back for legacy payloads; `required` rejects missing SEM; `off` uses residual covariance. It never changes fit weights. |
| `fit_window` | `list[float] \| None` | Manual `[min, max]` frequency window. If `None`, the window is derived from `freq_min`/`freq_max` or peak search. |
| `freq_min` / `freq_max` | `float \| None` | Optional global frequency bounds. |
| `clip_by_std` | `bool` | Enable squared-PSD-weighted clipping to ignore distant tails. |
| `clip_sigma` | `float` | Clip frequencies farther than `clip_sigma * std` from the squared-weighted mean. |
| `min_r2` | `float` | Minimum acceptable `R^2`. |
| `min_peak_height` | `float \| None` | Minimum fitted peak height. |
| `max_linewidth` | `float \| None` | Maximum acceptable FWHM linewidth. |
| `export` | `list[str]` | Artifacts to write. Defaults to `fit_results.csv`. |

### Output fields (`fit_results.csv`)

| Column | Meaning |
| :-- | :-- |
| `scan_param` | Named scan axis read from the logical SDE dataset. |
| `center` | Lorentzian peak center (rad/s). |
| `center_std` | Standard deviation of the fitted center. |
| `linewidth` | Full width at half maximum (FWHM). |
| `linewidth_std` | Propagated standard deviation of `2 * gamma`. |
| `base` | Constant baseline. |
| `base_std` | Standard deviation of the baseline. |
| `amplitude` | Lorentzian amplitude. |
| `amplitude_std` | Standard deviation of the amplitude. |
| `peak_intensity` | `amplitude + base`. |
| `peak_intensity_std` | Standard deviation including amplitude/base covariance. |
| `R2` | Coefficient of determination. |
| `reduced_chi2` | Reduced chi-square when fitting with `psd_sem`; otherwise `NaN`. |
| `uncertainty_source` | `psd_sem_sandwich` or the legacy `residual_covariance` fallback. |
| `status` | `ok` or `failed`. |
| `error` | Empty unless fitting failed. |
| `warning` | Diagnostics, e.g. std/FWHM mismatch. |

### Clipping rationale

PSD data often extends over a very wide frequency range determined by `dt`, while the peak is narrow. The analyzer computes the mean and standard deviation of the frequency axis weighted by `(PSD - min(PSD))^2`, then drops samples outside `mean ± clip_sigma * std`. This removes irrelevant tails while keeping the peak and enough nearby continuum for a stable baseline estimate.

A warning is emitted when the squared-weighted `std` deviates from the Lorentzian expectation `std ≈ linewidth / 2` by more than a factor of 2, which can indicate multiple peaks or insufficient frequency resolution.

PSD uncertainty does not change the unweighted `curve_fit` objective or fitted
parameters. Instead, the fitter evaluates the Lorentzian Jacobian at the fitted
parameters and propagates `psd_sem` with a heteroscedastic sandwich covariance.
This treats frequency bins as independent; windowing, leakage, and finite
trajectory dynamics can correlate neighboring bins, so the reported values are a
diagonal input-covariance approximation rather than a complete spectral covariance
model.
