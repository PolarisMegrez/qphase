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
```

| Key | Type | Description |
| :-- | :-- | :-- |
| `modes` | `list[int]` | Mode indices to analyze. |
| `kind` | `str` | PSD variant, e.g. `complex`, `real`, `imag`. |
| `expected_freq_max` | `float \| None` | Optional largest expected physical frequency in output-axis units. Analysis fails when it reaches the Nyquist limit. |
| `find_peaks` | `bool` | Whether to report peak locations. |

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

## `dist`

Computes marginal distributions of selected modes.

## `pdist`

Computes pairwise or higher-dimensional distributions for selected observables.

## `lorentz_fitter`

Fits a Lorentzian to aggregated PSD data. This is a **cross-job** analyzer intended for `mode: analyze`.

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
| `scan_param` | Sweep value from `aggregate_input`. |
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
