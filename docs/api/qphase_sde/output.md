---
layout: default
title: Output Formats
parent: qphase_sde
grand_parent: API Reference
nav_order: 5
---

# Output Formats

The SDE engine produces one logical result per job. A non-scan result is an
`SDEResult`; a scan is an `SDEScanResult` dataset with named axes and lazy point
views.

## Single dataset archive (`.npz`)

With `storage_layout: single`, each SDE job writes one NumPy archive containing:

| Key | Type | Description |
| :-- | :-- | :-- |
| `t0` | `float` | Start time. |
| `dt` | `float` | Saved sample spacing (`dt * save_stride`). |
| `meta` | `object` | Metadata such as model parameters and discard reason. |
| `analysis` | `object` | Analyzer payloads keyed by analyzer name. |
| `data` | `ndarray` | Raw trajectory, shape `(n_traj, n_saved, n_modes)`. Present only if `keep_traj: true` or the analyzer requested it. |

Load it from Python:

```python
import numpy as np
archive = np.load("run.npz", allow_pickle=True)
meta = archive["meta"].item()
psd = archive["analysis"].item().get("psd")
```

## PSD output

The `psd` analyzer stores:

*   `axis` — frequency or angular-frequency vector.
*   `psd` — mean PSD values per mode.
*   `psd_std` / `psd_sem` — cross-trajectory sample standard deviation and standard error.
*   `orientation` and its defining formula fields — the sign convention of the
    frequency axis. Missing metadata identifies a legacy forward-FFT result and
    must be interpreted as `phase_increasing`.

For a scan, PSD payloads remain attached to the named points of one logical SDE
dataset. A downstream `mode: analyze` job consumes that dataset once with the
`lorentz_fitter` analyzer.

With `storage_layout: sharded`, the same logical dataset is split into bounded
`shard_*.npz` files under one job directory. `artifact_manifest.json` records
the shape and loader, and `SDEScanResult.load_dataset` restores the full view.

## Lorentz fit output

`lorentz_fitter` writes up to three artifacts depending on `export`:

*   `fit_results.csv` — one row per scan point with columns described in [Analyzers](./analyzers.md).
*   `psd_merged.csv` — merged PSD table plus `<scan_value>_sem` columns used for uncertainty propagation.
*   `fit_results.npz` / `fit_results.pkl` — same data in alternative formats.

## Allan output

`allan_variance` remains inside the logical SDE dataset. It stores overlapping
and non-overlapping Allan variance, per-trajectory estimates, SEM, and valid
window counts, but not the original complex trajectories when `keep_traj` is
false. The downstream `allan_scaling` analyser writes:

*   `allan_points.csv` — one row per perturbation with the detected white-FM tau
    range, `tau * sigma_A^2`, SEM, effective non-overlapping window count, mean
    angular frequency, and gate status.
*   `allan_scaling.json` — selected common tau/epsilon windows, pure-power Allan
    fit, frequency-response fit, bootstrap interval, normal-form expectations,
    and explicit gate failures.

## Distribution outputs

*   `dist_merged.npz` — optional merged distribution export for a scan dataset.
*   `pdist_merged.pkl` — optional merged polar-distribution export for a scan dataset.

For details on the Session and Job directory layout, see [User Guide: Output](../../user_guide/output.md).
