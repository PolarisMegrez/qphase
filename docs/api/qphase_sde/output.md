---
layout: default
title: Output Formats
parent: qphase_sde
grand_parent: API Reference
nav_order: 5
---

# Output Formats

The SDE engine produces two kinds of artifacts: per-run archives and merged analysis bundles.

## Per-run archive (`.npz`)

When `save: true`, each SDE job writes a NumPy archive containing:

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

When the analyzer runs in multiple jobs with different parameter values, the scheduler can aggregate them into a single table for the `lorentz_fitter` analyzer.

## Lorentz fit output

`lorentz_fitter` writes up to three artifacts depending on `export`:

*   `fit_results.csv` — one row per scan point with columns described in [Analyzers](./analyzers.md).
*   `psd_merged.csv` — merged PSD table plus `<scan_value>_sem` columns used for weighted fitting.
*   `fit_results.npz` / `fit_results.pkl` — same data in alternative formats.

## Distribution outputs

*   `dist_merged.npz` — saved by the `dist` analyzer when aggregated across a scan.
*   `pdist_merged.pkl` — saved by the `pdist` analyzer when aggregated across a scan.

For details on the run directory layout, see [User Guide: Output](../../user_guide/output.md).
