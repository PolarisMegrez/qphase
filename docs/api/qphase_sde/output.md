---
layout: default
title: Output Formats
parent: qphase_sde
grand_parent: API Reference
nav_order: 5
---

# Output Formats

The SDE engine produces one logical result per job: an `SDEDataBundle` of named, typed data products (trajectories plus analyzer products). A scan is one bundle whose products carry named parameter axes; `point_view(index)` yields lazily backed per-point views.

## Artifact directory

Each saved SDE job writes one Artifact v4 directory:

| Entry | Description |
| :-- | :-- |
| `artifact_manifest.json` | Validated manifest (`qphase.artifact/4`): full product schemas, the `sde.bundle/1` bundle descriptor (scan grid, product roles), provenance, and per-variable payload references. |
| `<stem>.npz` | `storage_layout: single`: one file per product holding every variable as a declared key. |
| `<stem>__<variable>.npz` / `<stem>__<variable>__<NNNN>.npz` | Default/sharded layout: one `"data"` key per file, variables split into byte-bounded chunks. |

Payload arrays are stored in native dtypes — restoring never needs
`allow_pickle`. Load it from Python:

```python
from qphase.data import load_bundle

bundle = load_bundle("runs/2026/08/<session-id>/sim")
psd = bundle.products["psd"]      # lazily backed; no payload read yet
point = bundle.point_view((0,))   # per-scan-point view of the bundle
```

## PSD output

The `psd` analyzer stores:

*   `axis` — frequency or angular-frequency vector.
*   `psd` — mean PSD values per mode.
*   `psd_std` / `psd_sem` — cross-trajectory sample standard deviation and standard error.
*   `orientation` and its defining formula fields — the sign convention of the
    frequency axis. Missing metadata identifies a legacy forward-FFT result and
    must be interpreted as `phase_increasing`.

For a scan, the PSD product carries the named scan axis of the logical SDE
bundle. A downstream `mode: analyze` job consumes that bundle once with the
`lorentz_fitter` analyzer.

With `storage_layout: sharded`, the same logical dataset is split into bounded
chunk files under one job directory. The manifest records the chunk layout and
the `npz/3` storage adapter id; `qphase.data.load_bundle` restores the full
bundle lazily regardless of the physical layout.

## Lorentz fit output

`lorentz_fitter` writes up to three artifacts depending on `export`:

*   `fit_results.csv` — one row per scan point with columns described in [Analyzers](./analyzers.md).
*   `psd_merged.csv` — merged PSD table plus `<scan_value>_sem` columns used for uncertainty propagation.
*   `fit_results.npz` / `fit_results.pkl` — same data in alternative formats.

## Band-limited carrier output

`band_limited_carrier` writes three auditable tables when their names appear in
`export`:

*   `carrier_results.csv` — local resolution status plus the separately tracked
    scan carrier. A local or tracked frequency may be `NaN` when the data do not
    support a unique platform.
*   `carrier_candidates.csv` — every bandwidth/lag fit, including phase
    residual, frequency drift, decay rate, lag bounds, and rejection reason.
*   `carrier_platforms.csv` — every frequency platform retained before
    scan-level path tracking, including support and score.

The reported diagnostic uncertainty is conditional estimator sensitivity, not
a trajectory SEM. The output metadata states this explicitly.

In ridge-conditioned mode, all three tables additionally carry
`ridge_candidate_index`, `ridge_retention_tier`, ridge-center uncertainty,
nearest-ridge bandwidth limits, `ridge_carrier_correction`, and
`ridge_conditioned_uncertainty_upper`. There can be several rows per scan point.

`finite_delay_carrier` writes `finite_delay_carrier.csv`, one row per scan point,
readout, and detector rate. It contains the measurement name and kind, direct
detector carrier, its zero-delay SDE limit, finite-delay correction, coherent
weight, and numerical lag range.

`spectral_ridge` writes `spectral_ridge.csv`, one selected ridge per scan point
and readout, and `spectral_ridge_candidates.csv`, the complete scale-supported
candidate set. The selected table includes peak frequency, local and scale
uncertainty, curvature diagnostics, PSD-SEM confidence bounds, relative-height
plateau bounds, competing-ridge ambiguity bounds, path-selection index, and
status. Candidate rows additionally contain `retained_for_association`,
`retention_tier`, `tracking_path_ranks`, and `tracking_best_cost_delta`.

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
