---
description: Results & Reproducibility
---

# Results & Reproducibility

QPhase is designed with reproducibility in mind. Every simulation run generates a structured output directory containing not just the results, but also the full context required to reproduce them.

## Directory Structure

By default, all outputs are stored in the `runs/` directory.

### Session-Based Execution

Executing `qphase run` initiates a new **Session**. A session acts as a container for all jobs executed in that command.

```text
runs/
└── 2025-12-31T05-23-05_281415/      # Session Directory (Timestamp + UUID)
    ├── session_manifest.json        # Metadata for the entire session
    ├── vdp_sde/                     # Job Directory (Job Name)
    │   ├── config_snapshot.json     # Full configuration used for this job
    │   ├── artifact_manifest.json   # Logical result and physical layout
    │   ├── vdp_sde.npz              # SDE result archive, single layout
    │   └── qphase.log               # Execution log
    └── vdp_viz/                     # Downstream Job Directory
        ├── config_snapshot.json
        ├── artifact_manifest.json
        └── plot.png
```

## Reproducibility

### Configuration Snapshots (`config_snapshot.json`)

Every job directory contains a `config_snapshot.json`. This file records the
resolved logical-job configuration, including:
*   Merged values from Global and Job configs.
*   Resolved default values for plugins.
*   System environment information (QPhase version, Python version, OS).

To reproduce a result, you can simply run this snapshot file:

```bash
qphase run runs/2025-12-31.../vdp_sde/config_snapshot.json
```

### Session Manifest (`session_manifest.json`)

The session manifest tracks the status and relationships of all jobs in a session. It is useful for:
*   Debugging failed pipelines.
*   Programmatically analyzing run history.
*   Resuming interrupted sessions (advanced usage).

## Data Formats

### Logical Dataset Artifacts

A scan is stored as one logical dataset. It does not create one run directory
per parameter point. Every saved result has an `artifact_manifest.json` with
the result type, schema version, named axes, logical shape, storage layout,
physical files, and loader.

The layout is selected by `system.scan_runtime.storage_layout`:

* `single`: one primary dataset file.
* `sharded`: a bounded collection of chunk files under the same job directory.
* `per_point`: compatibility export only; files still remain under one job directory.
* `auto`: use `single` below `auto_shard_threshold_mib` and `sharded` above it.

The packaged automatic threshold is 512 MiB and the target shard size is 128
MiB. CAM and SDE dataset loaders restore the logical shape independently of the
physical layout.

The format of the result data depends on the Engine used.

*   **SDE Engine**: Saves a NumPy `.npz` archive. Top-level keys are `t0`, `t1`, `dt`, `meta`, `analysis`, and, when trajectory retention is enabled, raw `data` with shape `(n_traj, n_time, n_modes)`.
*   **Viz Engine**: Saves images (`.png`, `.pdf`) or processed data files.

The output format can be configured in the job configuration under the engine settings. For the detailed schema of SDE artifacts, see [Output Formats in the `qphase_sde` reference](../api/qphase_sde/output.md).

### SDE Analysis Payloads

When an SDE job config includes `analyser` plugins, their outputs are stored under `analysis` by analyser key.

*   `analysis["psd"]`:
    *   `axis`: 1-D frequency axis.
    *   `psd`: PSD matrix of shape `(n_frequency, n_modes)`.
    *   `modes`: list of analyzed mode indices.
    *   `kind`: `"complex"` or `"modular"`.
    *   `convention`: `"symmetric"`, `"unitary"`, or `"pragmatic"`.
    *   `peaks`: dict mapping each mode to serialized `PeakInfo` with `indices`, `frequencies`, `values`, and `properties`.
*   `analysis["dist"]`:
    *   `distributions`: dict mapping each mode to a histogram result. Complex modes use 2-D histograms (`hist`, `xedges`, `yedges`, `type="2d_complex"`); real modes use 1-D histograms (`hist`, `edges`, `type="1d_real"`).
    *   `modes`: list of analyzed mode indices.
    *   `bins`: number of bins used.
    *   `density`: whether histograms are normalized to PDFs.
*   `analysis["pdist"]`: experimental polar-distribution payload with the same high-level structure (`distributions`, `modes`, `bins_config`, `density`) when a polar-distribution analyser is configured.

If `engine.sde.keep_traj` is unset, the engine drops raw trajectories after analysis to reduce file size. In that case the `.npz` still contains `meta`, `analysis`, `t0`, `t1`, and `dt`.

## Postprocessing Exports

Cross-job postprocessing is implemented as a scheduler workflow using the
`analyser.lorentz_fitter` plugin with `engine.sde.mode: analyze`. The analyzer
consumes existing `analysis["psd"]` data and does not recompute PSD from
trajectories. It writes:

*   `fit_results.csv`: one row per scan value. Each Lorentzian parameter is paired with a `_std` column derived from the fit covariance; `uncertainty_source` records whether covariance propagated `psd_sem` or used the legacy residual fallback. PSD uncertainty does not reweight the Lorentz fit. `status` is `ok`, `low_quality` (when a quality threshold is violated), or `failed`.
*   `psd_merged.csv`: a frequency-indexed table with one PSD column per scan value and a `<scan_value>_sem` column when PSD standard errors are available.
*   `dist_merged.npz` (experimental): written when `export_dist: true` is set. Keys are `dist_list`, `scan_params`, `__schema_version__`, and `__created_by__`.
*   `pdist_merged.pkl` (experimental): written when `export_dist: true` is set. It is a pickled dict with `rows`, `__schema_version__`, and `__created_by__`.

Common analyzer options include `output_dir`, `psd_key`, `fit_window`, `freq_min`, `freq_max`, `min_r2`, `min_peak_height`, `max_linewidth`, `uncertainty`, `export_dist`, `clip_by_std`, and `clip_sigma`. The default `uncertainty: auto` uses PSD standard errors when available while remaining compatible with older result files. Set `clip_by_std: true` to first clip the frequency window to the squared-PSD-weighted mean ± `clip_sigma` standard deviations, which helps ignore distant long-tail bumps and speeds up fitting on wide grids.

For a Lorentzian, the squared-PSD-weighted standard deviation equals `linewidth / 2`. The default `clip_sigma: 10.0` therefore keeps approximately `±5 × FWHM` around the peak, which is wide enough to capture the line shape while still excluding very distant artifacts.

The fit result table also contains `amplitude` (height above baseline), `peak_intensity` (total height), `R2`, `status`, `error`, and `warning`. The `warning` field is populated when the squared-PSD-weighted standard deviation of the input differs from the Lorentzian expectation (`std = linewidth / 2`) by more than a factor of two, suggesting the data may not be single-peaked Lorentzian.

Example workflow:

```yaml
- name: sim
  save: true
  scan:
    axes:
      omega_a:
        target: model.kerr_2mode.omega_a
        values: [0.9, 1.0, 1.1]
  engine:
    sde: { t0: 0.0, t1: 1.0, dt: 0.01, n_traj: 8, seed: 42 }
  model:
    kerr_2mode:
      omega_a: 0.9
      omega_b: 1.0
      chi: 0.01
      gamma_a: 0.1
      gamma_b: 0.1
      g: 0.1
  analyser:
    psd: { modes: [0], kind: complex, find_peaks: true }

- name: fit
  input:
    from: sim
    mode: dataset
  engine:
    sde: { mode: analyze }
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```
