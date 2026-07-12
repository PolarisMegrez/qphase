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
    │   ├── config_snapshot.yaml     # Full configuration used for this job
    │   ├── vdp_sde.npz              # SDE result archive
    │   └── qphase.log               # Execution log
    └── vdp_viz/                     # Downstream Job Directory
        ├── config_snapshot.yaml
        └── plot.png
```

## Reproducibility

### Configuration Snapshots (`config_snapshot.yaml`)

Every job directory contains a `config_snapshot.yaml`. This file is the **exact** configuration used to run the job, including:
*   Merged values from Global and Job configs.
*   Resolved default values for plugins.
*   System environment information (QPhase version, Python version, OS).

To reproduce a result, you can simply run this snapshot file:

```bash
qphase run runs/2025-12-31.../vdp_sde/config_snapshot.yaml
```

### Session Manifest (`session_manifest.json`)

The session manifest tracks the status and relationships of all jobs in a session. It is useful for:
*   Debugging failed pipelines.
*   Programmatically analyzing run history.
*   Resuming interrupted sessions (advanced usage).

## Data Formats

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

*   `fit_results.csv`: one row per scan value. Columns are `job_name`, the scan parameter, Lorentzian `center`, `linewidth`, `base`, `peak_intensity`, `R2`, `status`, and `error`. `status` is `ok`, `low_quality` (when a quality threshold is violated), or `failed`.
*   `psd_merged.csv`: a frequency-indexed table with one PSD column per scan value.
*   `dist_merged.npz` (experimental): written when `export_dist: true` is set. Keys are `dist_list`, `scan_params`, `__schema_version__`, and `__created_by__`.
*   `pdist_merged.pkl` (experimental): written when `export_dist: true` is set. It is a pickled dict with `rows`, `__schema_version__`, and `__created_by__`.

Common analyzer options include `output_dir`, `psd_key`, `fit_window`, `freq_min`, `freq_max`, `min_r2`, `min_peak_height`, `max_linewidth`, `export_dist`, `clip_by_std`, and `clip_sigma`. Set `clip_by_std: true` to first clip the frequency window to the squared-PSD-weighted mean ± `clip_sigma` standard deviations, which helps ignore distant long-tail bumps and speeds up fitting on wide grids.

For a Lorentzian, the squared-PSD-weighted standard deviation equals `linewidth / 2`. The default `clip_sigma: 10.0` therefore keeps approximately `±5 × FWHM` around the peak, which is wide enough to capture the line shape while still excluding very distant artifacts.

The fit result table also contains `amplitude` (height above baseline), `peak_intensity` (total height), `R2`, `status`, `error`, and `warning`. The `warning` field is populated when the squared-PSD-weighted standard deviation of the input differs from the Lorentzian expectation (`std = linewidth / 2`) by more than a factor of two, suggesting the data may not be single-peaked Lorentzian.

Example workflow:

```yaml
- name: sim
  save: true
  engine:
    sde: { t0: 0.0, t1: 1.0, dt: 0.01, n_traj: 8, seed: 42 }
  model:
    kerr_3pa:
      epsilon: [0.025, 0.05, 0.1]
  analyser:
    psd: { modes: [0], kind: complex, find_peaks: true }

- name: fit
  input: sim
  aggregate_input:
    on: epsilon
  engine:
    sde: { mode: analyze }
  analyser:
    lorentz_fitter:
      scan_param: epsilon
      mode: 0
```
