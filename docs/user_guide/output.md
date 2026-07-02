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

The output format can be configured in the job configuration under the engine settings.

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

Use `qphase postprocess` to turn saved PSD analysis into stable CSV files:

```bash
qphase postprocess runs/2026-03-17T21-03-06_088ab0 --scan-param epsilon --mode 0
```

The command consumes existing `analysis["psd"]` data; it does not recompute PSD from trajectories. It writes:

*   `fit_results.csv`: one row per job. Columns are `job_name`, the scan parameter, Lorentzian `center`, `linewidth`, `base`, `peak_intensity`, `R2`, `status`, and `error`. `status` is `ok`, `low_quality` (when a quality threshold is violated), or `failed`.
*   `psd_merged.csv`: a frequency-indexed table with one PSD column per scan value.
*   `dist_merged.npz` (experimental): written when `--export-dist` is passed. Keys are `dist_list`, `scan_params`, `__schema_version__`, and `__created_by__`.
*   `pdist_merged.pkl` (experimental): written when `--export-dist` is passed. It is a pickled dict with `rows`, `__schema_version__`, and `__created_by__`.

Common options include `--output-dir`, `--psd-key`, `--fit-window`, `--freq-min`, `--freq-max`, `--min-r2`, `--min-peak-height`, `--max-linewidth`, `--overwrite`, `--export-dist`, and `--dry-run`.
