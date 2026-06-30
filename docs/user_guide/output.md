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

*   **SDE Engine**: Saves a NumPy `.npz` archive. The archive contains `t0`, `dt`, `meta`, `analysis`, and, when trajectory retention is enabled, raw `data` with shape `(n_traj, n_time, n_modes)`.
*   **Viz Engine**: Saves images (`.png`, `.pdf`) or processed data files.

The output format can be configured in the job configuration under the engine settings.

### SDE Analysis Payloads

When an SDE job config includes `analyser` plugins, their outputs are stored under `analysis` by analyser key.

*   `analysis["psd"]`: `axis`, `psd`, `modes`, `kind`, `convention`, and `peaks`. The PSD matrix is shaped `(n_frequency, n_modes)`.
*   `analysis["dist"]`: Cartesian phase-space distribution payloads keyed by mode.
*   `analysis["pdist"]`: Polar distribution payloads keyed by mode.

If `engine.sde.keep_traj` is unset, the engine drops raw trajectories after analysis to reduce file size. In that case the `.npz` still contains `meta`, `analysis`, `t0`, and `dt`.

## Postprocessing Exports

Use `qphase postprocess` to turn saved PSD analysis into stable CSV files:

```bash
qphase postprocess runs/2026-03-17T21-03-06_088ab0 --scan-param epsilon --mode 0
```

The command consumes existing `analysis["psd"]` data; it does not recompute PSD from trajectories. It writes:

*   `fit_results.csv`: one row per job, including the scan parameter, Lorentzian `center`, `linewidth`, `base`, `peak_intensity`, `R2`, `status`, and `error`.
*   `psd_merged.csv`: a frequency-indexed table with one PSD column per scan value.
*   `dist_merged.npz` and `pdist_merged.pkl`: optional experimental distribution exports when `--export-dist` is passed.

Useful options include `--output-dir`, `--psd-key`, `--fit-window`, `--overwrite`, and `--export-dist`.
