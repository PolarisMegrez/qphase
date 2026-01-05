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
    │   ├── data.npz                 # Simulation results (format depends on engine)
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

*   **SDE Engine**: Typically saves as `.npz` (NumPy compressed archive) or `.h5` (HDF5), containing trajectories and time points.
*   **Viz Engine**: Saves images (`.png`, `.pdf`) or processed data files.

The output format can be configured in the job configuration under the engine settings.
