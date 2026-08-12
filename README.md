# QPhase

QPhase is a project-oriented scientific workflow runtime for reproducible,
workstation- and lab-server-scale numerical research. The core package owns
project discovery, versioned workflow configuration, plugin registration,
logical-job scheduling, progress, cancellation, logging, and artifact
persistence. Resource packages own scientific algorithms and their internal
CPU/GPU parallelism.

Authors: Yu Xue-Hao (University of Chinese Academy of Sciences, UCAS)

## Core Vocabulary

- **Project**: the portable research boundary declared by `qphase.toml`.
- **Workflow**: one versioned YAML document containing a logical Job graph.
- **Job**: one logical node in that graph; a parameter scan remains one Job.
- **Execution**: one queued or running attempt to execute a Workflow.
- **Session**: the persisted record of one Execution attempt.
- **Artifact**: a typed output produced by a Job.
- **Collection** and **Tag**: portable Workflow organization metadata.
- **Archive**: user-local organization metadata; it is not part of the
  reproducible Project contract.

## Installation

Python 3.11 or newer is required. For this monorepo:

```powershell
git clone https://github.com/PolarisMegrez/qphase.git
cd qphase
uv sync
```

The workspace contains:

- `packages/qphase`: core runtime, CLI, local API, and GUI.
- `packages/qphase_sde`: stochastic differential-equation engine.
- `packages/qphase_cam`: workspace-only coherent-amplitude-matrix engine.
- `packages/qphase_viz`: visualization resource package.

## Project Layout

`qphase.toml` is the only source of Project storage paths. The standard layout
is:

```text
qphase.toml
configs/
  defaults.yaml
  workflows/<collection>/*.yaml
models/
runs/YYYY/MM/<session-id>/
```

Machine policy such as memory limits, GPU selection, progress, and logging is
kept separately in `~/.qphase/config.yaml`. It must not contain Project paths.

Create or inspect a Project:

```powershell
qphase project init my-research
qphase --project my-research project show
```

## Run A Workflow

List Workflows recursively by stable ID, inspect one, and execute it:

```powershell
qphase workflow list
qphase workflow show vdp_2mode_smoke
qphase run vdp_2mode_smoke --plan
qphase run vdp_2mode_smoke
```

The same commands work outside the Project when `--project <root>` is supplied.
A Workflow path may be used for development, but stable IDs are the normal CLI
and GUI contract.

A minimal Workflow document is:

```yaml
schema: qphase.workflow/2
id: demo_psd
title: Demo PSD
collection: examples
tags: [sde, quickstart]
jobs:
  - name: simulate
    save: true
    engine:
      sde:
        t0: 0.0
        t1: 10.0
        dt: 0.01
        n_traj: 16
        seed: 42
        ic: [["1.0+0.0j", "0.0+0.0j"]]
    backend:
      numpy: {}
    integrator:
      euler_maruyama: {}
    model:
      kerr_2mode:
        omega_a: 1.0
        omega_b: 1.0
        chi: 0.01
        gamma_a: 0.1
        gamma_b: 0.1
        g: 0.1
    analyser:
      psd: {modes: [0], kind: complex}
```

Each Execution creates one Session under `runs/YYYY/MM/`. A Session contains a
manifest, one directory per logical Job, resolved snapshots, logs, and Artifact
manifests. A scan does not create scheduler sub-jobs or parameter-point Session
directories.

## Interfaces

The CLI is the primary and complete interface for researchers and automation.
The local GUI provides visual Workflow and Session management using the same
service layer:

```powershell
qphase gui
```

See the documentation for Workflow configuration, Artifacts, plugin contracts,
and engine-specific numerical behavior.

## License

MIT License. See `LICENSE` for details.
