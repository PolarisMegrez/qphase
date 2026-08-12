---
description: Getting started with QPhase Projects and Workflows
---

# Getting Started

## Install

QPhase requires Python 3.11 or newer. Install core plus the resource packages
needed by the Project. From this monorepo:

```bash
git clone https://github.com/PolarisMegrez/qphase.git
cd qphase
uv sync
```

## Create A Project

```bash
qphase project init my-research --name "My Research"
cd my-research
qphase project show
```

Initialization creates:

```text
qphase.toml                    # Project identity and relative paths
configs/defaults.yaml         # Project-wide plugin defaults
configs/workflows/            # Versioned Workflow documents
models/                       # Local plugin root
runs/                         # Session records (normally not versioned)
```

Use `qphase --project <path> ...` when invoking QPhase outside the Project
directory. Project discovery otherwise walks upward until it finds
`qphase.toml`.

## Create A Workflow

Create `configs/workflows/examples/test_run.yaml`:

```yaml
schema: qphase.workflow/2
id: test_run
title: First SDE run
description: Small CPU example for installation verification
collection: examples
tags: [quickstart, sde]

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
      numpy: {float_dtype: float64}
    integrator:
      euler_maruyama: {}
    model:
      vdp_2mode:
        omega_a: 1.0
        omega_b: 1.0
        gamma_a: 0.1
        gamma_b: 0.1
        Gamma: 1.0
        g: 0.5
```

The document itself is a Workflow. `simulate` is its logical Job. A Workflow
may contain several Jobs connected by `input` or `depends_on`.

## Inspect And Run

```bash
qphase workflow list
qphase workflow show test_run
qphase run test_run --plan
qphase run test_run
```

`--plan` validates and displays the logical Job graph without creating a
Session. A normal run prints a concise progress view and finally reports the
Session path. Full diagnostic logs are stored in that Session according to
`SystemConfig`.

## Find Results

Sessions use a bounded date hierarchy:

```text
runs/YYYY/MM/<session-id>/
  session_manifest.json
  events.jsonl
  <job-name>/
    config_snapshot.json
    artifact_manifest.json
    qphase.log
    ...
```

Use `qphase gui` for visual Workflow and Session browsing. The CLI remains the
authoritative interface for scripting and remote/server use.

## Next Steps

- Read [Core Concepts](concepts.md) before designing automation around QPhase.
- See [Workflow Configuration](configuration.md) for scans and data flow.
- See [Results & Reproducibility](output.md) for Artifact layouts.
- Use `qphase list`, `qphase show`, and `qphase config schema` to inspect
  installed plugins.
