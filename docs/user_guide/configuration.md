---
description: Project, Workflow, Job, and system configuration
---

# Workflow Configuration

QPhase separates portable research intent from machine policy:

1. Plugin schema defaults are defined by installed plugins.
2. Project plugin defaults live at the `qphase.toml` `paths.defaults` location,
   normally `configs/defaults.yaml`.
3. A versioned Workflow document defines metadata and logical Jobs.
4. `job.system` may override the same `SystemConfig` schema for one Job.

Later layers have higher precedence. Project paths never belong to
`SystemConfig`; they are defined only in `qphase.toml`.

## Project Manifest

```toml
schema = "qphase.project/2"
project_id = "my-research"
name = "My Research"

[paths]
workflows = "configs/workflows"
defaults = "configs/defaults.yaml"
plugins = ["models"]
sessions = "runs"
```

All paths are portable paths relative to the Project root. Absolute paths and
`..` traversal are rejected. Project discovery uses `--project`, then
`QPHASE_PROJECT`, then upward search from the current directory.

## Workflow Document

```yaml
schema: qphase.workflow/2
id: vdp_cam
title: VDP CAM scan
description: Optional human-readable purpose
collection: vdp_2mode
tags: [cam, multistability]

jobs:
  - name: solve
    save: true
    engine:
      cam: {}
    backend:
      numpy: {float_dtype: float64}
    model:
      vdp_2mode:
        omega_a: 0.0
        omega_b: 0.0
        gamma_a: 2.0
        gamma_b: 0.5
        Gamma: 0.0001
        g: 0.5
    cam_solver:
      multistability: {n_guesses: 50, guess_bounds: auto}
```

`schema`, `id`, `title`, and a non-empty `jobs` list are required. Workflow IDs
must be unique within a Project and remain stable when a file moves. A document
without the `qphase.workflow/2` wrapper is rejected; QPhase 2 does not silently
load legacy top-level Job lists.

Collection and Tags are portable metadata. The directory hierarchy may mirror
Collection for readability but does not define identity.

## Logical Job

| Field | Meaning |
| --- | --- |
| `name` | Unique Job name inside the Workflow. |
| `engine` | Exactly one engine and its configuration. |
| plugin namespaces | `backend`, `model`, `integrator`, `analyser`, and other engine-specific plugin slots. |
| `params` | Optional engine-specific values. |
| `scan` | Optional explicit `ScanSpec`. |
| `input` | Optional structured upstream data input. |
| `depends_on` | Explicit control dependency on other Jobs. |
| `save` | `true`, `false`, or an Artifact base name. |
| `system` | Optional per-Job override of `SystemConfig`. |

Project-wide numerical defaults belong in `configs/defaults.yaml`. Optional
plugins are not activated merely because defaults exist; the Workflow must
select the plugin namespace.

## Parameter Scans

A plugin list is always a literal value. Scans must use `ScanSpec`:

```yaml
scan:
  combine: cartesian
  axes:
    omega_a:
      target: model.vdp_2mode.omega_a
      logspace: {start: -3, stop: -1, num: 31}
    gamma_b:
      target: model.vdp_2mode.gamma_b
      linspace: {start: 0.2, stop: 1.1, num: 101}
```

Each axis uses exactly one generator:

| Generator | Meaning |
| --- | --- |
| `values: [...]` | Explicit values. |
| `linspace: {start, stop, num, endpoint}` | Linear spacing; `endpoint` defaults to `true`. |
| `logspace: {start, stop, num, endpoint, base}` | Exponents of `base`; `base` defaults to `10`. |

`cartesian` produces dimensions in YAML axis order. `zipped` requires equal
axis lengths and produces one `point` dimension. A scan remains one logical Job:
the engine receives `ParameterGrid` and chooses pointwise, tiled, fused, or GPU
execution. Core does not create one Job or Session per parameter point.

## Data Flow

```yaml
input:
  from: simulate
  mode: dataset
```

`dataset` passes the complete upstream result once. `map` lazily presents
point/group views inside one downstream Job:

```yaml
input:
  from: source_scan
  mode: map
  select: {omega_a: [0.01, 0.1]}
  group_by: [gamma_b]
```

String-valued `input` and `aggregate_input` are removed.

## SystemConfig

`SystemConfig` is project-independent machine policy. Resolution order is:

1. packaged `qphase.core/system.yaml`;
2. optional site policy (`/etc/qphase/config.yaml` or `%PROGRAMDATA%`);
3. sparse user override `~/.qphase/config.yaml`;
4. `QPHASE_SYSTEM_CONFIG`;
5. an explicit loader path.

It contains result auto-save policy, scan Artifact layout/checkpoints/resource
hints, progress rendering, and logging. It contains no Workflow, plugin, or
Project paths. `qphase config show --system` displays the resolved policy;
`qphase config show` displays Project plugin defaults.

```yaml
auto_save_results: true
reporting:
  progress:
    refresh_interval: 0.5
    non_tty_milestone_percent: 10.0
  logging:
    session_file: true
    filename: qphase.log
    file_level: DEBUG
    console_level: WARNING
scan_runtime:
  storage_layout: auto
  auto_shard_threshold_mib: 512
  shard_target_mib: 128
  checkpoint:
    enabled: false
    interval_chunks: 1
    keep_on_success: false
  resources:
    cpu_worker_limit: null
    memory_limit_mib: null
    gpu_device: null
    gpu_memory_fraction: null
```

Checkpointing covers completed scan chunks, not internal SDE time steps.
