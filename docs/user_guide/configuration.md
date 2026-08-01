---
description: Job Configuration
---

# Job Configuration

QPhase uses validated YAML configuration for reproducible logical jobs. A job
selects one engine, configures its plugins, optionally declares a parameter
scan, and optionally consumes an upstream result.

## Configuration Hierarchy

Configuration is merged in this order, from lowest to highest priority:

1. Core and plugin schema defaults.
2. Project defaults from `configs/global.yaml`.
3. The job YAML.
4. `job.system`, which overrides the same `SystemConfig` fields for that job.

Framework runtime policy belongs to `SystemConfig`; it is not duplicated as
top-level job fields.

## Job Structure

```yaml
name: vdp_cam
save: true

engine:
  cam: {}

plugins:
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

| Field | Description |
| --- | --- |
| `name` | Unique logical job name. |
| `engine` | Exactly one engine configuration. |
| `plugins` | Namespaced plugin configurations required by the engine. |
| `params` | Optional engine-specific values. |
| `scan` | Optional explicit `ScanSpec`. |
| `input` | Optional structured upstream input. |
| `save` | `true`, `false`, or an output base name. |
| `system` | Optional override of the normal `SystemConfig` schema. |

Some resource packages also accept plugin namespaces at the job top level for
compatibility with existing files. New resource packages should prefer the
explicit `plugins` mapping.

## Parameter Scans

Scans are explicit. A list inside a plugin configuration is always a literal
plugin value; it is never interpreted as a scan. The removed list-as-scan
syntax raises a migration error for known scalar model parameters.

Each axis has a display name, a target plugin path, and exactly one value
generator:

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

Supported generators are:

| Generator | Meaning |
| --- | --- |
| `values: [...]` | Explicit values. |
| `linspace: {start, stop, num, endpoint}` | Linear spacing; `endpoint` defaults to `true`. |
| `logspace: {start, stop, num, endpoint, base}` | Exponents of `base`; `base` defaults to `10`. |

`combine: cartesian` creates one result dimension per axis in YAML declaration
order. The example above has shape `(31, 101)`. `combine: zipped` requires equal
axis lengths and creates one `point` dimension.

A scan does not create scheduler sub-jobs. The engine receives one runtime
`ParameterGrid` and chooses its own pointwise, tiled, fused, or GPU execution
strategy. The session manifest and output tree still contain one entry and one
directory for the logical job.

## Upstream Input

Inputs use a structured form:

```yaml
input:
  from: vdp_2mode_cayley_sim
  mode: dataset
```

`mode: dataset` passes the complete result to the downstream engine once.
`mode: map` lazily invokes the downstream engine on selected point or group
views while retaining one logical downstream job:

```yaml
input:
  from: source_scan
  mode: map
  select: {omega_a: [0.01, 0.1]}
  group_by: [gamma_b]
```

`select` filters named axis values. `group_by` collects views along the named
axes into `AggregateResult` inputs. Mapping never creates point directories.
String-valued `input` and the old `aggregate_input` field are rejected with a
migration error.

## System Configuration

The packaged defaults live in `qphase.core/system.yaml`. User-wide overrides
live in `~/.qphase/config.yaml`; `QPHASE_SYSTEM_CONFIG` or an explicit loader
path can supply another file. The default policy is:

```yaml
auto_save_results: true

reporting:
  progress:
    refresh_interval: 0.5
    non_tty_milestone_percent: 10.0
    eta_warmup_seconds: 2.0
    eta_min_samples: 3
    eta_smoothing: 0.25
  logging:
    session_file: true
    filename: qphase.log
    file_level: DEBUG
    console_level: WARNING
    format: text
    capture_warnings: true

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

`storage_layout` may be `auto`, `single`, `sharded`, or `per_point`. In `auto`
mode, datasets larger than 512 MiB are sharded by default. Resource values are
hints collected by core and forwarded through `ExecutionContext`; the scheduler
does not use them for multi-job resource allocation.

Checkpointing covers completed scan chunks only. It does not checkpoint the
internal time step of an SDE trajectory. Resume validates the configuration,
plugins, backend, and dtype before accepting compatible checkpoint data.
