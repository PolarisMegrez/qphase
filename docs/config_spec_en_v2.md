# QPhaseSDE Configuration v2 (jobs-based)

This document describes the new jobs-based YAML configuration for the CLI. It replaces the legacy triad format and enables parameter sweeps expanded into multiple jobs at load time.

- Audience: users running reproducible SDE simulations with sweeps and per-job visualization.
- Scope: profile (execution), run (numerical knobs), jobs (model + params + IC + noise + viz).

## Top-level schema

Fields:
- profile: execution defaults and styles
  - backend: "numpy" (default)
  - solver: "euler" (default)
  - save:
    - root: output root directory (default: runs)
    - save_every: int (optional decimation)
    - save_timeseries: bool
    - save_psd_complex: bool
    - save_psd_modular: bool
  - visualizer (optional): global viz style defaults
- run: numeric integration knobs
  - time:
    - dt: float
    - steps: int
    - t0: float (default 0.0)
  - trajectories:
    - n_traj: int
    - master_seed: int (optional)
    - rng_stream: "per_trajectory" | "batched" (optional)
- jobs: list[Job]

## Job

- name: string identifier (optional but recommended)
- module: python module path to model (e.g. models.vdp_two_mode)
- function: function inside module (e.g. build_sde)
- params: mapping of model parameters; supports sweep DSL (below)
- ic: list of complex strings or list[list] (same semantics as legacy)
- noise:
  - kind: independent | correlated
  - covariance: number[][] when kind=correlated
- visualizer (optional): list specs per figure (phase_portrait, psd)
- combinator: cartesian | zipped (optional; default cartesian via defaults)

At load time, sweeps in `params` are expanded into multiple jobs using the job-level `combinator`. Each materialized job inherits the fields and gets a distinct name (with numeric suffix if `name` is provided).

## Sweep DSL

Parameter values accept three forms. Mixing is allowed across different keys.

- Single value: a scalar, e.g. `g: 0.5`
- Direct array: a list of scalars, e.g. `D: [0.01, 0.02]`
- DSL object: one of
  - `{ lin: [start, stop, num] }` inclusive linear sequence
  - `{ log: [start, stop, num] }` base-10 logspace, inclusive
  - `{ values: [v1, v2, ...] }` explicit list

Combinators:
- cartesian (default): forms the cartesian product across all swept keys.
- zipped: zips element-wise across swept keys; all swept arrays must have equal length.

### Examples

Cartesian:
```yaml
profile:
  backend: numpy
  solver: euler
run:
  time: { dt: 0.001, steps: 120 }
  trajectories: { n_traj: 4, master_seed: 123 }
jobs:
  - name: sweep_cart
    module: models.vdp_two_mode
    function: build_sde
    params:
      g: { lin: [0.4, 0.5, 2] }   # -> [0.4, 0.5]
      D: [0.01, 0.02]
      omega_a: 0.005
      omega_b: 0.0
      gamma_a: 2.0
      gamma_b: 1.0
      Gamma: 0.01
    ic:
      - ["7.0+0.0j", "0.0-7.0j"]
    noise: { kind: independent }
```
This materializes 2 x 2 = 4 jobs.

Zipped:
```yaml
profile:
  backend: numpy
  solver: euler
run:
  time: { dt: 0.001, steps: 120 }
  trajectories: { n_traj: 4, master_seed: 321 }
jobs:
  - name: sweep_zip
    module: models.vdp_two_mode
    function: build_sde
    combinator: zipped
    params:
      g: { values: [0.4, 0.5, 0.6] }
      D: { values: [0.01, 0.02, 0.03] }
      omega_a: 0.005
      omega_b: 0.0
      gamma_a: 2.0
      gamma_b: 1.0
      Gamma: 0.01
    ic:
      - ["7.0+0.0j", "0.0-7.0j"]
    noise: { kind: independent }
```
This materializes 3 jobs, pairing (0.4,0.01), (0.5,0.02), (0.6,0.03).

## Runtime behavior

- The CLI expands sweeps during config loading; `RootConfig.jobs` contains only concrete jobs.
- Each job executes in its own run directory with a full config snapshot and artifacts.
- Visualizer specs are normalized to lists and validated before rendering.

## Migration notes

- Legacy triad configs are auto-migrated to a single-job v2 config at load time.
- `io.config_user` has been removed; core configuration lives under `core.config`.
- Default job combinator is `cartesian` (configurable via `core/defaults.yaml`).
