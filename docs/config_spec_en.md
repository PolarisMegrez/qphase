# QPhaseSDE v0.1.3 Configuration Specification (Triad YAML)

> Note: This document describes the legacy triad format. For the new jobs-based configuration with sweep DSL, see `config_spec_en_v2.md`.

This document describes the expected YAML configuration for the CLI `qps run sde` in v0.1.3. The configuration is organized into three sections ("triad"): `model`, `profile`, and `run`.

- Audience: practitioners familiar with SDE modeling and reproducible simulations.
- Goal: predictable, validated inputs with clear defaults and required fields.

## 1. model (A-class, no defaults)

Defines the physical model and its initialization.

Required fields:
- module: string — Python module path or a .py file path to load (must expose `build_sde(params)`)
- function: string — function in that module to call (e.g., `build_sde`)
- params: object — free-form parameter mapping passed into `build_sde`
- ic: list — initial conditions for the complex-valued mode amplitudes
  - Accepts either a flat list (single IC vector) or a nested list (multiple IC vectors)
  - Each element must be a string parsable by Python’s `complex()`, e.g., `"7.0+0.0j"`
  - Shapes:
    - Flat: ["7.0+0.0j", "0.0-7.0j"] → one vector of length `n_modes`
    - Nested: [[...], [...], ...] → either length 1 (broadcast) or length `n_traj`
- noise: object
  - kind: "independent" | "correlated"
  - covariance: number[][] when `kind=correlated` (real-valued, positive semi-definite)

Example:
```yaml
model:
  module: models.vdp_two_mode
  function: build_sde
  params:
    omega_a: 0.005
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.01
    g: 0.5
    D: 1.0
  ic: ["7.0+0.0j", "0.0-7.0j"]
  noise:
    kind: independent
```

Notes:
- The CLI validates that each IC vector length matches `model.n_modes`.
- A single IC vector broadcasts to all trajectories; otherwise provide `n_traj` vectors.

## 2. profile (B-class, has defaults)

Controls execution details that don’t affect the physics.

Fields:
- backend: "numpy" (default) | "numba" (reserved)
- solver: "euler" (default) | "milstein" (placeholder in v0.1.3, falls back to euler)
- save:
  - root: string (default: `runs`) — output root directory
  - save_every: int (optional) — decimation factor for saved time series
  - save_timeseries: bool (required) — whether to persist NPZ time series per IC
  - save_psd_complex: bool (required) — whether to compute/save complex-PSD NPZ per IC
  - save_psd_modular: bool (required) — whether to compute/save modular-PSD NPZ per IC
- visualizer (optional): matplotlib kwargs and PSD conventions
  - phase_portrait:
    - Re_Im: mapping for `Re-Im` portraits (preferred key; also accepts `re_im`)
    - abs_abs: mapping for `|.|-|.|` portraits
  - psd:
    - convention: `symmetric` (alias `unitary`) or `pragmatic`
    - x_scale: `linear` or `log`
    - y_scale: `linear` or `log`

Example:
```yaml
profile:
  backend: numpy
  solver: euler
  save:
    root: runs
    save_every: 20
    save_timeseries: true
    save_psd_complex: true
    save_psd_modular: false
  visualizer:
    phase_portrait:
      re_im:
        linewidth: 0.8
        alpha: 0.6
      abs_abs:
        linewidth: 0.8
        alpha: 0.6
    psd:
      convention: symmetric
      x_scale: linear
      y_scale: log
```

## 3. run (C-class, required where specified)

Defines the numeric integration and the requested visualizers.

Fields:
- time:
  - dt: float (required)
  - steps: int (required)
  - t0: float (default 0.0)
- trajectories:
  - n_traj: int (required)
  - master_seed: int (optional) — master seed used to derive per-trajectory seeds, or
  - seed_file: string (optional) — path to a file with N seeds (one per trajectory)
  - rng_stream: "per_trajectory" | "batched" (optional; default "per_trajectory") — controls RNG strategy:
    - per_trajectory: independent RNG stream per trajectory (stable across n_traj changes)
    - batched: single RNG stream for vectorized sampling (faster; sequence changes if n_traj/order changes)
- visualizer (optional):
  - phase_portrait: list of per-figure specifications
    - kind: "Re_Im" (preferred) or "re_im", or "abs_abs" (required)
    - modes: list[int] — 1 index for "re_im"; 2 indices for "abs_abs" (required)
    - t_range: [t_start, t_end] optional — limits the signal segment used for plotting
  - psd: list of PSD figure specifications
    - kind: `complex` | `modular` (required)
    - modes: list[int] — one or more modes per figure (required)
    - xlim: [xmin, xmax] optional
    - t_range: [t_start, t_end] optional

Example:
```yaml
run:
  time:
    dt: 0.001
    steps: 200
  trajectories:
    n_traj: 4
    master_seed: 42
  visualizer:
    phase_portrait:
      - kind: Re_Im
        modes: [0]
      - kind: abs_abs
        modes: [0, 1]
        t_range: [0.05, 0.15]
    psd:
      - kind: complex
        modes: [0, 1]
        xlim: [-0.5, 0.5]
        t_range: [20.0, 100.0]
```

## Validation summary

- Schema ensures required fields are present.
- `model.ic` is normalized to a nested list and validated for complex parsing.
- `run.viz.phase` entries validate kind/modes shape and t_range consistency.
- At runtime, the CLI checks that IC vector length equals `model.n_modes`, and IC list length is 1 or `n_traj`.

## Runtime behavior

- Run outputs:
  - `time_series/timeseries.npz` (data, t0, dt)
  - `config_snapshot/config.json` and `config_snapshot/triad.yaml`
  - figures per IC under `figures/icXX/*.png` when figures are requested
  - PSD NPZ per IC saved under `psd/icXX/psd_{kind}_{convention}.npz` when enabled
- Re-plotting:
  - `qps analyze phase --from-run <run_dir>` uses saved viz specs and styles.
  - `qps analyze psd --from-run <run_dir>` uses saved PSD specs and styles.
  - Or override with `--specs-json` to render new figures without recompute.

## Versioning notes and storage guard

- v0.1.3 supports NumPy + Euler–Maruyama, placeholder Milstein, and multi-IC semantics:
  - If `model.ic` contains multiple vectors, each IC is simulated independently with the same settings.
  - Time series are saved per-IC as `time_series/timeseries_icXX.npz`.
  - Figures are saved per-IC under `figures/icXX/`.
  - PSD files are saved per-IC when toggles are enabled.
- To prevent runaway disk usage, the CLI estimates time-series storage and aborts if it exceeds a default 1 GiB threshold. Override with `--max-storage-gb`.
- Future versions will expand solvers, backends, and visualizer types.
