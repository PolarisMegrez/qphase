---
layout: default
title: Engine
parent: qphase_sde
grand_parent: API Reference
nav_order: 1
---

# SDE Engine

The SDE engine (`qphase_sde.engine.Engine`) orchestrates the integration loop,
data storage, and optional per-step analysis.

## `EngineConfig`

Top-level keys live under `engine.sde` in a job file:

| Key | Type | Description |
| :-- | :-- | :-- |
| `dt` | `float` | Integration time step. Must be small enough for stability. |
| `t0` | `float` | Start time. |
| `t1` | `float` | End time. |
| `n_traj` | `int` | Number of trajectories in the ensemble. |
| `seed` | `int \| None` | Random seed for reproducibility. |
| `ic` | `Any \| None` | Initial condition. |
| `save_stride` | `int` | Store every `N`-th integrated step. See below. |
| `keep_traj` | `bool \| None` | Whether to keep raw trajectory data after analysis. |
| `record_modes` | `list[int] \| None` | Physical modes to retain; `None` stores all modes. |

## `save_stride` and memory control

`save_stride` lets the integrator take the small `dt` required for stability
while only storing (and later FFTing) every `N`-th sample. The stored trajectory
has effective sample interval `dt * save_stride`, which narrows the PSD Nyquist
frequency but leaves the true frequency resolution unchanged:

```text
df = 1 / t1                         # frequency resolution (unchanged)
f_Nyquist = pi / (dt * save_stride) # Nyquist frequency (reduced)
```

Rough memory for the stored trajectory:

```text
memory ~ n_traj * (t1 / (dt * save_stride)) * n_modes * dtype_bytes
```

`record_modes` reduces the final factor without changing the simulated state.
The trajectory stores `meta.mode_indices`, and SDE analyzers continue to accept
physical mode numbers:

```yaml
engine:
  sde:
    record_modes: [0]
analyser:
  psd:
    modes: [0]
```

Stored trajectory arrays retain the state dtype. A `complex64` CuPy simulation
therefore produces `complex64` history instead of being promoted to
`complex128`.

For a narrow low-frequency peak, choose `save_stride` so that
`f_Nyquist` stays well above the highest frequency of interest. For example,
with `dt = 0.1` and a peak near `0.1` rad/s, `save_stride = 50` gives
`f_Nyquist ~ 0.63` rad/s, which is plenty.

```yaml
engine:
  sde:
    t0: 0.0
    t1: 10000.0
    dt: 0.1
    save_stride: 50
    n_traj: 100
```

## `mode: analyze`

Setting `engine.sde.mode: analyze` runs the configured analyzers on upstream
input data without performing a new simulation. This is used for cross-job
post-processing, for example fitting Lorentzians to aggregated PSD data:

```yaml
- name: fit
  input: sim
  aggregate_input:
    on: params.epsilon
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: epsilon
      mode: 0
```

## `SDEResult`

The engine returns and saves an `SDEResult` as a NumPy `.npz` archive.

*   `trajectory` — a `TrajectorySet` or `None` if raw data was dropped after
    analysis.
*   `analysis` — analyzer payloads keyed by analyzer name (`psd`, `dist`,
    `pdist`).
*   `meta` — metadata including model `params`, `t0`, `dt`, and the drop reason
    when applicable.

Saved archives contain `t0`, `dt`, `meta`, `analysis`, and, when retention is
enabled, raw `data` with shape `(n_traj, n_time, n_modes)`.
