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
| `t0` | `float` | Observation start. The engine integrates from physical time zero and discards the warm-up interval `[0, t0)`. |
| `t1` | `float` | Integration and observation end time. |
| `n_traj` | `int` | Number of trajectories in the ensemble. |
| `seed` | `int \| None` | Random seed for reproducibility. |
| `ic` | `Any \| None` | Initial condition. |
| `save_stride` | `int` | Store every `N`-th integrated step. See below. |
| `keep_traj` | `bool \| None` | Whether to keep raw trajectory data after analysis. |
| `record_modes` | `list[int] \| None` | Physical modes to retain; `None` stores all modes. |

## Warm-up and observation

The high-level engine always initializes the model at physical time zero. It
integrates without retaining samples until `t0`, then stores samples from `t0`
through `t1`. Both boundaries must be integer multiples of a fixed `dt`.
Time-dependent models therefore continue to receive their physical integration
time during warm-up.

The warm-up is part of computational work but not stored trajectory memory. A
nonzero `t0` removes initial-condition relaxation from stationary PSD and
time-domain statistics; it does not remove stationary soft-mode fluctuations.

## `save_stride` and memory control

`save_stride` lets the integrator take the small `dt` required for stability
while only storing (and later FFTing) every `N`-th sample. The stored trajectory
has effective sample interval `dt * save_stride`, which narrows the PSD Nyquist
frequency but leaves the true frequency resolution unchanged:

```text
df = 1 / (t1 - t0)                  # frequency resolution (unchanged)
f_Nyquist = pi / (dt * save_stride) # Nyquist frequency (reduced)
```

Rough memory for the stored trajectory:

```text
memory ~ n_traj * ((t1 - t0) / (dt * save_stride)) * n_modes * dtype_bytes
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
input data without performing a new simulation. This is used for downstream
post-processing, for example fitting Lorentzians to a logical scan dataset:

```yaml
- name: fit
  input:
    from: sim
    mode: dataset
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: epsilon
      mode: 0
```

## `SDEDataBundle`

Every `engine.run()` exit point returns an `SDEDataBundle`: a catalog of named,
typed data products plus job provenance. The scheduler persists it through
`qphase.data` as an Artifact v4 directory inside the job folder:

*   `artifact_manifest.json` — the validated manifest (`qphase.artifact/4`):
    full product schemas, the `sde.bundle/1` bundle descriptor (scan grid and
    product roles), provenance, and per-variable payload references.
*   payload files — NumPy `.npz` chunks written by the `npz/3` storage adapter
    in native dtypes (never pickled objects). `storage_layout: single` writes
    one `.npz` per product; `sharded` splits large variables into byte-bounded
    chunk files.

Restore with `qphase.data.load_bundle(job_dir)` (core's `load_result` uses the
same path on resume): the manifest is fully validated, products reopen as
lazily backed datasets, and the registered `sde/1` bundle adapter rebuilds the
`SDEDataBundle` — including scan `axes`/`shape` and per-point
`point_view(index)` views.
