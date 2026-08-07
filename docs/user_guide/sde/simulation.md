---
layout: default
title: SDE Simulation
parent: User Guide
nav_order: 5
---

# SDE Simulation Guide

The `qphase_sde` package provides a robust engine for solving Stochastic Differential Equations (SDEs) in the phase space. It is designed to be modular, allowing you to easily switch between different integration schemes and noise models.

## Overview

The SDE engine solves equations of the form:

$$ d\mathbf{y} = \mathbf{a}(\mathbf{y}, t) dt + \mathbf{b}(\mathbf{y}, t) d\mathbf{W} $$

where:
*   $\mathbf{y}$ is the state vector (e.g., phase-space coordinates).
*   $\mathbf{a}(\mathbf{y}, t)$ is the **drift** vector.
*   $\mathbf{b}(\mathbf{y}, t)$ is the **diffusion** matrix.
*   $d\mathbf{W}$ is the Wiener process increment (noise).

## Configuration

To use the SDE engine, you need to specify it in your job configuration file (e.g., `job.yaml`).

```yaml
engine:
  sde:
    dt: 1e-3              # Time step size
    t_max: 10.0           # Total simulation time
    n_traj: 1000          # Number of trajectories
    integrator:           # Integrator configuration
      name: "srk"         # Use the Generic SRK solver
      method: "heun"      # Specific method (heun, euler)
      tol: 1e-4           # Tolerance for adaptive stepping
    backend: "numpy"      # Backend (numpy, torch, cupy)
```

### Key Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `dt` | `float` | The base time step size. For fixed-step solvers, this is the step used. For adaptive solvers, this is the initial step guess. |
| `t_max` | `float` | The end time of the simulation (starts at t=0). |
| `n_traj` | `int` | The number of parallel trajectories to simulate. |
| `integrator` | `dict` | Configuration for the numerical solver. |
| `backend` | `str` | The computational backend to use. |

### Trajectory escape guard

Models with only local attractors can leave their attraction basin during a
long stochastic run while remaining finite in floating-point arithmetic. Set
`max_state_norm` on `engine.sde` to reject such trajectories before PSD or
other stationary analysis is performed. `state_check_interval_steps` controls
the check cadence; accelerator runs synchronize only when a configured check
is due.

```yaml
engine:
  sde:
    max_state_norm: 30.0
    state_check_interval_steps: 1024
```

The bound is model- and parameter-dependent. It should lie above the stationary
fluctuation scale and, when known, near the outer edge of the attraction basin.
Exceeding it raises `TrajectoryDivergenceError`; it does not clip, reset, or
silently discard trajectories.

## Integrators

The framework supports several integration schemes. Choosing the right one is a trade-off between accuracy, stability, and computational cost.

### Available Methods

| Integrator | Interpretation | Strong order | Drift/diffusion evaluations per step | Typical use case |
| :-- | :-- | :-- | :-- | :-- |
| `euler_maruyama` | Itô | 0.5 | 1 | Large ensembles, additive noise, or when speed dominates accuracy. |
| `heun` (SRK) | Stratonovich | ~1.0 | 2 | Multiplicative noise, moderate accuracy, parameter scans. |
| `milstein` | Itô | 1.0 | 1 + Jacobian | Diagonal/commutative multiplicative noise where strong-order-1.0 accuracy is needed. |

#### Euler–Maruyama

*   **Update rule**: `dy = a(y)·dt + L(y)·dW`.
*   **Pros**: One evaluation of drift and diffusion per step; fastest per-step cost; pairs naturally with fused drift+diffusion kernels on CuPy.
*   **Cons**: Low strong order (0.5). Errors accumulate linearly with `dt`, so it needs small time steps for stiff or multiplicative-noise systems. Can become unstable when the diffusion is state-dependent and `dt` is too large.
*   **When to use**: Additive or weakly multiplicative noise; very long trajectories where you primarily need statistical moments; GPU batch jobs where minimizing kernel launches is important.

#### Stochastic Heun (SRK method `heun`)

*   **Update rule**: predictor–corrector using drift and diffusion at `y` and at a predicted `y_bar`.
*   **Pros**: Strong order ~1.0 under Stratonovich interpretation; more stable than Euler–Maruyama for state-dependent diffusion; no Jacobian required.
*   **Cons**: Two drift/diffusion evaluations per step, so roughly twice the compute of Euler–Maruyama. The predictor stage can also amplify transients if `dt` is large.
*   **When to use**: Multiplicative noise interpreted in the Stratonovich sense; parameter scans where you want better path-wise accuracy than EM without implementing a Jacobian.

#### Milstein

*   **Update rule**: `dy = a·dt + L·dW + 0.5·G·(dW² − dt)`, where `G` is a correction built from the diffusion Jacobian.
*   **Pros**: Strong order 1.0 in the Itô sense; captures leading-order multiplicative-noise corrections without the second evaluation of Heun.
*   **Cons**: Requires `model.diffusion_jacobian`; the Jacobian evaluation can be expensive and is not yet covered by the fused CuPy kernels, so the GPU speed advantage is smaller. Currently falls back to Euler–Maruyama when the model uses a complex noise basis and no compatible Jacobian is provided.
*   **When to use**: Diagonal or commutative multiplicative noise where Itô calculus is required and you need strong-order-1.0 accuracy.

### Stability and Cost Summary

*   **Cost (low → high)**: `euler_maruyama` < `milstein` (with cheap Jacobian) ≈ `heun` < `milstein` (with expensive Jacobian).
*   **Strong accuracy (low → high)**: `euler_maruyama` (0.5) < `heun` (~1.0) ≈ `milstein` (1.0).
*   **Stability for multiplicative noise**: `euler_maruyama` is the most restrictive on `dt`; `heun` and `milstein` tolerate larger steps.
*   **GPU batching**: `euler_maruyama` benefits most from fused kernels because it only needs one fused drift+diffusion evaluation per step. `heun` currently needs two fused evaluations; a fully fused Heun kernel would close this gap.

### Adaptive Stepping

The `srk` integrator supports **adaptive stepping** using Richardson extrapolation (step doubling). This allows the solver to automatically reduce the step size `dt` when the error is high (e.g., during fast dynamics) and increase it when the system is stable.

To enable adaptive stepping, simply provide a `tol` (tolerance) parameter in the integrator config.

```yaml
integrator:
  name: "srk"
  method: "heun"
  tol: 1e-5  # Enables adaptive stepping with target error 1e-5
```

**Note**: Even with adaptive stepping, the engine will interpolate the results to save data at fixed intervals defined by `dt` and `return_stride`. This ensures that your output data is always on a regular time grid, simplifying analysis.

## Defining Models

To simulate a system, you need to define a model that implements the `SDEModel` protocol. This involves specifying the `drift` and `diffusion` functions.

See the [Plugin Development](../../dev_guide/plugin_development.md) guide for details on how to write and register custom models.

## Resource-Aware Parameter Scans

SDE scans use an explicit job-level `ScanSpec`. The scheduler passes one
`ParameterGrid` and one `ExecutionContext` to the SDE engine; scan points are
never expanded into scheduler jobs or per-point run directories.

Before allocating trajectory arrays, the engine validates the time grid and
analyser bandwidth, estimates the state, noise, trajectory, and analyser
workspaces, and builds an execution plan. Host and device budgets are tracked
separately, including host-side analysers used with a CuPy backend. Resource policy is read from the
`ExecutionContext.resources` object. The SDE package does not discover or read
a system configuration file. On CuPy, the plan also queries current device
memory and applies the configured GPU memory fraction.

When analysers are configured and `keep_traj` is false, the engine may split a
large scan into internal tiles and split each point into trajectory batches.
Each batch is integrated, analysed, merged with Chan/Welford statistics, and
released. Periodogram, Welch, and multitaper PSD estimators all support this
trajectory dimension batching; it preserves each estimator's time-domain
resolution and the existing `psd`, `psd_std`, and `psd_sem` result fields.
Stable logical RNG groups make results independent of the selected scan tile
and physical trajectory batch size for a fixed seed.

If the full trajectory is requested, it must fit the available resource budget
and is materialized as one logical result. With CuPy, explicit
`keep_traj: true` runs analysis on device first and then transfers the retained
record to host memory for serialization. If even one trajectory plus analyser
accumulator cannot fit, planning fails before integration with a memory estimate
instead of relying on an out-of-memory error. Increasing
`save_stride` reduces saved trajectory and FFT sizes, but it does not reduce the
number of integration steps.

`trajectory_batching` accepts `auto` (default), `off`, or `required`.
`trajectory_batch_size` is an optional diagnostic/benchmark override; normal
jobs should leave it unset so current host/device resources determine the batch.

The selected tile size, estimated byte counts, resource budget, and random
stream strategy are recorded in result metadata under `execution_plan`.

## Online Observers

Observers are plugins that inspect the live state without mutating it or
consuming RNG. They are separate from analysers: an observer may request
whole-batch control flow while an analyser consumes recorded trajectories.

```yaml
observer:
  first_passage:
    rule: state_norm
    direction: above
    threshold: 100.0
    check_interval_steps: 100
    debounce_checks: 2
    action: record
```

`observer.first_passage` supports `state_norm`, `mode_magnitude`,
`linear_projection`, and canonical-R `matrix_projection` rules. Its actions
are `record`, `stop_batch`, and `fail_job`. A non-hit is right-censored and is
stored with `first_hit_time=NaN` and `first_hit_step=-1`.

On CuPy, observables and debounce masks remain on the active device; only
newly confirmed event indices and values are copied to the host. Observer
cadence can clamp a fused chunk so that checks occur on exact integration
steps, but it does not disable the CuPy backend or the fused kernel. Progress
continues to report actual completed trajectory-steps through the existing
engine reporter. A smaller cadence may reduce throughput and is recorded as
`effective_chunk_steps` in the execution plan.

Observer implementations return a generic `ObserverDecision` and own their
trajectory-batch payload merge and fused-scan payload split. Consequently each
scan point view contains only its own trajectory events and recomputed aggregate
counters. The SDE engine only schedules checks and
applies `stop_batch`/`fail_job`; it does not interpret plugin-specific fields.

## Trajectory Diagnostics

`analyser.trajectory_diagnostics` provides reduced time-domain diagnostics:
complex first-order coherence and model fits, angular-frequency Allan
variance, non-overlapping block statistics and spectra, stationarity details,
and optional canonical-R matrix projection. It is intended for selected,
reduced experiments rather than unrestricted production trajectories.

The current implementation materializes its input on the host. In a CuPy job,
this means an explicit device-to-host transfer after sampling; it does not
change the integration backend. Leave `matrix_projection_keep_coordinates`
disabled unless the full coordinate trajectory is required. The analyser
declares this host/full-record requirement and its workspace to the planner.
Its internal coherence, Allan, block-spectrum, projection, and stationarity
children share one host trajectory and one lazily built canonical-coordinate
array; they are not scheduler jobs. Time-streaming diagnostics remain future
work.

## Kernelized Terms (CuPy)

Some built-in models provide an optional **fused drift+diffusion kernel** for the CuPy backend. When a kernel is available and the job uses `backend: cupy`, the integrator prefers the kernel path over calling the Python `drift` and `diffusion` methods separately. This removes Python-layer dispatch overhead and fuses the two term evaluations into a single CUDA launch.

Current models with CuPy kernels:

* `model.vdp_2mode`
* `model.kerr_2mode`
* `model.crosskerr_2mode`
* `model.kerr_3mode`
* `model.kerr_full_3mode`
* `model.pair_hopping_2mode`

Kernelization is **automatic**; there is no explicit switch in the job file. If the model advertises support for the active backend, the integrator uses it. Otherwise it falls back to the standard Python implementation, so switching `backend` between `numpy` and `cupy` always produces valid results.

### Example: CuPy Kernel Workflow

```yaml
scan:
  axes:
    omega_a:
      target: model.vdp_2mode.omega_a
      values: [0.001, 0.00251189, 0.01]
engine:
  sde:
    t0: 0.0
    t1: 2000.0
    dt: 0.1
    n_traj: 20
backend:
  cupy:
    float_dtype: float32
    device: cuda
model:
  vdp_2mode:
    omega_a: 0.001
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.00001
    g: 0.5
```

Because `omega_a` has three values and the backend is CuPy, the SDE adapter
fuses as many points as the execution plan permits. The `vdp_2mode` kernel
evaluates each internal tile with fused CUDA kernels. The scheduler still sees
one logical job.
