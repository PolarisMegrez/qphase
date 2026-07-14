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

## Batched Parameter Scans

When a job contains a parameter scan (for example `model.omega_a: [0.001, 0.002, 0.003]`), the scheduler normally expands it into multiple independent simulation jobs. If those jobs share the **same engine, integrator, backend, and time grid** and only differ in model parameters, `qphase_sde` can fuse them into a single batched simulation.

In a batched run:

* The scan values are broadcast into a single `(n_scan * n_traj,)` ensemble.
* One backend call advances every trajectory for every scan point at once.
* Results are automatically split back into the original per-scan jobs, each with its own run directory and manifest entry.

This is especially effective on GPUs: a small CPU launch overhead per time step is amortized over many trajectories, and a single CuPy kernel launch can update the whole ensemble.

Batched execution is **automatic** and requires no extra configuration. The only requirement on the model is that its parameters accept either a scalar or a one-dimensional array so the planner can broadcast the scan values across the ensemble.

## Kernelized Terms (CuPy)

Some built-in models provide an optional **fused drift+diffusion kernel** for the CuPy backend. When a kernel is available and the job uses `backend: cupy`, the integrator prefers the kernel path over calling the Python `drift` and `diffusion` methods separately. This removes Python-layer dispatch overhead and fuses the two term evaluations into a single CUDA launch.

Current models with CuPy kernels:

* `model.vdp_2mode`
* `model.kerr_3mode`

Kernelization is **automatic**; there is no explicit switch in the job file. If the model advertises support for the active backend, the integrator uses it. Otherwise it falls back to the standard Python implementation, so switching `backend` between `numpy` and `cupy` always produces valid results.

### Example: CuPy Kernel Workflow

```yaml
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
    omega_a: [0.001, 0.00251189, 0.01]
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.00001
    g: 0.5
    D: 1.0
```

Because `omega_a` has three values and the backend is CuPy, the scheduler will batch the three scan points and the `vdp_2mode` kernel will evaluate all `3 * 20` trajectories in fused CUDA kernels.
