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

The framework supports several integration schemes via the `GenericSRK` (Stochastic Runge-Kutta) class.

### Available Methods

*   **`euler` (Euler-Maruyama)**:
    *   **Order**: Strong 0.5, Weak 1.0.
    *   **Use case**: Simple additive noise or when speed is critical and high accuracy is not required.
    *   **Interpretation**: Ito.

*   **`heun` (Stochastic Heun)**:
    *   **Order**: Strong 1.0 (approx), Weak 2.0.
    *   **Use case**: Multiplicative noise where Stratonovich interpretation is desired.
    *   **Interpretation**: Stratonovich.

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
