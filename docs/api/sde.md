---
layout: default
title: SDE API
parent: API Reference
nav_order: 2
---

# SDE API Reference

This section documents the `qphase_sde` package, which provides the core engine and components for stochastic differential equation simulations.

## Engine

### `class qphase_sde.engine.Engine`

The main simulation driver. It orchestrates the integration loop, manages data storage, and handles progress reporting.

**Configuration (`EngineConfig`):**

*   `dt` (`float`): Time step.
*   `t_max` (`float`): Simulation duration.
*   `n_traj` (`int`): Number of trajectories.
*   `integrator` (`dict`): Integrator settings.
*   `backend` (`str`): Backend name.

**Methods:**

#### `run(model: SDEModel, ...) -> SDEResult`

Executes the simulation for the given model.

---

## Integrators

### `protocol qphase_sde.integrator.Integrator`

The interface that all numerical solvers must implement.

**Methods:**

*   `step(y, t, dt, model, noise, backend) -> dy`: Performs a single fixed time step.
*   `step_adaptive(y, t, dt, tol, model, noise, backend, rng) -> (y_next, t_next, dt_next, error)`: (Optional) Performs an adaptive time step.

### `class qphase_sde.integrator.GenericSRK`

A generic Stochastic Runge-Kutta solver supporting multiple methods and adaptive stepping.

**Parameters:**

*   `method` (`str`): The integration scheme to use (`"euler"`, `"heun"`).
*   `tol` (`float`, optional): Error tolerance for adaptive stepping.

---

## Models

The `qphase_sde` package supports a hierarchical modeling approach.

### Level 1: Master Equation

#### `class qphase_sde.model.MasterEquation`

Represents the system dynamics in Hilbert space.

**Attributes:**
*   `hamiltonian`: The Hamiltonian operator.
*   `lindblad_ops`: List of Lindblad collapse operators.

### Level 2: Phase Space (FPE)

#### `class qphase_sde.model.PhaseSpaceModel`

Represents the system dynamics in phase space via Kramers-Moyal coefficients.

**Attributes:**
*   `terms` (`dict[int, Any]`): Dictionary mapping order $n$ to coefficient $D_n(\alpha)$.
    *   $n=1$: Drift vector.
    *   $n=2$: Diffusion tensor.

### Level 3: Stochastic (SDE)

#### `protocol qphase_sde.model.SDEModel`

The interface for defining physical systems consumed by the engine.

**Attributes:**

*   `n_modes` (`int`): Dimension of the state vector.
*   `noise_dim` (`int`): Dimension of the noise vector.
*   `noise_basis` (`str`): `"real"` or `"complex"`.

**Methods:**

*   `drift(y, t, params) -> Any`: Computes the drift vector $\mathbf{a}(\mathbf{y}, t)$.
*   `diffusion(y, t, params) -> Any`: Computes the diffusion matrix $\mathbf{b}(\mathbf{y}, t)$.

#### `class qphase_sde.model.DiffusiveSDEModel`

Concrete implementation for Langevin-type SDEs (Continuous, Gaussian noise).

#### `class qphase_sde.model.JumpSDEModel`

Concrete implementation for Jump-Diffusion SDEs.

### Converters

#### `qphase_sde.model.fpe_to_sde(fpe: PhaseSpaceModel) -> DiffusiveSDEModel`

Converts a 2nd-order PhaseSpaceModel to a DiffusiveSDEModel.
*   Drift $A = D_1$
*   Diffusion $B = \sqrt{D_2}$

---

## Noise Specification

Defines the properties of the noise driving the system.

**Attributes:**

*   `kind` (`str`): `"independent"` or `"correlated"`.
*   `dim` (`int`): Number of noise channels.
*   `covariance` (`Any`, optional): Covariance matrix for correlated noise.

---

## Analyzers

### `protocol qphase_sde.analyser.AnalyzerProtocol`

The interface for analysis plugins.

**Methods:**

*   `analyze(data: Any, backend: BackendBase) -> ResultProtocol`: Performs analysis on the simulation data.
