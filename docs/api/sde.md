---
layout: default
title: SDE API
parent: API Reference
nav_order: 2
---

# SDE API Reference

This section documents the `qphase_sde` package, which provides the core engine and components for stochastic differential equation simulations.

For a more detailed, topic-oriented reference, see the dedicated [`qphase_sde` section](./qphase_sde/index.md).

## Engine

### `class qphase_sde.engine.Engine`

The main simulation driver. It orchestrates the integration loop, manages data storage, and handles progress reporting.

**Configuration (`EngineConfig`):**

*   `dt` (`float`): Time step.
*   `t0` (`float`): Start time.
*   `t1` (`float`): End time.
*   `n_traj` (`int`): Number of trajectories.
*   `seed` (`int | None`): Random seed.
*   `ic` (`Any | None`): Initial condition.
*   `save_stride` (`int`): Save every N-th step.
*   `keep_traj` (`bool | None`): Keep or drop raw trajectories after analysis.

**Methods:**

#### `run(...) -> SDEResult`

Executes the configured SDE job. The engine requires `backend`, `model`, and `integrator` plugins and accepts optional `analyser` plugins.

### `class qphase_sde.result.SDEResult`

Container returned by the SDE engine and saved as `.npz`.

*   `trajectory`: A `TrajectorySet` or `None` if raw data was dropped after analysis.
*   `analysis`: Analyzer payloads keyed by analyzer name, for example `psd`, `dist`, or `pdist`.
*   `meta`: Metadata, including model `params`, `t0`, `dt`, and drop reason when applicable.

Saved archives contain `t0`, `dt`, `meta`, `analysis`, and optional `data`. When present, `data` has shape `(n_traj, n_time, n_modes)`.

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

### PSD Analyzer

`qphase_sde.analyser.PsdAnalyzer` consumes a `TrajectorySet` and writes a PSD payload:

*   `axis`: frequency axis.
*   `psd`: PSD matrix with shape `(n_frequency, n_modes)`.
*   `modes`: analyzed mode indices.
*   `peaks`: optional peak finder output from the PSD analyzer.

PSD analyzer peak detection is local to one job. Cross-job Lorentzian fitting is handled by the `analyser.lorentz_fitter` plugin when the SDE engine runs in `mode: analyze`.

## Postprocessing

Cross-job postprocessing is implemented as a scheduler workflow:

```yaml
- name: sim
  save: true
  engine:
    sde: { ... }
  model:
    kerr_3pa:
      epsilon: [0.025, 0.05]
  analyser:
    psd:
      modes: [0]

- name: fit
  input: sim
  aggregate_input:
    on: epsilon
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: epsilon
      mode: 0
```

The `lorentz_fitter` analyzer reads aggregated `analysis["psd"]` payloads, fits one Lorentzian per scan value, and writes `fit_results.csv` and `psd_merged.csv` to the job's run directory. Generic aggregation/export utilities live in `qphase.core.aggregation`.
