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
*   `t0` (`float`): Observation start; `[0, t0)` is integrated as warm-up and not retained.
*   `t1` (`float`): Integration and observation end time.
*   `n_traj` (`int`): Number of trajectories.
*   `seed` (`int | None`): Random seed.
*   `ic` (`Any | None`): Initial condition.
*   `save_stride` (`int`): Save every N-th step.
*   `keep_traj` (`bool | None`): Keep or drop raw trajectories after analysis.

**Methods:**

#### `run(...) -> SDEDataBundle`

Executes the configured SDE job and returns its result bundle. The engine requires `backend`, `model`, and `integrator` plugins and accepts optional `analyser` plugins.

### `class qphase_sde.result.SDEDataBundle`

Catalog of typed data products plus job provenance returned by the engine.

*   `products`: named datasets — the `trajectories` time-series product plus one product per analyzer payload (for example a `spectral` product for `psd`).
*   `provenance`: the job-level SDE provenance record.
*   `axes` / `shape`: named scan-axis coordinates and grid shape (empty for single-point jobs).
*   `point_view(index)`: lazily backed view of one scan point; `metadata["params"]` reports that point's swept values.
*   `metadata`: job metadata (model `params`, scan point info) plus the JSON provenance record.

The scheduler persists the bundle as an Artifact v4 directory — `artifact_manifest.json` plus `npz/3` payload chunks — through `qphase.data`. `qphase.data.load_bundle(job_dir)` restores the `SDEDataBundle` with lazily backed products (no `allow_pickle`); core's `load_result` uses the same manifest path on resume.

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
  scan:
    axes:
      omega_a:
        target: model.kerr_2mode.omega_a
        values: [0.9, 1.1]
  engine:
    sde: { ... }
  model:
    kerr_2mode:
      omega_a: 0.9
      omega_b: 1.0
      chi: 0.01
      gamma_a: 0.1
      gamma_b: 0.1
      g: 0.1
  analyser:
    psd:
      modes: [0]

- name: fit
  input:
    from: sim
    mode: dataset
  engine:
    sde:
      mode: analyze
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```

The `lorentz_fitter` analyzer reads the logical SDE scan dataset, fits one
Lorentzian per scan value, and writes `fit_results.csv` and `psd_merged.csv` to
the Session Job directory. `band_limited_carrier` is an alternative downstream
analyzer for an adaptive, filtered long-time `G^(1)` carrier when a Lorentzian
profile is not assumed. It refuses unresolved single-carrier points and exports
local platforms separately from scan-continuous tracking. Dataset views and
generic export utilities live in core.

`finite_delay_carrier` is the complementary detector-defined observable. It
integrates the complete reconstructed coherence with exponential detector
weights and does not select one pole. Its high-rate limit is the direct SDE
instantaneous carrier.

`spectral_ridge` is the model-independent alternative when the experimental
observable is the PSD maximum rather than a coherence phase average. It reports
scale stability, curvature, and peak-location intervals instead of imposing a
single-peak line shape.
