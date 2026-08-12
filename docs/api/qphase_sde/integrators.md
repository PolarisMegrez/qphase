---
layout: default
title: Integrators
parent: qphase_sde
grand_parent: API Reference
nav_order: 2
---

# SDE Integrators

## `Integrator` protocol

All numerical solvers consumed by the SDE engine implement the
`qphase_sde.integrator.Integrator` protocol:

*   `step(y, t, dt, model, noise, backend) -> dy` — single fixed time step.
*   `step_adaptive(y, t, dt, tol, model, noise, backend, rng) -> (y_next, t_next, dt_next, error)` —
    optional adaptive step.

## `GenericSRK`

The stochastic Runge-Kutta plugin (`integrator.srk`) provides two built-in
schemes and adaptive stepping. Its schema accepts only `euler` and `heun`;
arbitrary Butcher coefficient tables are intentionally not supported.

| Method | Order | Interpretation | Evaluations per step | Typical use |
| :-- | :-- | :-- | :-- | :-- |
| `euler` | Strong 0.5, Weak 1.0 under standard assumptions | Ito | 1 | General Ito SDE baseline. |
| `heun` | Strong 1.0 under suitable regularity and noise assumptions | Stratonovich | 2 | Stratonovich SDEs; additive-noise Ito SDEs. |

The `heun` implementation is a Stratonovich predictor-corrector. For an Ito
SDE with state-dependent diffusion, first apply the Ito-to-Stratonovich drift
correction; selecting `heun` does not perform that conversion automatically.
For additive noise the correction vanishes, so the Ito and Stratonovich forms
coincide.

The standalone `integrator.euler_maruyama` plugin provides the same `euler`
scheme with identical numerical behavior; it is often used when you want a
dedicated integrator namespace rather than the generic SRK dispatcher.

Enable adaptive stepping by providing a tolerance:

```yaml
integrator:
  srk:
    method: heun
    tol: 1e-5
```

The engine interpolates adaptive output back to the regular grid defined by
`dt` and `save_stride` before storage.

## `CayleyMaruyama`

`integrator.cayley_maruyama` is a fixed-step Ito integrator for models whose
drift can be written as `A(y,t) @ y`. It uses

```text
(I - dt*A_n/2) y_(n+1) = (I + dt*A_n/2) y_n + B_n dW_n
```

with both `A_n` and `B_n` evaluated at the left endpoint. For a neutral
oscillatory eigenvalue, the Cayley transform preserves unit modulus and avoids
the artificial radial gain produced by explicit Euler integration.

```yaml
integrator:
  cayley_maruyama:
    fused: auto       # auto, required, or off
    chunk_steps: 128  # 1 disables multi-step fusion
    max_modes: 16     # configurable up to 64
```

The generic path uses backend batched linear solves and supports small systems
with arbitrary mode counts. A model may provide a specialized fused step or
chunk kernel. `fused: required` is recommended for production GPU jobs so a
missing accelerator cannot silently fall back to the generic path.
The workspace VDP2, Kerr2, cross-Kerr2, and Kerr3 models provide CuPy fused
`step` and `step_chunk` kernels for both `complex64` and `complex128` states.

`ChunkIntegrator` is an optional capability. The SDE engine uses it only for
fixed-step jobs when the selected model/backend combination supports the same
scheme; all existing integrators continue to use the ordinary `step()` path.

For long `complex64` trajectories, roundoff accumulation can leave a small
frequency residual even after the Euler bias is removed. In the VDP validation
at `omega_a=0.001`, this residual was about `5e-6`; the same fused kernel with
`complex128` agreed with the Cayley dispersion relation to machine precision.
Choose `float64` only when that residual matters more than GPU throughput and
trajectory memory.
