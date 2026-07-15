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

The generic stochastic Runge-Kutta solver (`integrator.srk`) supports multiple
schemes and adaptive stepping.

| Method | Order | Interpretation | Evaluations per step | Typical use |
| :-- | :-- | :-- | :-- | :-- |
| `euler` | Strong 0.5, Weak 1.0 | Ito | 1 | Additive noise, speed over accuracy. |
| `heun` | Strong 1.0 (approx), Weak 2.0 | Stratonovich | 2 | Multiplicative noise. |

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

`ChunkIntegrator` is an optional capability. The SDE engine uses it only for
fixed-step jobs when the selected model/backend combination supports the same
scheme; all existing integrators continue to use the ordinary `step()` path.

For long `complex64` trajectories, roundoff accumulation can leave a small
frequency residual even after the Euler bias is removed. In the VDP validation
at `omega_a=0.001`, this residual was about `5e-6`; the same fused kernel with
`complex128` agreed with the Cayley dispersion relation to machine precision.
Choose `float64` only when that residual matters more than GPU throughput and
trajectory memory.
