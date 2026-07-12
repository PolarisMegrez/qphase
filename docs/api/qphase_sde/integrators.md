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

| Method | Order | Interpretation | Typical use |
| :-- | :-- | :-- | :-- |
| `euler` | Strong 0.5, Weak 1.0 | Ito | Additive noise, speed over accuracy. |
| `heun` | Strong 1.0 (approx), Weak 2.0 | Stratonovich | Multiplicative noise. |

Enable adaptive stepping by providing a tolerance:

```yaml
integrator:
  srk:
    method: heun
    tol: 1e-5
```

The engine interpolates adaptive output back to the regular grid defined by
`dt` and `save_stride` before storage.
