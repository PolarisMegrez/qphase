---
layout: default
title: Models
parent: qphase_sde
grand_parent: API Reference
nav_order: 3
---

# SDE Models

A model plugin describes the SDE

```text
dy = drift(t, y) dt + diffusion(t, y) dW
```

Models are registered under `qphase_sde.models` and selected with `model.<name>` in job configs.

## `SDEModel` Protocol

```python
class SDEModel(Protocol):
    dim: int
    noise_dim: int

    def drift(self, t: float, y: np.ndarray) -> np.ndarray:
        ...

    def diffusion(self, t: float, y: np.ndarray) -> np.ndarray:
        ...
```

*   `dim` — state-space dimension.
*   `noise_dim` — number of independent Wiener increments per step.
*   `drift` returns a vector of shape `(dim,)`.
*   `diffusion` returns a matrix of shape `(dim, noise_dim)`.

The engine evaluates `drift` and `diffusion` each step and passes them to the selected integrator.

## Built-in Models

### `vdp_2mode`

A Van der Pol / Kerr-style two-mode model used for narrow-peak benchmarks.

Key parameters:

*   `omega_a`, `omega_b` — mode frequencies.
*   `gamma_a`, `gamma_b` — damping rates.
*   `Gamma` — nonlinear gain parameter.
*   `g` — coupling strength.
*   `D` — noise strength.

Typical usage:

```yaml
model:
  vdp_2mode:
    omega_a: 0.00251189
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.00001
    g: 0.5
    D: 1.0
```

### `kerr_3pa` and `kerr_3mode`

Kerr-nonlinear three-photon absorption and three-mode models. See [Models source](https://github.com/your-org/qphase/tree/main/models) or the package reference for parameter lists.

## Adding a New Model

Sub-class `SDEModel` and register an entry point:

```toml
[project.entry-points."qphase.models"]
my_model = "my_pkg.models:MyModel"
```

The model class is instantiated from the `model.<name>` block in the job config.
