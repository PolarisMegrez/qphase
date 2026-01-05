---
layout: default
title: SDE Modeling Guide
parent: User Guide
nav_order: 3
---

# SDE Modeling Guide

QPhase SDE adopts a three-level architecture for modeling physical systems. This hierarchy allows for separation of concerns between the physical definition (Quantum/Phase Space) and the numerical implementation (SDE).

## The Three-Level Architecture

### Level 1: Master Equation (ME)
**Class:** `qphase_sde.model.MasterEquation`

Describes the system in the Hilbert space using the Hamiltonian ($\hat{H}$) and Lindblad collapse operators ($\hat{L}_k$). This is the most fundamental physical description.

*   **Use case:** Defining the physics from first principles.
*   **Components:** Hamiltonian, Lindblad operators.

### Level 2: Phase Space Model (FPE)
**Class:** `qphase_sde.model.PhaseSpaceModel`

Describes the system dynamics in phase space (e.g., Wigner, P-representation, Q-function). It is defined by the Kramers-Moyal expansion coefficients of the Fokker-Planck Equation (FPE).

$$ \frac{\partial P}{\partial t} = \sum_{n=1}^\infty \frac{(-1)^n}{n!} \frac{\partial^n}{\partial \alpha^n} [D_n(\alpha) P] $$

*   **Use case:** Analytical derivation, studying phase space distributions.
*   **Components:** Drift vector ($D_1$), Diffusion tensor ($D_2$), and potentially higher-order terms ($D_3, \dots$).

### Level 3: Stochastic Model (SDE)
**Classes:** `qphase_sde.model.DiffusiveSDEModel`, `qphase_sde.model.JumpSDEModel`

Describes the stochastic trajectories for numerical simulation. This is the level consumed by the simulation engine.

*   **DiffusiveSDEModel (Langevin):** For systems with only 1st and 2nd order terms (Gaussian noise).
    $$ d\mathbf{y} = \mathbf{a}(\mathbf{y}, t) dt + \mathbf{b}(\mathbf{y}, t) d\mathbf{W} $$
*   **JumpSDEModel:** For systems with higher-order terms mapped to jump processes.

## Workflow

1.  **Define Physics:** Start by defining a `PhaseSpaceModel` (Level 2) containing the drift ($D_1$) and diffusion ($D_2$) coefficients.
2.  **Convert:** Use `qphase_sde.model.fpe_to_sde()` to automatically convert the FPE model to a `DiffusiveSDEModel` (Level 3).
3.  **Simulate:** Pass the Level 3 model to the `Engine`.

Alternatively, advanced users can define a `DiffusiveSDEModel` directly if they want manual control over the noise decomposition.

## Example: Van der Pol Oscillator

### Level 2 Definition
```python
from qphase_sde.model import PhaseSpaceModel

def drift_fn(y, t, p):
    # ... calculate D1 ...
    return d1

def diffusion_fn(y, t, p):
    # ... calculate D2 ...
    return d2

fpe_model = PhaseSpaceModel(
    name="vdp_fpe",
    n_modes=1,
    terms={1: drift_fn, 2: diffusion_fn},
    params={...}
)
```

### Conversion
```python
from qphase_sde.model import fpe_to_sde

sde_model = fpe_to_sde(fpe_model)
```
