# qphase-sde

**SDE Solver for QPhase**

`qphase-sde` is a numerical library for solving Stochastic Differential Equations (SDEs), primarily focused on quantum optics applications. It implements common integration schemes and supports multiple computation backends.

## Features

- **Integrators**:
    - **Euler-Maruyama**: Basic first-order strong approximation.
    - **Milstein**: Higher-order scheme for multiplicative noise.
    - **SRK**: Stochastic Runge-Kutta methods.
- **Backends**:
    - **NumPy**: Standard implementation.
    - **Numba**: JIT-compiled for better CPU performance.
    - **PyTorch/CuPy**: Support for GPU acceleration.
- **Model Definition**:
    - Define custom Hamiltonians and Dissipators via `SDEModel`.
    - Supports additive and multiplicative noise.
- **Logical Parameter Scans**:
    - Accepts the core `ParameterGrid` and compiles it into the existing
      per-trajectory parameter repetition and GPU fusion path.
    - Returns one named-axis SDE dataset with point views and single or sharded
      persistence.

## Installation

```bash
pip install qphase-sde
```

## Usage

### As a QPhase Plugin
When installed with `qphase`, you can define `sde` jobs in your configuration file:

```yaml
name: my_simulation
save: true
scan:
  axes:
    omega_a:
      target: model.vdp_2mode.omega_a
      values: [0.001, 0.01, 0.1]
engine:
  sde: {t0: 0.0, t1: 100.0, dt: 0.01, n_traj: 100, seed: 42}
backend:
  numpy: {}
integrator:
  euler_maruyama: {}
model:
  vdp_2mode:
    omega_a: 0.001
    omega_b: 0.0
    gamma_a: 2.0
    gamma_b: 1.0
    Gamma: 0.0001
    g: 0.5
```

The SDE engine is scheduler-facing; instantiate it through QPhase so plugin
validation, execution context, snapshots, and artifact manifests remain intact.

## License

MIT License
