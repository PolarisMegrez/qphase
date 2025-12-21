# qphase-sde

The physics engine for **QPhase**, specializing in Stochastic Differential Equations (SDEs) for quantum optics.

This package implements the core numerical methods for simulating phase-space dynamics, including support for various backends (NumPy, PyTorch, etc.) and integrators (Euler-Maruyama, Milstein).

## Features

- **Multi-Backend Support**: Run simulations on CPU or GPU using NumPy, PyTorch, or CuPy.
- **Stochastic Integrators**: Built-in Euler-Maruyama and Milstein solvers.
- **Quantum Models**: Extensible model interface for defining custom Hamiltonians and dissipators.

## Installation

```bash
pip install qphase-sde
```

## Usage

This package is typically used as a plugin for `qphase`, but can also be imported directly:

```python
from qphase_sde.core.engine import Engine
# ...
```

## License

This project is licensed under the MIT License.

```python
from qphase_sde import run, SDEModel, NoiseSpec
# ... define your model and run simulations ...
```

See the main project documentation for advanced usage and configuration.

## License
MIT
