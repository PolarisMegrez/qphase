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

## Installation

```bash
pip install qphase-sde
```

## Usage

### As a QPhase Plugin
When installed with `qphase`, you can define `sde` jobs in your configuration file:

```yaml
jobs:
  - name: "my_simulation"
    type: "sde"
    config:
      t1: 100.0
      dt: 1e-3
      trajectories: 1000
      model: "models/my_model.py"
```

### Standalone Usage
You can also use the library directly in your Python scripts:

```python
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.model import SDEModel

# Define model and config
config = EngineConfig(dt=1e-3, t1=10.0)
engine = Engine(config)

# Run simulation
result = engine.run(my_model)
```

## License

MIT License
