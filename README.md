# qphase - Modular Quantum Phase-Space Simulation Framework

qphase is a high-performance, modular Python framework designed for simulating quantum optical systems using Stochastic Differential Equations (SDEs) in the phase space, decoupling the physics model from the numerical engine to allow researchers to focus on equations while the framework handles parameter scanning, parallel execution, and data management.

**Authors: Yu Xue-Hao (University of Chinese Academy of Sciences, UCAS)**

- Modular architecture separates Core orchestration, Physics Engine, and Visualization tools.
- Plugin-based system supports extensible Backends (NumPy, PyTorch, CuPy), Integrators, and Models.
- Configuration-driven simulations ensure full reproducibility via declarative YAML files.
- High-performance execution is achieved through GPU acceleration and JIT compilation support.
- Automated workflow manages parameter scanning, job scheduling, and result persistence.
- Type-safe design relies on robust Protocols and Pydantic validation for reliability.

## Installation

Requirements:
- Python >= 3.10
- Git

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install from source (editable for development):

```powershell
git clone https://github.com/PolarisMegrez/qphase.git
cd qphase
pip install -r requirements.txt
pip install -e packages/qphase
pip install -e packages/qphase_sde
pip install -e packages/qphase_viz
```

Or install directly from a local checkout without editable mode:

```powershell
pip install packages/qphase packages/qphase_sde packages/qphase_viz
```

## Physical Foundations

The framework solves Stochastic Differential Equations (SDEs) derived from quantum master equations using phase-space representations such as the Positive-P or Wigner distributions. This approach maps quantum dynamics onto classical stochastic trajectories, enabling efficient simulation of high-dimensional systems.

## Module Overview

The project is organized as a monorepo containing three main packages:

- `packages/qphase`: The core framework, CLI, and plugin system.
- `packages/qphase_sde`: The physics engine implementing SDE solvers and models.
- `packages/qphase_viz`: Visualization tools for analyzing simulation results.

## Quick Start

To run a simulation, create a configuration file (e.g., `simulation.yaml`):

```yaml
version: "1.0"
name: "quick_start_demo"

engine:
  sde:
    backend: numpy
    integrator: euler

model:
  vdp_two_mode:
    kappa: 1.0
    chi: 0.5
    pump: 2.0

params:
  t_start: 0.0
  t_end: 10.0
  dt: 0.01
  trajectories: 100

output: "results/demo_run"
```

Run the simulation using the CLI:

```powershell
qps run simulation.yaml
```

## Notes

- Python 3.10 or higher is required.
- Reproducibility is guaranteed for the same backend and package version.
- Storage guard aborts execution if estimated disk usage exceeds the default 1 GiB limit.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.
