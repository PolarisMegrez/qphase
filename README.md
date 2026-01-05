# QPhase — A Modular Framework for Phase-Space Representation Based Numerical Simulation

QPhase is a small, research-oriented Python project for running phase-space simulations in quantum optics. The main goal is to reduce repeated “boilerplate” work (configuration, parameter sweeps, and result saving) so you can focus on the model equations.

**Authors: Yu Xue-Hao (University of Chinese Academy of Sciences, UCAS)**

- Modular structure separates the CLI/scheduler, the SDE engine, and plotting utilities.
- Plugin-based design allows different backends and engines to be added over time.
- Configuration-driven runs save a snapshot of the exact configuration used.
- Built-in helpers for parameter sweeps and organized run output.

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

QPhase currently focuses on SDE-based phase-space methods commonly used in quantum optics. The long-term goal is to expand support around phase-space representations and quantum-optics system workflows, while keeping the “Shell vs Kernel” separation (operational tooling vs. physics model).

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

Run the simulation using the CLI (assuming the config is in `configs/jobs/`):

```powershell
qphase run quick_start_demo
```

## Notes

- Python 3.10 or higher is required.
- Runs aim to be reproducible by recording configuration snapshots; numerical details can still depend on backend and library versions.
- Storage guard aborts execution if estimated disk usage exceeds the default 1 GiB limit.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.
