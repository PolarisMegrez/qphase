# qphase - Modular Quantum Phase-Space Simulation Framework

qphase is a Python framework designed for simulating quantum optical systems using Stochastic Differential Equations (SDEs) in the phase space. It decouples the simulation engine from the model definition, allowing researchers to focus on the physics while the framework handles parameter scanning, parallel execution, and data management. With a plugin-based architecture, it supports various numerical integrators and backend accelerators (NumPy, PyTorch, etc.).

**Authors: Yu Xue-Hao (University of Chinese Academy of Sciences, UCAS)**

- **Configuration-Driven**: Define simulations using declarative YAML files for full reproducibility.
- **Plugin Architecture**: Easily extend the framework with custom models, integrators, and backends.
- **High Performance**: Support for multiple backends including NumPy, PyTorch, and Numba.
- **Automated Workflow**: Built-in tools for parameter scanning, parallel execution, and result management.
- **Type-Safe Design**: Built on robust protocols and Pydantic validation for reliability.

## Installation

Requirements:
- Python >= 3.10

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
pip install packages/qphase
pip install packages/qphase_sde
pip install packages/qphase_viz
```

## Quick Start

To run a simulation, you need a configuration file. Here is a minimal example:

```yaml
# config.yaml
version: "1.0"
engine:
  type: sde
  backend: numpy
  integrator: euler

model:
  type: kerr_cavity
  params:
    chi: 1.0
    delta: 0.0
    kappa: 1.0
    epsilon: 2.0

simulation:
  t_start: 0.0
  t_end: 10.0
  dt: 0.01
  trajectories: 100
```

Run the simulation using the CLI:

```powershell
qps run config.yaml
```

## Project Structure

The project is organized as a monorepo containing three main packages:

- `packages/qphase`: The core framework, CLI, and plugin system.
- `packages/qphase_sde`: The physics engine implementing SDE solvers and models.
- `packages/qphase_viz`: Visualization tools for analyzing simulation results.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.


- `packages/qphase_sde` — core library (engine, protocols, backends, integrators, IO, visualizer)
  - `core/` — minimal protocols, registry, and integration engine
  - `backends/` — backend registrations (NumPy built‑in)
  - `integrators/` — Euler–Maruyama (Milstein alias)
  - `visualizer/` — Spec (Pydantic) → Renderer → Service; PSD and phase portraits
  - `analysis/` — reusable analysis (e.g., PSD computation)
  - `io/` — results saving/loading; snapshots
- `packages/qphase` — Typer‑based CLI (`qps`) for running and analyzing
- `models/` — example models (e.g., `vdp_two_mode.py`)
- `configs/` — example YAMLs (run and PSD)
- `runs/` — generated outputs
- `docs/` — configuration specs and user guides (EN/ZH)
- `tests/` — local smoke tests

Internals rely on a central registry (`qphase_sde.core.registry.registry`) to create pluggable components by name (e.g., `visualizer:psd`). The visualizer service validates specs, slices time windows, merges styles, dispatches plotters, and saves images.

## Notes

- Python ≥ 3.9 recommended. Requires NumPy and Matplotlib. Install `ruamel.yaml` or `PyYAML` to use YAML configs.
- Storage guard: before saving time‑series, the CLI estimates disk usage and aborts if it exceeds a default 1 GiB limit. Override with `--max-storage-gb`.
- Reproducibility: runs are reproducible for the same backend and package version (and device type where applicable). Cross‑backend equivalence is not guaranteed.
- PSD data are saved per IC under `psd/icXX/psd_{kind}_{convention}.npz` when enabled.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.