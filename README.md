# QPhase — A Modular Framework for Phase-Space Representation Based Numerical Simulation

QPhase is a small, research-oriented Python project for running phase-space simulations in quantum optics. The main goal is to reduce repeated “boilerplate” work (configuration, parameter sweeps, and result saving) so you can focus on the model equations.

Authors: Yu Xue-Hao (University of Chinese Academy of Sciences, UCAS)

- Modular structure separates the CLI/scheduler, the SDE engine, and plotting utilities.
- Plugin-based design allows different backends and engines to be added over time.
- Configuration-driven runs save a snapshot of the exact configuration used.
- Built-in helpers for parameter sweeps and organized run output.

## Installation

Requirements:

- Python >= 3.11
- Git

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install from source in editable mode:

```powershell
git clone https://github.com/PolarisMegrez/qphase.git
cd qphase
pip install -r requirements.txt
pip install -e packages/qphase
pip install -e packages/qphase_sde
pip install -e packages/qphase_viz
```

## Quick Start

Job files live under `configs/jobs/`. A minimal SDE job uses the scheduler-facing
plugin sections directly:

```yaml
name: demo_psd
save: true

engine:
  sde:
    t0: 0.0
    t1: 10.0
    dt: 0.01
    n_traj: 16
    seed: 42
    ic:
      - ["1.0+0.0j", "0.0+0.0j"]

analyser:
  psd:
    modes: [0]
    kind: complex

backend:
  numpy: {}

integrator:
  euler_maruyama: {}

model:
  kerr_2mode:
    omega_a: 1.0
    omega_b: 1.0
    chi: 0.01
    gamma_a: 0.1
    gamma_b: 0.1
    g: 0.1
```

Run the job by name:

```powershell
qphase run demo_psd
```

The scheduler writes a timestamped run directory under `runs/`. SDE jobs save a
job-named `.npz` file containing `meta`, `analysis`, `t0`, `dt`, and optionally
raw trajectory `data`.

## Result Postprocessing

Cross-job PSD analysis is postprocessed through the scheduler workflow. Add a
second job that consumes the first one and runs the SDE engine in `mode:
analyze`:

```yaml
# configs/jobs/kerr_2mode_fit.yaml
- name: kerr_2mode_sim
  save: true
  engine:
    sde: { t0: 0.0, t1: 1.0, dt: 0.01, n_traj: 8, seed: 42 }
  model:
    kerr_2mode:
      omega_a: [0.9, 1.0, 1.1]
      omega_b: 1.0
      chi: 0.01
      gamma_a: 0.1
      gamma_b: 0.1
      g: 0.1
  analyser:
    psd: { modes: [0], kind: complex, find_peaks: true }

- name: kerr_2mode_fit
  input: kerr_2mode_sim
  aggregate_input:
    on: params.omega_a
  engine:
    sde: { mode: analyze }
  analyser:
    lorentz_fitter:
      scan_param: omega_a
      mode: 0
```

Run the workflow:

```powershell
qphase run configs/jobs/kerr_2mode_fit.yaml
```

The analyzer writes `fit_results.csv` and `psd_merged.csv` to the fit job's run
directory. `fit_results.csv` contains the scan parameter, Lorentzian center,
linewidth, baseline, peak intensity, $R^2$, and status/error fields.

## Module Overview

The project is organized as a monorepo containing three main packages:

- `packages/qphase`: The core framework, CLI, and plugin system.
- `packages/qphase_sde`: The physics engine implementing SDE solvers and models.
- `packages/qphase_viz`: Visualization tools for analyzing simulation results.

## Notes

- Python 3.11 or higher is required.
- Runs aim to be reproducible by recording configuration snapshots; numerical details can still depend on backend and library versions.
- Storage guard aborts execution if estimated disk usage exceeds the default 1 GiB limit.
- `scipy` is required by PSD peak finding and Lorentzian postprocessing.

## Acknowledgements and License

This project was developed with the assistance of Copilot. Licensed under the MIT License. See LICENSE for details.
