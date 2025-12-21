---
layout: default
title: Quick Start Guide
parent: User Guide
nav_order: 1
---

# Quick Start Guide

This tutorial will guide you through the complete workflow of using QPhase: from installation to running your first physics simulation and analyzing the results.

## Prerequisites

*   **Python 3.10** or newer.
*   Basic familiarity with command-line interfaces.

## 1. Installation

QPhase is available as a Python package. We recommend installing it in a virtual environment.

### Standard Installation
For most users, the standard installation includes all necessary scientific libraries (NumPy, SciPy, etc.):

```bash
pip install "qphase[standard]"
```

### Minimal Installation
If you want to manage dependencies manually:

```bash
pip install qphase
```

## 2. Project Initialization

QPhase enforces a structured workspace to keep your simulations organized. Let's create a new project directory.

```bash
mkdir my_simulation_project
cd my_simulation_project
qps init
```

The `qps init` command scaffolds the following directory structure:

```text
my_simulation_project/
├── configs/          # Configuration files
│   ├── global.yaml   # Project-wide defaults
│   └── jobs/         # Job definitions (where you define simulations)
├── plugins/          # Custom plugins (models, backends)
└── runs/             # Output directory for simulation results
```

## 3. Running a Demo Simulation

To verify your installation, run the built-in demo job. This runs a simple simulation using the default settings.

```bash
qps run jobs demo
```

**What happens?**
1.  QPhase loads the `demo` job configuration.
2.  It initializes the simulation engine (SDE solver).
3.  It executes the simulation, showing a progress bar.
4.  Results are saved to a timestamped folder in `runs/`.

You should see output similar to:
```text
[INFO] Loading 1 configuration file(s)
[INFO] Starting job execution
[demo] 100.0% ~00:00
[INFO] All 1 jobs completed successfully

Run directories:
  [demo] /path/to/my_simulation_project/runs/2023-10-27T10-00-00Z_a1b2c3d4
```

## 4. Defining Your First Simulation

Simulations are defined in **YAML** files located in `configs/jobs/`. Let's create a simulation for a **Van der Pol oscillator**.

Create a file named `configs/jobs/vdp_sim.yaml`:

```yaml
name: vdp_simulation  # Unique identifier for this job

# Engine Configuration: Controls the time evolution
engine:
  sde:
    t_end: 20.0       # Simulate for 20 time units
    dt: 0.01          # Time step size
    n_trajectories: 100 # Number of parallel stochastic paths

# Plugin Configuration: Selects the components to use
plugins:
  # Backend: Choose 'numpy' (CPU) or 'torch'/'cupy' (GPU)
  backend:
    name: numpy
    params:
      float_dtype: float64

  # Integrator: Numerical method for solving the SDE
  integrator:
    name: srk  # Stochastic Runge-Kutta
  
  # Model: The physics system to simulate
  model:
    name: vdp_oscillator
    params:
      mu: 2.0     # Nonlinearity parameter
      eta: 0.1    # Noise strength
```

## 5. Executing the Job

Run your newly defined job using the `qps run jobs` command:

```bash
qps run jobs vdp_sim
```

QPhase will automatically find `vdp_sim.yaml` in the `configs/jobs/` directory.

## 6. Inspecting Results

Navigate to the output directory printed in the console (e.g., `runs/<timestamp>_<uuid>/`). You will find:

*   `config_snapshot.json`: A complete record of the configuration used for this run (for reproducibility).
*   `vdp_simulation.npz` (or similar): The simulation data file.

You can load this data using Python:

```python
import numpy as np

# Load the results
data = np.load("runs/YOUR_TIMESTAMP/vdp_simulation.npz")
print(data.files)  # See available arrays (e.g., 't', 'x', 'y')
```

## 7. Next Steps

*   **[Configuration Guide](configuration.md)**: Learn how to run parameter scans (e.g., simulate for `mu = [0.5, 1.0, 2.0]` in one go).
*   **[Architecture Overview](../dev_guide/architecture.md)**: Understand how QPhase works under the hood.

