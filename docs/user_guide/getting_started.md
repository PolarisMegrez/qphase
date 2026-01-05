---
description: Getting Started with QPhase
---

# Getting Started

This guide covers the installation process for QPhase and walks you through running your first simulation.

## 1. Installation

### Prerequisites

*   **Python**: Version 3.10 or higher.
*   **Operating System**: Windows, macOS, or Linux.

### Recommended: Virtual Environment

It is strongly recommended to use a virtual environment to avoid conflicts with other Python packages.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### Install from PyPI (Recommended)

QPhase is available on PyPI. You can install it directly using pip:

```bash
pip install qphase
```

This installs the core framework. **For the examples in this guide, you will also need the SDE (Stochastic Differential Equation) extension:**

```bash
pip install qphase-sde
```

### Install from Source (For Developers)

To contribute to QPhase or use the latest unreleased features:

```bash
git clone https://github.com/PolarisMegrez/qphase.git
cd qphase
pip install -e packages/qphase[standard]
pip install -e packages/qphase_sde
pip install -e packages/qphase_viz
```

## 2. Initialize a Project

QPhase works best when you have a dedicated folder for your simulations. This keeps your configuration files and results organized.

```bash
# Create a folder for your research project
mkdir my_research
cd my_research

# Initialize the QPhase structure
qphase init
```

This creates the following folders:

*   `configs/`: **Directory for configuration files.**
    *   Contains YAML configuration files defining simulations (Jobs) and global settings.
    *   See [Job Configuration](configuration.md) for details on how to write these files.
*   `plugins/`: **Directory for custom physics code.**
    *   A place for user-defined Python modules (Models, Backends, etc.) that QPhase will automatically load.
    *   See [Plugin Development](../dev_guide/plugin_development.md) to learn how to extend QPhase.
*   `runs/`: **Directory for simulation data.**
    *   All simulation outputs, logs, and reproducibility snapshots are organized here by date and run ID.
    *   See [Results & Reproducibility](output.md) for the directory structure.

## 3. Create Your First Job

A "Job" is a single simulation run defined in a YAML file.
Create a new file named `configs/jobs/test_run.yaml` and paste the following:

```yaml
# configs/jobs/test_run.yaml
name: test_run

# 1. Choose the Engine (SDE Solver)
engine:
  sde:
    t_end: 10.0
    dt: 0.01
    n_traj: 100

# 2. Choose the Physics Model
# (Here we use a built-in example model)
plugins:
  model:
    vdp_two_mode:  # Built-in Van der Pol oscillator
      D: 1.0       # Diffusion strength

  backend:
    numpy:         # Run on CPU
      float_dtype: float64
```

## 4. Run the Simulation

Now, tell QPhase to run the job you just defined:

```bash
qphase run test_run
```

You should see a progress bar indicating the simulation status.

## 5. Check the Results

Look in the `runs/` folder. You will see a new directory with a timestamp (e.g., `runs/2026-01-01T12-00-00_test_run/`). Inside, you will find:
*   `config_snapshot.json`: A record of the exact configuration used.
*   `result.h5` (or similar): The simulation output data.

## 6. Verification

To verify your installation and see available plugins:

```bash
qphase --version
qphase list
```
