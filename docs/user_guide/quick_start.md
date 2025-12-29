---
layout: default
title: Quick Start Guide
parent: User Guide
nav_order: 1
---

# Quick Start Guide

This guide will help you set up QPhase and run your first simulation.

## 1. Installation

We strongly recommend using a virtual environment to keep your project dependencies clean.

### Step 1: Create a Virtual Environment
Open your terminal (PowerShell or Bash) and run:

```bash
# Create a virtual environment named '.venv'
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install QPhase
Install the package using pip:

```bash
pip install qphase
```

## 2. Initialize a Project

QPhase works best when you have a dedicated folder for your simulations. This keeps your configuration files and results organized.

```bash
# Create a folder for your research project
mkdir my_research
cd my_research

# Initialize the QPhase structure
qps init
```

This creates the following folders:
*   `configs/`: Where you tell QPhase *what* to run.
*   `plugins/`: Where you put your custom physics code.
*   `runs/`: Where QPhase saves your data.

## 3. Create Your First Job

A "Job" is a single simulation run. You define it in a YAML file.
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
# (Here we use a built-in example model if available, or you can define your own)
# For this example, let's assume we have a 'vdp_oscillator' model available.
# If not, you might need to write a plugin first (see Developer Guide).
plugins:
  model:
    vdp_two_mode:  # This is a built-in example model
      D: 1.0       # Diffusion strength

  backend:
    numpy:         # Run on CPU
      float_dtype: float64
```

## 4. Run the Simulation

Now, tell QPhase to run the job you just defined:

```bash
qps run jobs test_run
```

You should see a progress bar:
```text
[INFO] Loading 1 configuration file(s)
[INFO] Starting job execution
[test_run] 100.0% ~00:00
[INFO] All 1 jobs completed successfully
```

## 5. Check the Results

Look in the `runs/` folder. You will see a new directory with a timestamp:

```text
runs/
└── 2025-12-29T12-00-00Z_test_run/
    ├── config_snapshot.json  # The exact settings used
    └── results.h5            # The simulation data (format depends on engine)
```

**Congratulations!** You have successfully run a reproducible physics simulation without writing any boilerplate code.
